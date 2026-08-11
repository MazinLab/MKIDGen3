"""Driver for the stage-2 IQ transform peripheral (``0x8012_0000``).

The transform sits upstream of the phase halfband on the 4-lane 1 MHz stream
and replaces the CORDICs: per DDC sample it computes the two Zobrist
coordinates ``TH2`` and ``D2`` from a per-channel constant row ``c1..c7``.
Software owns those constants; the peripheral owns nothing but the arithmetic.

Register map, AXI-Lite, offsets 0x00-0x18 (handoff 2026-08-10 s1.1). The
window is 64 kB but only these seven registers are documented -- the CSR
bridge is addr_width=8 inside an addr_width=10 wrapper, so anything above
0x18 may alias. Do not touch it.

The 2048x8 constant table is WRITE ONLY (there is no read port; a shadow
would duplicate the block RAM). Whoever writes it keeps the shadow copy --
that is gen3d, which needs it for status, digest, and the offline postage
closure test.

Committing swaps a whole bank, so the inactive bank must always be a
complete, valid table: ``write_table`` writes all 16384 words every time.
Each lane takes the swap at its own next TLAST, so ``pending`` never clears
unless the 512 MHz clocks are running and the pipeline is streaming.
"""
import logging
import time

import numpy as np

try:
    from pynq import DefaultIP
    _PYNQ = True
except Exception:  # pragma: no cover - allows import (and tests) off-board
    DefaultIP = object
    _PYNQ = False

_logger = logging.getLogger(__name__)

CONTROL = 0x00
TABLE_INDEX = 0x04
TABLE_DATA = 0x08
CLAMP_THRESHOLD = 0x0C
VERSION = 0x10
FORMAT = 0x14
FORMAT2 = 0x18

N_CHANNELS = 2048
N_COLUMNS = 8
N_TABLE_WORDS = N_CHANNELS * N_COLUMNS   # 16384
COLUMN_NAMES = ('b1', 'b2', 'a3', 'b4', 'b5', 'a6', 'n', 'spare')

# Reciprocal clamp: when uu = N^2|1-Z|^2 <= uu_min both outputs rail on the
# sign of their numerator. A divide-by-zero guard, not an operating point --
# the smallest uu any modelled photon excursion reaches sits ~8x above it.
CLAMP_THRESHOLD_RESET = 256

# Writable Control bits: bypass[0] and manual_index[4]. commit[1] is
# write-only/self-clearing; pending[2], write_bank[3] and overflow[8:5] are
# read-only status that must never be written back.
_CONTROL_WRITABLE = 0b1_0001

EXPECTED_VERSION = dict(version=1, lanes=4, beat_bits=9, columns=8, latency=24)
EXPECTED_FORMAT = dict(in_frac=15, out_frac_rad=13, c_frac=12, off_frac=4,
                       guard=10, recip_frac=22)
EXPECTED_FORMAT2 = dict(lut_addr_bits=8, mant_bits=24, c_bits=27, off_bits=24,
                        n_bits=16)

# Fabric widths of the table columns (handoff s1.3): the c1/c2/c4/c5 columns
# are 27-bit signed, c3/c6 24-bit signed, c7 (N) 16-bit unsigned, c8 unused.
_C_LIMIT = 1 << 26
_OFF_LIMIT = 1 << 23
_N_LIMIT = 1 << 16


def decode_control(word):
    """Split a Control word into its fields."""
    w = int(word)
    return dict(bypass=bool(w & 1), pending=bool((w >> 2) & 1),
                write_bank=bool((w >> 3) & 1), manual_index=bool((w >> 4) & 1),
                overflow=(w >> 5) & 0xf)


def decode_version(word):
    w = int(word)
    return dict(version=w & 0xff, lanes=(w >> 8) & 0xf,
                beat_bits=(w >> 12) & 0xf, columns=(w >> 16) & 0xf,
                latency=(w >> 20) & 0xff)


def decode_format(word):
    w = int(word)
    return dict(in_frac=w & 0x1f, out_frac_rad=(w >> 5) & 0x1f,
                c_frac=(w >> 10) & 0x1f, off_frac=(w >> 15) & 0x1f,
                guard=(w >> 20) & 0x1f, recip_frac=(w >> 25) & 0x1f)


def decode_format2(word):
    w = int(word)
    return dict(lut_addr_bits=w & 0x1f, mant_bits=(w >> 5) & 0x3f,
                c_bits=(w >> 11) & 0x3f, off_bits=(w >> 17) & 0x3f,
                n_bits=(w >> 23) & 0x3f)


def vet_table(rows):
    """Vet a (2048, 8) constant table and return it as uint32 fabric words.

    Columns are [b1, b2, a3, b4, b5, a6, n, spare]. Integer input of any
    dtype is accepted: uint32 is taken as already two's-complement encoded,
    anything else as signed values. The fabric reads only the low 27/24/16
    bits of each column, so a value out of range would be silently truncated
    into a different constant -- refuse it instead, naming the channel.
    """
    a = np.asarray(rows)
    if a.shape != (N_CHANNELS, N_COLUMNS):
        raise ValueError(f'table must have shape ({N_CHANNELS}, {N_COLUMNS}), '
                         f'got {a.shape}')
    if a.dtype.kind not in 'iu':
        raise ValueError(f'table must be an integer array, got {a.dtype}')

    def _first_bad(mask):
        return int(np.argmax(mask.any(axis=1)))

    if a.dtype == np.uint32:
        signed = a.view(np.int32).astype(np.int64)
    else:
        # int64 cannot hold every uint64 code, and the cast wraps silently:
        # 0xffff_ffff_ffff_ffff would become -1, pass every range check below
        # and be emitted as a legal constant. Range-check before the cast.
        if a.dtype.kind == 'u' and a.dtype.itemsize > 4:
            huge = a > np.uint64(np.iinfo(np.int64).max)
            if huge.any():
                r = _first_bad(huge)
                c = int(np.argmax(huge[r]))
                raise ValueError(f'channel {r} column {COLUMN_NAMES[c]}='
                                 f'{a[r, c]} does not fit in a signed 64-bit '
                                 f'integer, so it cannot be a fabric constant')
        signed = a.astype(np.int64)

    # The fabric contract is a strict magnitude bound, |v| < limit, not a
    # two's-complement range: the most negative code is not available.
    for cols, limit in (((0, 1, 3, 4), _C_LIMIT), ((2, 5), _OFF_LIMIT)):
        block = signed[:, list(cols)]
        bad = (block >= limit) | (block <= -limit)
        if bad.any():
            r = _first_bad(bad)
            c = list(cols)[int(np.argmax(bad[r]))]
            raise ValueError(f'channel {r} column {COLUMN_NAMES[c]}='
                             f'{signed[r, c]} is outside the fabric range '
                             f'({-limit}, {limit})')
    bad = (signed[:, 6:7] < 0) | (signed[:, 6:7] >= _N_LIMIT)
    if bad.any():
        r = _first_bad(bad)
        raise ValueError(f'channel {r} column n={signed[r, 6]} is outside '
                         f'[0, {_N_LIMIT})')
    bad = signed[:, 7:8] != 0
    if bad.any():
        r = _first_bad(bad)
        raise ValueError(f'channel {r} column spare must be 0, got '
                         f'{signed[r, 7]}')
    return (signed & 0xffffffff).astype(np.uint32)


class IQTransform(DefaultIP):
    """Control-bank driver for the stage-2 IQ transform peripheral."""

    bindto = ['xilinx.com:module_ref:iq_transform:1.0']

    N_CHANNELS = N_CHANNELS
    N_COLUMNS = N_COLUMNS

    def __init__(self, description):
        super().__init__(description=description)
        self.check_identity()

    # ---------- identity ----------
    def check_identity(self):
        """Assert the fabric's fixed-point contract against ours.

        A mismatch means the bitstream and this driver disagree about
        quantization, so every constant written would be silently wrong.
        """
        got = {}
        got.update(decode_version(self.read(VERSION)))
        got.update(decode_format(self.read(FORMAT)))
        got.update(decode_format2(self.read(FORMAT2)))
        expect = {}
        expect.update(EXPECTED_VERSION)
        expect.update(EXPECTED_FORMAT)
        expect.update(EXPECTED_FORMAT2)
        bad = {k: (v, expect[k]) for k, v in got.items() if v != expect[k]}
        if bad:
            detail = ', '.join(f'{k}={v} (expected {e})'
                               for k, (v, e) in sorted(bad.items()))
            raise RuntimeError(
                'IQ transform CSR mismatch: ' + detail + '. The bitstream and '
                'this driver disagree about quantization; every constant '
                'written would be silently wrong.')

    # ---------- control ----------
    @property
    def bypass(self):
        """True = raw (I, Q) passed to the (TH2, D2) outputs, same latency."""
        return decode_control(self.read(CONTROL))['bypass']

    @bypass.setter
    def bypass(self, value):
        cur = self.read(CONTROL) & _CONTROL_WRITABLE
        self.write(CONTROL, (cur & ~1) | (1 if value else 0))

    @property
    def pending(self):
        return decode_control(self.read(CONTROL))['pending']

    @property
    def write_bank(self):
        return decode_control(self.read(CONTROL))['write_bank']

    @property
    def manual_index(self):
        return decode_control(self.read(CONTROL))['manual_index']

    @property
    def overflow(self):
        """Sticky per-lane output-queue overflow, bit i = lane i.

        Should read 0 forever; nonzero means the flow-control guard failed
        and the fabric's output is suspect.
        """
        return decode_control(self.read(CONTROL))['overflow']

    @property
    def clamp_threshold(self):
        """Reciprocal clamp uu_min (read-only here; leave it at 256)."""
        return self.read(CLAMP_THRESHOLD)

    def status(self):
        """Every readable field, for gen3d's transform_status command."""
        s = decode_control(self.read(CONTROL))
        s.update(decode_version(self.read(VERSION)))
        s.update(decode_format(self.read(FORMAT)))
        s.update(decode_format2(self.read(FORMAT2)))
        s['clamp_threshold'] = self.clamp_threshold
        return s

    # ---------- constant table ----------
    def write_table(self, rows):
        """Write the whole (2048, 8) constant table into the inactive bank.

        Always all 16384 words: a commit swaps a whole bank, so the inactive
        bank must be complete, not a delta. Flat word index is r*8 + column
        and TableIndex auto-increments, so one index write starts it.
        """
        words = vet_table(rows)
        if self.manual_index:
            cur = self.read(CONTROL) & _CONTROL_WRITABLE
            self.write(CONTROL, cur & ~(1 << 4))
        self.write(TABLE_INDEX, 0)
        write = self.write
        for w in words.reshape(-1).tolist():
            write(TABLE_DATA, int(w))

    def commit(self, timeout_s=1.0):
        """Arm the bank swap and wait for every lane to take it."""
        cur = self.read(CONTROL) & _CONTROL_WRITABLE
        self.write(CONTROL, cur | (1 << 1))
        deadline = time.monotonic() + float(timeout_s)
        while decode_control(self.read(CONTROL))['pending']:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f'IQ transform commit still pending after {timeout_s:.3f} s: '
                    'each lane swaps banks at its own next TLAST, so `pending` '
                    'never clears unless the 512 MHz clocks are running and the '
                    'pipeline is streaming. Program the clocks and start the '
                    'pipeline before committing.')
            time.sleep(0.001)
