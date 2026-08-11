"""Record-version-keyed constants, lane arithmetic and filter geometry.

Two photon-record contracts are live in this driver:

* **v2** -- the production 4-lane build. Visits every channel every 1 us,
  ``lane = r % 4``, 512 beats per frame, holdoff CSR counts 1 us visits.
* **v3** -- the stage-1/stage-2 build. Visits every 2 us (500 kHz),
  ``lane = r % 2`` downstream of the phase halfband, 1024 beats per frame,
  holdoff CSR counts 2 us visits, records carry ``dt`` and ``pileup``.

One wheel serves both bitstreams, so every fork is keyed on the record
version and written down exactly once -- here. This module is numpy-only and
imports without pynq, so all of it is unit-testable off-board.

NOTE the transform peripheral (mkidgen3.drivers.iqtransform) keeps its OWN
4-lane convention on the v3 build: it sits upstream of the halfband on the
4-lane 1 MHz stream, so its constant table is indexed with ``lane = r % 4``
and ``row = r // 4``, which the fabric derives from the flat index itself.
Nothing in this module applies to it.

Source: the stage-2 driver handoff, 2026-08-10, ss3.2-3.3 and s4.
"""
import numpy as np

# --- record version CSR (trigger subsystem, byte offset 0x40) --------------
RECORD_VERSION_OFFSET = 0x40
DEFAULT_RECORD_VERSION = 2
SUPPORTED_RECORD_VERSIONS = (2, 3)
STAGE2_RECORD_VERSION = dict(version=3, lanes=2, beat_bits=10)


def decode_record_version(word):
    """{version[7:0], lanes[11:8], beat_bits[15:12]} from the CSR word."""
    w = int(word)
    return dict(version=w & 0xff, lanes=(w >> 8) & 0xf,
                beat_bits=(w >> 12) & 0xf)


def check_version(version):
    v = int(version)
    if v not in SUPPORTED_RECORD_VERSIONS:
        raise ValueError(f'unsupported record version {v}; this driver '
                         f'implements {SUPPORTED_RECORD_VERSIONS}')
    return v


# --- time and lane constants ----------------------------------------------
CYCLE_NS_BY_VERSION = {2: 1000, 3: 2000}
HOLDOFF_CYCLE_US_BY_VERSION = {2: 1, 3: 2}
TRIGGER_LANES_BY_VERSION = {2: 4, 3: 2}
FRAME_BEATS_BY_VERSION = {2: 512, 3: 1024}


def cycle_ns(version):
    """ns per unit of the record's ``cycle`` field (= one visit)."""
    return CYCLE_NS_BY_VERSION[check_version(version)]


def holdoff_cycle_us(version):
    """Microseconds per count of the 8-bit trigger holdoff field."""
    return HOLDOFF_CYCLE_US_BY_VERSION[check_version(version)]


def trigger_lane(bin, version):
    """Lane a channel's records and postage engine live on."""
    return int(bin) % TRIGGER_LANES_BY_VERSION[check_version(version)]


def record_info(version):
    """The canonical {version, lanes, beat_bits} for a supported version.

    Derived from the tables above rather than restated, so a driver that
    declares a version instead of reading the CSR cannot end up holding a
    ``lanes`` that disagrees with :func:`trigger_lane`. A frame is
    ``2**beat_bits`` beats, which is what ties beat_bits to
    FRAME_BEATS_BY_VERSION.

    Compare with :func:`decode_record_version`, which reports what the
    gateware says it is; the two agreeing is the check that this driver
    matches the bitstream.
    """
    v = check_version(version)
    return dict(version=v, lanes=TRIGGER_LANES_BY_VERSION[v],
                beat_bits=FRAME_BEATS_BY_VERSION[v].bit_length() - 1)


# --- matched filter bank geometry -----------------------------------------
# Four 30-tap banks, 1024 coefficient sets each, two reload slots each,
# routed by TDEST through the reload switch at 0x800E_0000:
#
#   TDEST 0 = TH2 lane 0    TDEST 2 = D2 lane 0
#   TDEST 1 = TH2 lane 1    TDEST 3 = D2 lane 1    TDEST 4 = config broadcast
#
# On a v3 (stage-1/2) build lane = r % 2, set = r // 2, and both quadratures
# of a channel must be reloaded together and committed by the same config
# packet or the two planes briefly carry templates from different loads.
# A v2 build has one quadrature on four lanes: lane = r % 4, set = r // 4.
FIR_TAPS = 30
N_RES = 2048
FIR_LANES_BY_VERSION = {2: 4, 3: 2}
FIR_SETS_BY_VERSION = {2: 512, 3: 1024}
FIR_QUADRATURES_BY_VERSION = {2: ('th2',), 3: ('th2', 'd2')}
FIR_CONFIG_TDEST = 4

# Single-tap unity. The filter's output stage keeps the top 22 bits of a
# 37-bit accumulator (out = acc >> 15), so true unity is a tap of 2**15 --
# and +32768 is not representable in signed 16 bits, so on the stage-2 build
# unity is -32768 and passes the stream through INVERTED. Absorb the sign in
# the template polarity: pulses must arrive negative-going at the trigger.
# The v2 value is the historical one and is left alone.
UNITY_TAP_BY_VERSION = {2: 32767, 3: -32768}


def fir_lane(res_id, version):
    """Filter lane a resonator channel's taps live on."""
    return int(res_id) % FIR_LANES_BY_VERSION[check_version(version)]


def fir_set(res_id, version):
    """Coefficient set index within that lane's bank."""
    return int(res_id) // FIR_LANES_BY_VERSION[check_version(version)]


def fir_tdest(res_id, version, quadrature='th2'):
    """Reload TDEST for one channel's bank in one quadrature."""
    v = check_version(version)
    if quadrature not in FIR_QUADRATURES_BY_VERSION[v]:
        raise ValueError(f'record version {v} has no D2 quadrature: this '
                         f'bitstream carries {FIR_QUADRATURES_BY_VERSION[v]}')
    lane = fir_lane(res_id, v)
    return lane if quadrature == 'th2' else FIR_LANES_BY_VERSION[v] + lane


def pack16_to_32(data):
    """Pack a uint16 sequence into uint32 words, two per word, low half first.

    Widen before shifting: under numpy >= 2 the weak-promotion rules make
    ``np.uint16(y) << 16`` evaluate to 0, which would silently drop the high
    half of every word. An odd-length input leaves its last value alone in
    the low half of a final word (that is what the FIR reload packet, 31
    uint16, relies on, together with last_bytes=2 on the transfer).
    """
    d = np.asarray(data, dtype=np.uint16).astype(np.uint32)
    n_pairs = d.size // 2
    out = np.zeros(n_pairs + (d.size % 2), dtype=np.uint32)
    if n_pairs:
        out[:n_pairs] = d[0:2 * n_pairs:2] | (d[1:2 * n_pairs:2] << 16)
    if d.size % 2:
        out[-1] = d[-1]
    return out


def fir_config_packet(version):
    """The channel-sequence packet that commits pending reloads (TDEST 4)."""
    v = check_version(version)
    return pack16_to_32(np.arange(FIR_SETS_BY_VERSION[v], dtype=np.uint16))


def fir_reload_packet(res_id, taps, version):
    """Build one reload packet: set number, then the 30 taps reversed.

    ``taps`` are already in coefficient word form (plain signed 16-bit
    integers -- the FIRs are configured with Coefficient_Fractional_Bits 0,
    so the IP does no rescaling). Send with ``last_bytes=2``.
    """
    v = check_version(version)
    t = np.asarray(taps)
    if t.size != FIR_TAPS:
        raise ValueError(f'need exactly {FIR_TAPS} taps, got {t.size}')
    words = np.zeros(FIR_TAPS + 1, dtype=np.uint16)
    words[0] = fir_set(res_id, v)
    words[1:] = (t.astype(np.int64)[::-1] & 0xffff).astype(np.uint16)
    return pack16_to_32(words)


def unity_coefficient_sets(n, version, n_res=N_RES):
    """(n_res, 30) int16 taps: unity on the first ``n`` channels, else zero."""
    v = check_version(version)
    c = np.zeros((int(n_res), FIR_TAPS), dtype=np.int16)
    c[:int(n), 0] = UNITY_TAP_BY_VERSION[v]
    return c
