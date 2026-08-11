"""
Driver for the amaranth photon trigger subsystem ("benexperiment" gateware).

This targets the module-reference IP ``xilinx.com:module_ref:trigger_subsystem:1.0``
(the amaranth mkidaranth trigger merged in gen3-vivado-top PR #8 and extended on
the benexperiment branch with the phase baseline filter and the v2 photon
record). It is intentionally distinct from :mod:`mkidgen3.drivers.trigger`,
which drives the older HLS trigger IPs (``mazinlab:mkidgen3:trigger:0.4`` /
``photon_maxi:0.2``) with a 64-bit ``(time, id, phase)`` record. The two
drivers coexist; pynq picks by VLNV.

Photon event record v2 (one little-endian 128-bit word per photon)::

    phase   [15:0]    int16  peak phase
    baseline[31:16]   int16  channel baseline at trigger time (same units as
                             phase; corrected pulse height = phase - baseline)
    cycle   [75:32]   uint44 ABSOLUTE visit count, 2 us units (~407 d rollover)
    bin     [86:76]   uint11 resonator channel
    read    [88:87]   uint2  chunk read tag (drop detection only)
    x       [100:89]  uint12 beammap x (reserved, reads 0)
    y       [112:101] uint12 beammap y (reserved, reads 0)
    pad     [127:113] zero

Chunks: the trigger DMA writes fixed-size chunks of packed events to buffers
whose physical addresses software pushes into an address FIFO. Reading the
chunk header CSR returns the PPS timestamp and absolute cycle sampled at the
read, plus sticky dropped/fault/empty flags, AND (side effect) clears those
latches and advances the 2-bit read tag. Wall-clock time for every photon
follows from one affine map per capture::

    t_ns(photon) = header_time_ns + (photon.cycle - header.cycle) * 2000

The register map is generated from the amaranth CSR decoder
(rtl/mkidaranth in gen3-vivado-top, branch benexperiment). Word offsets on
``s_axi_ctrl``: Trigger bank at 0, TriggerDMA at 4096, PostageDMA at 4352.

Operational notes (first light, 2026-07-22):

* The DMAControl flush bit is a NO-OP in the current gateware: a partial
  chunk cannot be force-completed and a queued DMA address cannot be
  retired from software. ``capture_photons``/``capture_postage`` handle
  timeouts losslessly instead, by sentinel-prefilling buffers, recovering
  whatever landed, and parking still-queued buffers on the driver to be
  resumed by the next capture call (see ``CaptureTimeout``). True
  time-rotated (bounded-latency) chunking needs gateware flush support.
* ``capture_postage`` drives the per-lane postage engines; stamps are raw
  unlabeled IQ snippets (the RTL's stamp metadata never reaches memory).
* Trigger holdoff counts visit cycles, and a visit is 1 us on a v2 build and
  2 us on a v3 (stage-1/2) build. ``us_to_holdoff(us, version)`` converts and
  takes the version explicitly -- there is no safe default.
"""
import logging
import math
import time

import numpy as np

try:
    from pynq import DefaultIP, allocate
    _PYNQ = True
except Exception:  # pragma: no cover - allows import (and unpacker tests) off-board
    DefaultIP = object
    allocate = None
    _PYNQ = False

from mkidgen3.recordfmt import (DEFAULT_RECORD_VERSION, RECORD_VERSION_OFFSET,
                                SUPPORTED_RECORD_VERSIONS, cycle_ns,
                                decode_record_version, holdoff_cycle_us,
                                record_info, trigger_lane)

_logger = logging.getLogger(__name__)

# Time constants are keyed on the record version, not fixed: a v2 build
# visits every channel every 1 us, a v3 (stage-1/2) build every 2 us. The
# constants that used to live here (CYCLE_NS = 2000, HOLDOFF_CYCLE_US = 2)
# were right for v3 and wrong for v2 by a factor of two; see
# mkidgen3.recordfmt.

# Capture buffers are prefilled with this before being queued: real v2
# records always have zero pad bits [127:113], so an all-ones hi word can
# never be written by the gateware and marks not-yet-DMA'd space.
SENTINEL_U64 = np.uint64(0xffff_ffff_ffff_ffff)

# Postage stamps: raw IQ snippets, one 32-bit word per sample (real[15:0],
# imag[31:16]), 128 samples per stamp at the 2 us visit rate (256 us window,
# 8 samples of pre-trigger lead-in). One stamp = exactly one 512-byte DMA
# burst. NOTE (gateware limitation): the per-stamp metadata (bin/cycle/read)
# is generated in the RTL but never routed to memory, so stamps in a capture
# carry no channel labels.
POSTAGE_SAMPLES = 128
POSTAGE_PRETRIGGER = 8
POSTAGE_STAMP_BYTES = POSTAGE_SAMPLES * 4
POSTAGE_MAX_PER_LANE = 4

PHOTON_V2_PACKED_DTYPE = np.dtype([('lo', '<u8'), ('hi', '<u8')])
PHOTON_V2_DTYPE = np.dtype([('cycle', '<u8'), ('phase', '<i2'), ('baseline', '<i2'),
                            ('id', '<u2'), ('read', 'u1'), ('x', '<u2'), ('y', '<u2')])


def unpack_photons_v2(packed, out=None, n=0):
    """Unpack v2 128-bit photon records.

    packed: array of PHOTON_V2_PACKED_DTYPE, or a uint64 array of even length
    (lo, hi interleaved), or raw bytes. Returns a PHOTON_V2_DTYPE array,
    optionally accumulating into ``out`` at offset ``n`` (mirroring
    mkidpynq.unpack_photons).
    """
    x = np.asarray(packed)
    if x.dtype != PHOTON_V2_PACKED_DTYPE:
        x = np.frombuffer(x.tobytes(), dtype=PHOTON_V2_PACKED_DTYPE)
    lo, hi = x['lo'], x['hi']

    if out is None:
        n = 0
        ret = np.zeros(x.shape[0], dtype=PHOTON_V2_DTYPE)
    else:
        if x.shape[0] + n > out.shape[0]:
            raise IndexError('Output array is too small')
        ret = out
    sl = slice(n, n + x.shape[0])
    ret['phase'][sl] = (lo & 0xffff).astype(np.uint16).view(np.int16)
    ret['baseline'][sl] = ((lo >> 16) & 0xffff).astype(np.uint16).view(np.int16)
    ret['cycle'][sl] = (lo >> 32) | ((hi & 0xfff) << 32)
    ret['id'][sl] = (hi >> 12) & 0x7ff
    ret['read'][sl] = (hi >> 23) & 0x3
    ret['x'][sl] = (hi >> 25) & 0xfff
    ret['y'][sl] = (hi >> 37) & 0xfff
    return ret


def pack_photons_v2(photons, out=None):
    """Inverse of unpack_photons_v2 (testing / simulation)."""
    p = photons
    ret = np.zeros(p.shape[0], dtype=PHOTON_V2_PACKED_DTYPE) if out is None else out
    cycle = p['cycle'].astype(np.uint64)
    ret['lo'] = (p['phase'].astype(np.int64).view(np.uint64) & 0xffff) \
        | ((p['baseline'].astype(np.int64).view(np.uint64) & 0xffff) << 16) \
        | ((cycle & 0xffffffff) << 32)
    ret['hi'] = ((cycle >> 32) & 0xfff) \
        | ((p['id'].astype(np.uint64) & 0x7ff) << 12) \
        | ((p['read'].astype(np.uint64) & 0x3) << 23) \
        | ((p['x'].astype(np.uint64) & 0xfff) << 25) \
        | ((p['y'].astype(np.uint64) & 0xfff) << 37)
    return ret


def photon_times_ns(photons, header, version):
    """Absolute time in ns for each unpacked photon, given a chunk header.

    ``version`` is the record version (2 or 3): one ``cycle`` count is 1 us
    on v2 and 2 us on v3.
    """
    return header['time_ns'] + (photons['cycle'].astype(np.int64)
                                - int(header['cycle'])) * cycle_ns(version)


def valid_photon_prefix(packed):
    """Number of leading records in ``packed`` that are real events.

    ``packed`` is a (possibly partial) capture buffer that was prefilled with
    ``SENTINEL_U64``; events are DMA'd strictly in order, so the valid events
    are exactly the prefix before the first sentinel record.
    """
    x = np.asarray(packed)
    if x.dtype != PHOTON_V2_PACKED_DTYPE:
        x = np.frombuffer(x.tobytes(), dtype=PHOTON_V2_PACKED_DTYPE)
    s = x['hi'] == SENTINEL_U64
    return int(np.argmax(s)) if s.any() else x.shape[0]


def unpack_postage(raw, n_stamps=None):
    """Unpack raw postage DMA memory into a complex64 array (n, 128).

    raw: uint32 array or bytes of packed IQ words (real[15:0], imag[31:16],
    int16 each). Any trailing partial stamp is dropped; ``n_stamps`` clips
    the result.
    """
    x = np.asarray(raw)
    if x.dtype != np.uint32:
        x = np.frombuffer(x.tobytes(), dtype=np.uint32)
    n = x.size // POSTAGE_SAMPLES
    if n_stamps is not None:
        n = min(n, int(n_stamps))
    w = x[:n * POSTAGE_SAMPLES]
    real = (w & 0xffff).astype(np.uint16).view(np.int16).astype(np.float32)
    imag = (w >> 16).astype(np.uint16).view(np.int16).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64).reshape(n, POSTAGE_SAMPLES)


def us_to_holdoff(us, version):
    """Convert a trigger holdoff in microseconds to visit cycles.

    One cycle is 1 us on a v2 build and 2 us on a v3 build, so the version
    is required. Rounds up (the holdoff is never shorter than requested) and
    clips to the 8-bit field.
    """
    return int(min(255, max(0, math.ceil(us / holdoff_cycle_us(version)))))


class CaptureTimeout(TimeoutError):
    """A capture did not complete in time.

    Whatever DID land is attached (``photons``/``headers`` for photon
    captures, ``stamps`` for postage). Nothing is lost: buffers whose DMA
    addresses remain queued in the gateware are parked on the driver and
    resume as the start of the next capture call. (The DMAControl flush bit
    is a NO-OP in the current gateware - the field exists but drives no
    logic - so software can neither force a partial chunk to complete nor
    retire a queued address; parking is the only safe recovery.)
    """

    def __init__(self, msg, photons=None, headers=None, stamps=None):
        super().__init__(msg)
        self.photons = photons
        self.headers = headers
        self.stamps = stamps


class _Reg:
    """Word offsets (into the s_axi_ctrl bank) from the amaranth CSR map."""
    CHUNK_SAMPLER = 0        # 4 words: chunk_header, 121 bits, READ HAS SIDE EFFECTS
    TRIGGER_CONTROL = 4      # 2 words: input_gate[0], config[37:1]
    POSTAGE_CONTROL = 6      # 2 words
    VALVE_CONTROL = 8        # trigger[1:0], cuber[3:2], stamper[5:4]
    VALVE_STATUS = 9
    INTERRUPT_STATUS = 10    # dropped, dropped_postage, fault, fault_postage, halfchunk, fullchunk
    INTERRUPT_ENABLE = 11
    BASELINE_CONTROL = 12    # enable[0], shift[4:1]
    BASELINE_HOLDOFF = 13    # n[15:0], us units
    BASELINE_READ = 14       # bin[10:0]
    BASELINE_VALUE = 15      # baseline[15:0], RO, <=2us stale
    RECORD_VERSION = 16      # byte offset 0x40: version[7:0], lanes[11:8],
                             # beat_bits[15:12]. Appended after every other
                             # trigger CSR, so no earlier offset moved.
    TRIG_DMA = 4096
    POSTAGE_DMA = 4352
    # HuskyDMA bank layout (relative to bank base)
    DMA_ADDRESS_FIFO = 0     # 4 words: address[47:0], depth[55:48], count[63:56], lowmark[71:64]
    DMA_CONTROL = 4          # buffer_size[23:0], flush[24], fault[25]
    DMA_INPUT_FIFO = 6       # 2 words
    DMA_DEBUG = 8            # 2 words


class ValvePosition:
    OPEN = 0b00
    CLOSED = 0b01
    DUMP = 0b10


class _HuskyDMA:
    """One chunked DMA engine (TriggerDMA or PostageDMA bank)."""

    def __init__(self, parent, base_word):
        self._p = parent
        self._b = base_word

    def _rd_multi(self, word, n):
        v = 0
        for i in range(n):
            v |= self._p.read((self._b + word + i) * 4) << (32 * i)
        return v

    @property
    def buffer_size(self):
        return self._rd_multi(_Reg.DMA_CONTROL, 1) & 0xffffff

    @property
    def fault(self):
        return bool(self._rd_multi(_Reg.DMA_CONTROL, 1) >> 25 & 1)

    @property
    def fifo_count(self):
        v = self._rd_multi(_Reg.DMA_ADDRESS_FIFO, 4)
        return (v >> 56) & 0xff

    @property
    def fifo_depth(self):
        v = self._rd_multi(_Reg.DMA_ADDRESS_FIFO, 4)
        return (v >> 48) & 0xff

    def fifo_ready(self):
        v = self._rd_multi(_Reg.DMA_ADDRESS_FIFO, 4)
        return ((v >> 56) & 0xff) < ((v >> 48) & 0xff)

    def completed(self):
        """True when no queued buffer remains and the engine is idle at a wait state."""
        if self.fifo_count != 0:
            return False
        dbg = self._rd_multi(_Reg.DMA_DEBUG, 2)
        address_wait = (dbg >> 14) & 1
        data_wait = (dbg >> 13) & 1
        return bool(address_wait or data_wait)

    @property
    def burst_count(self):
        """Free-running 16-bit count of completed DMA bursts (Debug reg).

        One burst is 2048 bytes (128 events) on the trigger engine and
        512 bytes (one postage stamp) on the postage engine. Take deltas
        modulo 2**16 to count data landed in memory.
        """
        return (self._rd_multi(_Reg.DMA_DEBUG, 2) >> 21) & 0xffff

    def push_buffer(self, buf, timeout=1.0):
        """Queue a pynq buffer (its full nbytes = one chunk) for DMA fill.

        Changing chunk size while buffers are in flight waits for the engine
        to drain first, mirroring the verified simulation harness sequence.
        """
        if buf.nbytes != self.buffer_size:
            t0 = time.time()
            while not self.completed():
                if time.time() - t0 > timeout:
                    raise TimeoutError('DMA engine did not drain before buffer_size change')
        self._p.write((self._b + _Reg.DMA_CONTROL) * 4, buf.nbytes & 0xffffff)
        t0 = time.time()
        while not self.fifo_ready():
            if time.time() - t0 > timeout:
                raise TimeoutError('DMA address FIFO full')
        addr = int(buf.physical_address) | (1 << 64)  # lowmark=1 in bits [71:64]
        for i in range(4):
            self._p.write((self._b + _Reg.DMA_ADDRESS_FIFO + i) * 4,
                          (addr >> (32 * i)) & 0xffffffff)

    def flush(self):
        """Set the DMAControl flush bit.

        WARNING: this is a NO-OP in the current gateware. The flush field
        exists in the register map but is not connected to any logic in
        AXIDMA.elaborate - it cannot force a partial chunk to complete or
        retire a queued address. Kept for forward compatibility with a
        future gateware that implements it; do not rely on it for recovery
        (use the capture methods' buffer parking instead).
        """
        self._p.write((self._b + _Reg.DMA_CONTROL) * 4,
                      (self.buffer_size & 0xffffff) | (1 << 24))


class TriggerSubsystemV2(DefaultIP):
    """Control-bank driver for the amaranth trigger subsystem.

    pynq instantiates one object per mapped slave interface; the full driver
    surface is only meaningful on the ``s_axi_ctrl`` segment (trigger CSRs +
    both DMA engines). The cuber banks (``s_axi_ctrl_slow``/``s_axi_cube``)
    get a thin instance you can poke with read()/write() directly.
    """

    bindto = ['xilinx.com:module_ref:trigger_subsystem:1.0']

    N_CHANNELS = 2048
    EVENT_BYTES = 16

    def __init__(self, description):
        super().__init__(description=description)
        self._init_state()

    def _init_state(self):
        """Driver-side state, separated from DefaultIP construction."""
        self.trigger_dma = _HuskyDMA(self, _Reg.TRIG_DMA)
        self.postage_dma = _HuskyDMA(self, _Reg.POSTAGE_DMA)
        # Buffers whose DMA addresses are still queued in the gateware after
        # a timed-out capture (flush is a gateware no-op, so they cannot be
        # retired). Oldest first; entries are (pynq_buffer, n_already_taken).
        # The next capture resumes them as its first chunks/stamps.
        self._inflight = []
        self._postage_inflight = None
        # Shadow of the last programmed per-channel trigger config, so
        # capture_postage can toggle the postage flag without callers
        # restating threshold/holdoff.
        self._thr_shadow = np.zeros(self.N_CHANNELS, np.int32)
        self._hold_shadow = np.zeros(self.N_CHANNELS, np.int32)
        self._en_shadow = np.zeros(self.N_CHANNELS, bool)
        self._shadow_valid = np.zeros(self.N_CHANNELS, bool)
        # Record layout in force. Stays at the v2 default until somebody who
        # knows the overlay carries the stage-1/2 trigger probes the CSR:
        # on older builds offset 0x40 is not a RecordVersion register.
        self._record_version_info = None

    # ---------- record version ----------
    @property
    def record_version(self):
        """Photon record version in force (2 unless probed or set)."""
        if self._record_version_info is None:
            return DEFAULT_RECORD_VERSION
        return self._record_version_info['version']

    @property
    def record_version_info(self):
        """{version, lanes, beat_bits} once known, else None."""
        return self._record_version_info

    def probe_record_version(self):
        """Read the RecordVersion CSR (byte offset 0x40) and cache it.

        Only call this on an overlay already known to carry the stage-1/2
        trigger -- established from the hwh (the IQ transform peripheral is
        present), never by probing the bus blind.
        """
        word = self.read(RECORD_VERSION_OFFSET)
        info = decode_record_version(word)
        if info['version'] not in SUPPORTED_RECORD_VERSIONS:
            raise RuntimeError(
                f'RecordVersion CSR reads {word:#010x} -> record version '
                f'{info["version"]}, lanes {info["lanes"]}, beat_bits '
                f'{info["beat_bits"]}; this driver implements '
                f'{SUPPORTED_RECORD_VERSIONS}')
        self._record_version_info = info
        return info

    def set_record_version(self, version):
        """Declare the record version without touching the bus.

        The geometry comes from mkidgen3.recordfmt and is never restated
        here: a second copy of {lanes, beat_bits} could drift out of step
        with trigger_lane() and nothing would notice. Raises ValueError on an
        unsupported version.
        """
        self._record_version_info = record_info(version)

    # ---------- chunk header ----------
    def read_chunk_header(self):
        """Read (and CLEAR - this advances the read tag and clears the sticky
        dropped/fault/empty latches) the chunk sampler.

        Returns a dict with secs/ns/subns (PPS time), time_ns, absolute
        cycle, next read tag, and the dropped/fault/empty flags accumulated
        since the previous header read.
        """
        w = [self.read((_Reg.CHUNK_SAMPLER + i) * 4) for i in range(4)]
        v = w[0] | (w[1] << 32) | (w[2] << 64) | (w[3] << 96)
        secs = v & 0xffffffff
        ns = (v >> 32) & 0xffffffff
        subns = (v >> 64) & 0xff
        cycle = (v >> 72) & ((1 << 44) - 1)
        read = (v >> 116) & 0x3
        return dict(secs=secs, ns=ns, subns=subns,
                    time_ns=secs * 1_000_000_000 + ns + subns / 256,
                    cycle=cycle, read=read,
                    dropped=bool((v >> 118) & 1),
                    fault=bool((v >> 119) & 1),
                    empty=bool((v >> 120) & 1))

    # ---------- trigger configuration ----------
    def configure_channel(self, bin, threshold, holdoff, postage=False, enabled=True):
        """Program one channel. threshold is a raw signed 16-bit phase,
        holdoff in visit cycles (0-255). A visit is 1 us on a v2 build and
        2 us on a v3 build; use us_to_holdoff(us, version) to convert."""
        b = int(bin) & 0x7ff
        self._thr_shadow[b] = int(threshold)
        self._hold_shadow[b] = int(holdoff)
        self._en_shadow[b] = bool(enabled)
        self._shadow_valid[b] = True
        cw = (int(bin) & 0x7ff) \
            | ((int(threshold) & 0xffff) << 11) \
            | ((int(holdoff) & 0xff) << 27) \
            | ((1 << 35) if postage else 0) \
            | ((1 << 36) if enabled else 0)
        val = (self.input_gate & 1) | (cw << 1)
        self.write(_Reg.TRIGGER_CONTROL * 4, val & 0xffffffff)
        self.write((_Reg.TRIGGER_CONTROL + 1) * 4, (val >> 32) & 0xffffffff)

    def configure(self, thresholds, holdoffs, enabled=True, postage_channels=()):
        """Program all 2048 channels. thresholds: raw int16 array; holdoffs:
        cycles array; enabled: bool or per-channel bool array."""
        thresholds = np.asarray(thresholds).astype(int)
        holdoffs = np.asarray(holdoffs).astype(int)
        if thresholds.size != self.N_CHANNELS or holdoffs.size != self.N_CHANNELS:
            raise ValueError(f'need {self.N_CHANNELS} thresholds and holdoffs')
        en = np.broadcast_to(np.asarray(enabled, dtype=bool), (self.N_CHANNELS,))
        pset = set(int(p) for p in postage_channels)
        for b in range(self.N_CHANNELS):
            self.configure_channel(b, thresholds[b], holdoffs[b],
                                   postage=b in pset, enabled=bool(en[b]))

    @property
    def input_gate(self):
        return self.read(_Reg.TRIGGER_CONTROL * 4) & 1

    @input_gate.setter
    def input_gate(self, value):
        self.write(_Reg.TRIGGER_CONTROL * 4, 1 if value else 0)
        self.write((_Reg.TRIGGER_CONTROL + 1) * 4, 0)

    # ---------- valves ----------
    @property
    def valve_status(self):
        v = self.read(_Reg.VALVE_STATUS * 4)
        return dict(trigger=v & 3, cuber=(v >> 2) & 3, stamper=(v >> 4) & 3)

    def set_valves(self, trigger=None, cuber=None, stamper=None):
        cur = self.read(_Reg.VALVE_CONTROL * 4)
        for shift, val in ((0, trigger), (2, cuber), (4, stamper)):
            if val is not None:
                cur = (cur & ~(3 << shift)) | ((val & 3) << shift)
        self.write(_Reg.VALVE_CONTROL * 4, cur)

    # ---------- interrupts ----------
    @property
    def interrupt_status(self):
        v = self.read(_Reg.INTERRUPT_STATUS * 4)
        keys = ('dropped', 'dropped_postage', 'fault', 'fault_postage', 'halfchunk', 'fullchunk')
        return {k: bool((v >> i) & 1) for i, k in enumerate(keys)}

    def enable_interrupts(self, **kw):
        keys = ('dropped', 'dropped_postage', 'fault', 'fault_postage', 'halfchunk', 'fullchunk')
        v = self.read(_Reg.INTERRUPT_ENABLE * 4)
        for i, k in enumerate(keys):
            if k in kw:
                v = (v & ~(1 << i)) | (int(bool(kw[k])) << i)
        self.write(_Reg.INTERRUPT_ENABLE * 4, v)

    # ---------- baseline filter ----------
    @property
    def baseline_enabled(self):
        return bool(self.read(_Reg.BASELINE_CONTROL * 4) & 1)

    def configure_baseline(self, enable=True, shift=10, holdoff_us=0):
        """Enable the per-channel baseline tracker.

        shift: IIR pole, alpha = 2**-shift at the 1 MHz visit rate
        (shift=10 ~ 155 Hz single pole). holdoff_us: extra post-pulse gating
        beyond the trigger holdoff, for qp recombination tails.
        """
        self.write(_Reg.BASELINE_HOLDOFF * 4, int(holdoff_us) & 0xffff)
        self.write(_Reg.BASELINE_CONTROL * 4,
                   (1 if enable else 0) | ((int(shift) & 0xf) << 1))

    def baseline(self, bin):
        """Spot-read one channel's current baseline (<= 2 us stale)."""
        self.write(_Reg.BASELINE_READ * 4, int(bin) & 0x7ff)
        v = self.read(_Reg.BASELINE_VALUE * 4) & 0xffff
        return int(np.int16(v))

    # ---------- photon capture ----------
    @property
    def parked_chunks(self):
        """Number of chunk buffers parked by a previous timed-out capture."""
        return len(self._inflight)

    def capture_photons(self, n_chunks=8, events_per_chunk=4096, timeout=10.0,
                        with_headers=False, partial='raise'):
        """Synchronous double-buffered capture of n_chunks chunks.

        Returns a PHOTON_V2_DTYPE array (and the list of chunk-header dicts
        if with_headers). The trigger valve must be OPEN and channels
        configured. events_per_chunk must be a multiple of 128 (one DMA
        burst); a chunk only completes when the gateware has written that
        many events into it.

        Low count rate / timeout behavior: a chunk that never fills cannot
        be force-completed (the gateware flush bit is a no-op), but nothing
        is lost. Buffers are sentinel-prefilled, so on timeout the events
        that already landed in the current chunk are recovered, and buffers
        whose DMA addresses are still queued in the gateware are parked on
        the driver (see parked_chunks) - the next capture_photons call
        resumes them as its first chunks, deduplicating anything already
        recovered. With partial='raise' (default) a CaptureTimeout (a
        TimeoutError subclass) carries the recovered .photons/.headers;
        with partial='return' they are returned normally and a warning is
        logged. On timeout the last header in headers is read at recovery
        time and timestamps the partial chunk.
        """
        if not _PYNQ:
            raise RuntimeError('pynq not available')
        if partial not in ('raise', 'return'):
            raise ValueError("partial must be 'raise' or 'return'")
        if (events_per_chunk * self.EVENT_BYTES) % 2048 or events_per_chunk <= 0 \
                or events_per_chunk * self.EVENT_BYTES >= (1 << 24):
            raise ValueError('events_per_chunk must be a positive multiple of 128 '
                             'with chunk size under 16 MB')
        queued = list(self._inflight)   # [(buf, events_already_taken)]
        self._inflight = []
        if queued and queued[0][0].shape[0] != events_per_chunk:
            events_per_chunk = queued[0][0].shape[0]
            _logger.warning('resuming %d parked chunk buffer(s): events_per_chunk '
                            'coerced to %d', len(queued), events_per_chunk)
        resumed = [b for b, _ in queued]   # allocated by an earlier call
        fresh = []   # buffers allocated by this call
        free = []    # completed buffers available for requeue
        out, headers = [], []

        def _queue_target():
            b = free.pop() if free else None
            if b is None:
                b = allocate(shape=(events_per_chunk, 2), dtype=np.uint64)
                fresh.append(b)
            b[:] = SENTINEL_U64
            b.flush()
            self.trigger_dma.push_buffer(b)
            queued.append((b, 0))

        try:
            try:
                self.read_chunk_header()  # clear latches, learn starting read tag
                for i in range(n_chunks):
                    while len(queued) < min(2, n_chunks - i):
                        _queue_target()
                    cur, skip = queued[0]
                    t0 = time.time()
                    while self.trigger_dma.fifo_count > len(queued) - 1 \
                            or not (self.interrupt_status['fullchunk']
                                    or self.trigger_dma.completed()):
                        if time.time() - t0 > timeout:
                            raise CaptureTimeout(
                                f'chunk {i} did not complete '
                                f'(status={self.interrupt_status})')
                        time.sleep(0.001)
                    queued.pop(0)
                    cur.invalidate()
                    hdr = self.read_chunk_header()
                    headers.append(hdr)
                    if hdr['fault']:
                        _logger.warning('trigger DMA fault flagged in chunk %d', i)
                    if hdr['dropped']:
                        _logger.warning('events dropped in chunk %d', i)
                    out.append(unpack_photons_v2(np.array(cur))[skip:])
                    free.append(cur)
            except TimeoutError as e:
                # Recover events that already landed in the head buffer; the
                # buffers stay queued in the gateware and are parked in the
                # finally block below.
                n_partial = 0
                if queued:
                    cur, skip = queued[0]
                    cur.invalidate()
                    k = valid_photon_prefix(np.array(cur))
                    if k > skip:
                        out.append(unpack_photons_v2(np.array(cur)[:k])[skip:])
                        n_partial = k - skip
                        queued[0] = (cur, k)
                    headers.append(self.read_chunk_header())
                photons = (np.concatenate(out) if out
                           else np.zeros(0, dtype=PHOTON_V2_DTYPE))
                _logger.warning(
                    'capture timed out after %d/%d chunks; recovered %d events '
                    'from the partial chunk; %d buffer(s) parked for the next '
                    'capture', len(out) - (1 if n_partial else 0), n_chunks,
                    n_partial, len(queued))
                if partial == 'return':
                    return (photons, headers) if with_headers else photons
                raise CaptureTimeout(str(e), photons=photons,
                                     headers=headers) from None
            photons = np.concatenate(out) if out else np.zeros(0, dtype=PHOTON_V2_DTYPE)
            return (photons, headers) if with_headers else photons
        finally:
            # Park buffers whose addresses the gateware still holds; free the
            # rest. Parked buffers must outlive this call - the DMA will
            # eventually write to them.
            self._inflight = queued
            parked = set(id(b) for b, _ in queued)
            for b in fresh + resumed:
                if id(b) not in parked:
                    b.freebuffer()

    # ---------- postage capture ----------
    def _postage_control_write(self, lane_counts):
        """Write PostageControl.count (3 bits per lane, 4 lanes). Both CSR
        words must be written: amaranth-soc commits a multi-word register on
        the write of its last word."""
        w = 0
        for lane, n in lane_counts.items():
            w |= (int(n) & 0x7) << (3 * int(lane))
        self.write(_Reg.POSTAGE_CONTROL * 4, w)
        self.write((_Reg.POSTAGE_CONTROL + 1) * 4, 0)

    def _postage_status(self):
        """Read PostageControl status bits.

        NOTE: reading word 0 latches the register shadow and clears the
        per-lane dropped bits (read side effect in the gateware).
        """
        w0 = self.read(_Reg.POSTAGE_CONTROL * 4)
        w1 = self.read((_Reg.POSTAGE_CONTROL + 1) * 4)
        v = (w0 & 0xffffffff) | ((w1 & 0xffffffff) << 32)
        return dict(dropped=(v >> 12) & 0xffff, fault=(v >> 28) & 0xffff,
                    flushed=(v >> 44) & 0xf)

    def capture_postage(self, bins, n_stamps=16, timeout=10.0,
                        threshold=None, holdoff=None, partial='raise'):
        """Capture postage stamps (raw IQ waveform snippets) around triggers.

        Each stamp is 128 complex samples of one channel at the 2 us visit
        rate (a 256 us window with 8 samples of pre-trigger lead-in),
        captured whenever that channel's trigger fires. Returns a complex64
        array of shape (n_captured, 128).

        bins: a channel number or a sequence of them, at most 4 per lane
        (lane = bin & 3). GATEWARE LIMITATION: stamps carry no channel or
        cycle metadata in memory (the RTL generates it but never routes it
        to the DMA), so with more than one bin enabled stamps cannot be
        attributed to channels - use one bin at a time unless attribution
        does not matter. The pre-trigger samples of the very first stamp
        per lane may contain stale data from before the engines started.

        threshold/holdoff: scalar or per-bin sequences; default to the
        values last programmed via configure_channel()/configure(). The
        channels are temporarily forced enabled with the postage flag set,
        and restored afterward.

        Timeout: mirrors capture_photons - stamps already landed are
        recovered; with partial='raise' (default) they ride on a
        CaptureTimeout as .stamps, with partial='return' they are returned.
        A partially filled buffer stays queued in the gateware (flush is a
        no-op) and resumes on the next capture_postage call.
        """
        if not _PYNQ:
            raise RuntimeError('pynq not available')
        if partial not in ('raise', 'return'):
            raise ValueError("partial must be 'raise' or 'return'")
        bins = [int(bins)] if np.isscalar(bins) else [int(b) for b in bins]
        if not bins or len(set(bins)) != len(bins):
            raise ValueError('bins must be a non-empty set of distinct channels')
        lanes = {}
        for b in bins:
            if not 0 <= b < self.N_CHANNELS:
                raise ValueError(f'bin {b} out of range')
            lanes.setdefault(b & 3, []).append(b)
        if any(len(v) > POSTAGE_MAX_PER_LANE for v in lanes.values()):
            raise ValueError(f'at most {POSTAGE_MAX_PER_LANE} postage bins '
                             'per lane (lane = bin & 3)')
        if len(bins) > 1:
            _logger.warning('multiple postage bins enabled: stamps carry no '
                            'channel metadata and cannot be attributed')
        n_stamps = int(n_stamps)
        if not 0 < n_stamps < (1 << 24) // POSTAGE_STAMP_BYTES:
            raise ValueError('n_stamps must be in [1, 32767]')

        def _per_bin(val, name, shadow):
            if val is None:
                missing = [b for b in bins if not self._shadow_valid[b]]
                if missing:
                    raise ValueError(f'{name} not given and channel(s) '
                                     f'{missing} were never configured')
                return {b: int(shadow[b]) for b in bins}
            if np.isscalar(val):
                return {b: int(val) for b in bins}
            val = [int(v) for v in val]
            if len(val) != len(bins):
                raise ValueError(f'{name} must be scalar or one per bin')
            return dict(zip(bins, val))

        thr = _per_bin(threshold, 'threshold', self._thr_shadow)
        hold = _per_bin(holdoff, 'holdoff', self._hold_shadow)
        orig_en = {b: bool(self._en_shadow[b]) if self._shadow_valid[b] else True
                   for b in bins}

        if self._postage_inflight is not None:
            buf, done = self._postage_inflight
            self._postage_inflight = None
            pushed = True   # its address is still queued in the gateware
            capacity = buf.size // POSTAGE_SAMPLES
            if n_stamps > capacity - done:
                _logger.warning('resuming parked postage buffer: n_stamps '
                                'clipped to %d', capacity - done)
                n_stamps = capacity - done
        else:
            buf = allocate(shape=(n_stamps * POSTAGE_SAMPLES,), dtype=np.uint32)
            done = 0
            pushed = False
        capacity = buf.size // POSTAGE_SAMPLES
        prev_stamper = self.valve_status['stamper']
        b0 = self.postage_dma.burst_count
        timed_out = False
        status = {}
        try:
            if not pushed:
                try:
                    self.postage_dma.push_buffer(buf)
                except TimeoutError:
                    buf.freebuffer()
                    raise
            self.set_valves(stamper=ValvePosition.OPEN)
            for b in bins:
                self.configure_channel(b, thr[b], hold[b], postage=True,
                                       enabled=True)
            self._postage_control_write({l: len(v) for l, v in lanes.items()})
            t0 = time.time()
            while ((self.postage_dma.burst_count - b0) & 0xffff) < capacity - done:
                if time.time() - t0 > timeout:
                    timed_out = True
                    break
                time.sleep(0.005)
        finally:
            # Stop the engines; they finish writing any whole stamps still
            # buffered, then assert their flushed bits.
            self._postage_control_write({})
            mask = 0
            for lane in lanes:
                mask |= 1 << lane
            t0 = time.time()
            status = self._postage_status()
            while (status['flushed'] & mask) != mask and time.time() - t0 < 1.0:
                time.sleep(0.001)
                status = self._postage_status()
            for b in bins:
                self.configure_channel(b, thr[b], hold[b], postage=False,
                                       enabled=orig_en[b])
            self.set_valves(stamper=prev_stamper)
        if (status['flushed'] & mask) != mask:
            _logger.warning('postage engines did not flush cleanly '
                            '(flushed=%#x, expected mask %#x)',
                            status['flushed'], mask)
        if status.get('dropped') or status.get('fault'):
            _logger.warning('postage dropped/fault flagged: %s', status)
        landed = min((self.postage_dma.burst_count - b0) & 0xffff,
                     capacity - done)
        buf.invalidate()
        stamps = unpack_postage(np.array(buf))[done:done + landed]
        if done + landed < capacity:
            # Buffer not full: its DMA address is still queued and cannot be
            # retired; park it (keeping it allocated) for the next capture.
            self._postage_inflight = (buf, done + landed)
        else:
            buf.freebuffer()
        if timed_out:
            _logger.warning('postage capture timed out: %d/%d stamps landed; '
                            'buffer parked', landed, n_stamps)
            if partial == 'raise':
                raise CaptureTimeout(
                    f'postage capture: {landed}/{n_stamps} stamps landed',
                    stamps=stamps)
        return stamps
