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
"""
import logging
import time

import numpy as np

try:
    from pynq import DefaultIP, allocate
    _PYNQ = True
except Exception:  # pragma: no cover - allows import (and unpacker tests) off-board
    DefaultIP = object
    allocate = None
    _PYNQ = False

_logger = logging.getLogger(__name__)

CYCLE_NS = 2000  # one trigger visit cycle per channel = 2 us

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


def photon_times_ns(photons, header):
    """Absolute time in ns for each unpacked photon, given a chunk header dict."""
    return header['time_ns'] + (photons['cycle'].astype(np.int64) - int(header['cycle'])) * CYCLE_NS


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
        self.trigger_dma = _HuskyDMA(self, _Reg.TRIG_DMA)
        self.postage_dma = _HuskyDMA(self, _Reg.POSTAGE_DMA)

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
        holdoff in visit cycles (2 us units, 0-255)."""
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
    def capture_photons(self, n_chunks=8, events_per_chunk=4096, timeout=10.0,
                        with_headers=False):
        """Synchronous double-buffered capture of n_chunks chunks.

        Returns a PHOTON_V2_DTYPE array (and the list of chunk-header dicts if
        with_headers). The trigger valve must be OPEN and channels configured.
        """
        if not _PYNQ:
            raise RuntimeError('pynq not available')
        bufs = [allocate(shape=(events_per_chunk, 2), dtype=np.uint64) for _ in range(2)]
        try:
            self.read_chunk_header()  # clear latches, learn starting read tag
            out, headers = [], []
            self.trigger_dma.push_buffer(bufs[0])
            for i in range(n_chunks):
                nxt = bufs[(i + 1) % 2]
                if i + 1 < n_chunks:
                    self.trigger_dma.push_buffer(nxt)
                cur = bufs[i % 2]
                t0 = time.time()
                while self.trigger_dma.fifo_count > (1 if i + 1 < n_chunks else 0) \
                        or not (self.interrupt_status['fullchunk'] or self.trigger_dma.completed()):
                    if time.time() - t0 > timeout:
                        raise TimeoutError(f'chunk {i} did not complete '
                                           f'(status={self.interrupt_status})')
                    time.sleep(0.001)
                cur.invalidate()
                hdr = self.read_chunk_header()
                headers.append(hdr)
                if hdr['fault']:
                    _logger.warning('trigger DMA fault flagged in chunk %d', i)
                if hdr['dropped']:
                    _logger.warning('events dropped in chunk %d', i)
                out.append(unpack_photons_v2(np.array(cur)))
            photons = np.concatenate(out) if out else np.zeros(0, dtype=PHOTON_V2_DTYPE)
            return (photons, headers) if with_headers else photons
        finally:
            for b in bufs:
                b.freebuffer()
