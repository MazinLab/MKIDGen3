from pynq import DefaultIP, buffer
import numpy as np
from logging import getLogger
import threading
from queue import Queue
import queue
import asyncio
import time

class PhotonTrigger(DefaultIP):
    ADDR_RESMAP = 0x1000
    bindto = ['mazinlab:mkidgen3:trigger:0.4']

    def __init__(self, description):
        """
        The core uses an array of 512 values, each consisting of 4 pairs of values packed into a 64bit dword.
        These pairs of values consist of the threshold (a byte treated as an ap_fixed<8,1> number for comparison
        against the ap_fixed<16,1> phase) and a holdoff value (a byte that specified the number of samples over which to
        pick the minimal phase for a photon).

        The holdoff value is in units of the phase stream sample rate (nominally 1MHz). It must not exceed 245 as the
        driver needs to add 1 to the requested value internally. It is expected that a value below ~7 will cause
        aberrant and unpredictable behavior in the full design even if the core itself supports it

        Although the core will issue information about photon overflow as a stream it does not maintain a register
        for it and, as such, this driver does not provide information in that regard.

        x[0] consists of (from low to high byte) thresh0 hoff0 thresh1 hoff1 thresh2 hoff2 thresh3 hoff3
        x[511] consists of (from low to high byte) thresh2044 hoff2044 ... thresh2047 hoff2047

        control
        0x1000 ~
        0x1fff : Memory 'threshoffs' (512 * 64b)
                 Word 2n   : bit [31:0] - threshoffs[n][31: 0]
                 Word 2n+1 : bit [31:0] - threshoffs[n][63:32]
        """
        super().__init__(description=description)

    def _fetch(self):
        """ Return arrays of the thresholds (in raw format) and the holdoffs assigned to each of the  2048 resonator
        channels
        """
        sl = slice(0x1000 // 4, 0x1000 // 4 + 1024)
        x = np.frombuffer(np.array(self.mmio.array[sl]), dtype=np.uint8)
        return x[::2], x[1::2]

    def configure(self, thresholds=None, holdoffs=None):
        """Thresholds shall be floating point numbers in [-1,1) and will be converted to ap_fixed<8,0>"""
        if thresholds is None and holdoffs is None:
            return

        _thresholds, _holdoffs = None, None
        if thresholds is not None:
            thresholds = (thresholds.clip(-1, 1) * 128).round().astype(int).clip(-128, 127)
            if len(thresholds) != 2048:
                raise ValueError('len(thresholds)!=2048')
        else:
            _thresholds, _holdoffs = self._fetch()

        if holdoffs is not None:
            holdoffs = holdoffs.astype(int).clip(0, 254)
            if len(holdoffs) != 2048:
                raise ValueError('len(holdoffs)!=2048')
        elif _holdoffs is None:
            _thresholds, _holdeffos = self._fetch()

        if thresholds is None:
            thresholds = _thresholds
        if holdoffs is None:
            holdoffs = _holdoffs + 1

        assert thresholds.size == holdoffs.size

        data = np.zeros((2048, 2), dtype=np.uint8)
        data[:, 0] = thresholds.astype(np.uint8)
        data[:, 0] = holdoffs
        sl = slice(0x1000 // 4, 0x1000 // 4 + 1024)
        self.mmio.array[sl]=np.frombuffer(data, dtype=np.uint32)


class PhotonPostageFilter(DefaultIP):
    ADDR_MONITOR_CHAN = tuple(range(0x10, 0x49, 0x8))
    bindto = ['mazinlab:mkidgen3:postage_filter:0.2']

    def __init__(self, description):
        """
        The core watches for trigger events on 8 resonator channels and forwards IQ snippets around trigger events on
        for capture by other parts of the firmware. It is configured with 8 resonator channel values for monitoring.

        control
        0x10 : Data signal of monitor_0
               bit 11~0 - monitor_0[11:0] (Read/Write)
               others   - reserved
        0x18, 0x20, 0x28,0x30,0x38,0x40 0x48 : Data signal of monitor_1 - 7

        """
        super().__init__(description=description)

    @property
    def monitor_channels(self):
        return tuple(self.read(a) & 0x7ff for a in self.ADDR_MONITOR_CHAN)

    def configure(self, monitor_channels=None):
        """ monitor_channels shall be 8 integers in [0,2047] """
        if monitor_channels is None:
            return
        monitor_channels = np.asarray(monitor_channels, dtype=np.uint16).clip(0, 2047)
        for a, c in zip(self.ADDR_MONITOR_CHAN, monitor_channels):
            self.write(a, int(c))


class PhotonPostageMAXI(DefaultIP):
    POSTAGE_BUFFER_LEN = 1000*8  # Must be POSTAGE_BUFSIZE from the HLS
    N_CAPDATA = 127  # Must be 1 less than the HLS value
    bindto = ['mazinlab:mkidgen3:postage_maxi:0.2']

    def __init__(self, description):
        """
        The core captures up to some number of Photon events to a pynq buffer. The resonators for monitoring are
        configured by PhotonPostageFilter.

        // control
        // 0x00 : Control signals
        //        bit 0  - ap_start (Read/Write/COH)
        //        bit 1  - ap_done (Read/COR)
        //        bit 2  - ap_idle (Read)
        //        bit 3  - ap_ready (Read/COR)
        //        bit 7  - auto_restart (Read/Write)
        //        bit 9  - interrupt (Read)
        //        others - reserved
        // 0x04 : Global Interrupt Enable Register
        //        bit 0  - Global Interrupt Enable (Read/Write)
        //        others - reserved
        // 0x08 : IP Interrupt Enable Register (Read/Write)
        //        bit 0 - enable ap_done interrupt (Read/Write)
        //        bit 1 - enable ap_ready interrupt (Read/Write)
        //        others - reserved
        // 0x0c : IP Interrupt Status Register (Read/COR)
        //        bit 0 - ap_done (Read/COR)
        //        bit 1 - ap_ready (Read/COR)
        //        others - reserved
        // 0x10 : Data signal of iq
        //        bit 31~0 - iq[31:0] (Read/Write)
        // 0x14 : Data signal of iq
        //        bit 31~0 - iq[63:32] (Read/Write)
        // 0x18 : reserved
        // 0x1c : Data signal of event_count
        //        bit 15~0 - event_count[15:0] (Read)
        //        others   - reserved
        // 0x20 : Control signal of event_count
        //        bit 0  - event_count_ap_vld (Read/COR)
        //        others - reserved
        // 0x2c : Data signal of max_events
        //        bit 15~0 - max_events[15:0] (Read/Write)
        //        others   - reserved
        // 0x30 : reserved
        """
        super().__init__(description=description)
        self._buf = None

    def capture(self, max_events=None):
        if max_events is None:
            max_events = self.POSTAGE_BUFFER_LEN
        if not self.register_map.CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf
        self.register_map.IP_IER.CHAN0_INT_EN = 1
        self.register_map.GIER = 1
        self.read(0x0C)
        self.write(0x2c, min(max(max_events, 1), self.POSTAGE_BUFFER_LEN))
        self._buf = buffer.allocate((self.POSTAGE_BUFFER_LEN, self.N_CAPDATA+1, 2), dtype=np.int16)
        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())
        self.register_map.CTRL.AP_START = 1
        return self._buf

    def get_postage(self, raw=False, scaled=True):
        """
        Raw takes precedence

        Returns array of event resonator channels and array of [n_events,N_CAPDATA]

        get channels with events via set(ids)
        select events from a channel via events[ids==<your_channel>]
        """
        count = self.event_count
        data = np.array(self._buf[:count])
        ids = data[:, 0, 0].astype(np.uint16)
        events = data[:, 1:, :]
        if not raw:
            if scaled:
                events /= 2**14
            events = events[:, :, 0]+events[:, :, 0]*1j
        return ids, events

    @property
    def event_count(self):
        return self.read(0x1c)

    def configure(self, max_events=None):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture(max_events)


class PhotonMAXI(DefaultIP):
    N_PHOTON_BUFFERS = 2  # Must match HLS C
    PHOTON_BUFF_N = 102400  # 10ms 5000 cps 2048 resonators Must match HLS C
    PHOTON_DTYPE = np.dtype([('time', np.uint64), ('phase', np.int16), ('id', np.uint16)])
    PHOTON_PACKED_DTYPE = np.uint64
    bindto = ['mazinlab:mkidgen3:photon_maxi:0.2']

    @staticmethod
    def pack_photons(x, out=None):
        ret = np.zeros(x.size, dtype=PhotonMAXI.PHOTON_PACKED_DTYPE) if out is None else out
        ret[:] = (((x['time'] << 12) | x['id']) << 16) | x['phase'].astype(np.uint16)
        return ret

    @staticmethod
    def unpack_photons(x, out=None):
        ret = np.zeros(x.size, dtype=PhotonMAXI.PHOTON_DTYPE) if out is None else out
        ret['phase'] = x & 0xffff
        ret['time'] = x >> 28
        ret['id'] = (x >> 16) & 0xfff
        return ret

    @staticmethod
    def gen_fake_buffer_data(n, time_us):
        n = int(n)
        data = np.zeros(n, dtype=PhotonMAXI.PHOTON_DTYPE)
        data['time'] = np.sort(np.random.randint(0, high=time_us, size=n))
        data['id'] = np.random.randint(0, high=2047, size=n)
        data['phase'] = np.random.randint(-0x7fff, high=0x7fff, size=n)
        return data

    def __init__(self, description):
        """
        The core watches for trigger events on 8 resonator channels and forwards IQ snippets around trigger events on
        for capture by other parts of the firmware. It is configured with 8 resonator channel values for monitoring.

        // 0x00 : Control signals
        //        bit 0  - ap_start (Read/Write/COH)
        //        bit 1  - ap_done (Read/COR)
        //        bit 2  - ap_idle (Read)
        //        bit 3  - ap_ready (Read/COR)
        //        bit 7  - auto_restart (Read/Write)
        //        bit 9  - interrupt (Read)
        //        others - reserved
        // 0x04 : Global Interrupt Enable Register
        //        bit 0  - Global Interrupt Enable (Read/Write)
        //        others - reserved
        // 0x08 : IP Interrupt Enable Register (Read/Write)
        //        bit 0 - enable ap_done interrupt (Read/Write)
        //        bit 1 - enable ap_ready interrupt (Read/Write)
        //        others - reserved
        // 0x0c : IP Interrupt Status Register (Read/COR)
        //        bit 0 - ap_done (Read/COR)
        //        bit 1 - ap_ready (Read/COR)
        //        others - reserved
        // 0x10 : Data signal of photons_out
        //        bit 31~0 - photons_out[31:0] (Read/Write)
        // 0x14 : Data signal of photons_out
        //        bit 31~0 - photons_out[63:32] (Read/Write)
        // 0x18 : reserved
        // 0x28 : Data signal of active_buffer
        //        bit 7~0 - active_buffer[7:0] (Read)
        //        others  - reserved
        // 0x2c : Control signal of active_buffer
        //        bit 0  - active_buffer_ap_vld (Read/COR)
        //        others - reserved
        // 0x38 : Data signal of photons_per_buf
        //        bit 16~0 - photons_per_buf[16:0] (Read/Write)
        //        others   - reserved
        // 0x3c : reserved
        // 0x40 : Data signal of time_shift
        //        bit 4~0 - time_shift[4:0] (Read/Write)
        //        others  - reserved
        // 0x44 : reserved
        // 0x20 ~
        // 0x27 : Memory 'n_photons' (2 * 17b)
        //        Word n : bit [16:0] - n_photons[n]
        //                 others     - reserved
        """
        super().__init__(description=description)
        self._buf = None

    def get_photons(self):
        ab = self.read(0x28) & 0xff
        count = self.read(0x20) if ab else self.read(0x24)
        count &= 0x1ffff
        ret = np.array(self._buf[0 if ab else 1][:count])
        ab2 = self.read(0x28) & 0xff
        getLogger(__name__).debug(f'Active Buffer: {ab}. Buffer counts: {count}')
        if not ab == ab2:
            raise RuntimeError('Buffer changed during read')
        return ret

    def stop_capture(self):
        self.register_map.CTRL.AUTO_RESTART = 0

    def photon_fountain(self):
        """
        returns photon_queue, kill_event, future. kill with kill_event.set()
        get photons with PhotonMAXI.unpack_photons(q.get())
        """
        async def get_photons_corot(self, q, kill):
            while not kill.is_set():
                await self.interrupt.wait()
                try:
                    tic = time.time()
                    p = self.get_photons()
                    toc = time.time()
                    getLogger(__name__).getChild('timing').debug(f'Get Photons took {(toc-tic)*1000:.2f} ms')
                except RuntimeError:
                    getLogger(__name__).error(f"Dropping photons, couldn't keep up with with buffer rotation")
                    continue
                try:
                    q.put_nowait(p)
                except queue.Full:
                    getLogger(__name__).debug(f'Dropping photons, fountain Q full')
        loop = asyncio.new_event_loop()
        q = Queue(maxsize=10)
        kill = threading.Event()
        coro = get_photons_corot(self, q, kill)
        # Submit the coroutine to a given loop
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return q, kill, future

    @property
    def buffer_count_interval(self):
        return self.read(0x38)

    @property
    def buffer_interval(self):
        return 2 ** self.read(0x40) / 1e3

    @buffer_interval.setter
    def buffer_interval(self, interval_ms):
        if not self.register_map.CTRL.AP_IDLE:
            getLogger(__name__).warning('buffer_interval change will not take effect until core restart')
        l2_buffer_shift = np.round(np.log2(interval_ms * 1000))
        _buffer_time_ms = 2 ** l2_buffer_shift / 1000
        if l2_buffer_shift < 9 or l2_buffer_shift > 20:
            getLogger(__name__).warning(f'Requested photon buffer interval ({interval_ms:.2f} ms) unsupported, '
                                        f'using {_buffer_time_ms:.2f}')
            l2_buffer_shift = min(max(l2_buffer_shift, 9), 20)
        elif abs(_buffer_time_ms - interval_ms) >= .01:
            getLogger(__name__).info(f'Photon buffer interval {interval_ms:.2f} ms rounded to '
                                     f'{_buffer_time_ms:.2f} ms')
        else:
            getLogger(__name__).debug(f'Photon buffer interval: {_buffer_time_ms:.2f} ms')
        self.write(0x40, int(l2_buffer_shift))

    def capture(self, n_photons_per_buffer=2 ** 16 - 1, buffer_time_ms=4.096):
        if not self.register_map.CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf
        self.register_map.IP_IER.CHAN0_INT_EN = 1
        self.register_map.GIER = 1
        self.read(0x0C)
        self._buf = buffer.allocate((self.N_PHOTON_BUFFERS, self.PHOTON_BUFF_N), dtype=self.PHOTON_PACKED_DTYPE,
                                    cacheable=True)
        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())
        self.buffer_interval = buffer_time_ms
        if n_photons_per_buffer > 2 ** 16 - 1:
            getLogger(__name__).warning(f'Requested rotation count too high, photon buffer will be rotated at n'
                                        f'={2 ** 16 - 1}')
        self.write(0x38, min(max(int(n_photons_per_buffer), 0), 2 ** 16 - 1))
        self.register_map.CTRL.AUTO_RESTART = 1
        self.register_map.CTRL.AP_START = 1
        return self._buf

    def configure(self):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture()
