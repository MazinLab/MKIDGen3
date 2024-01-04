import zmq
from pynq import DefaultIP, buffer
import numpy as np
from logging import getLogger
import threading
from queue import Queue
import queue
import asyncio
import time
from collections import namedtuple
from enum import Enum
from datetime import datetime

from ..mkidpynq import PHOTON_DTYPE as _PHOTON_DTYPE
from ..system_parameters import LOWPASSED_IQ_SAMPLE_RATE, PHOTON_POSTAGE_WINDOW_LENGTH
from ..util import do_asyncio_thing


class WaitMode(Enum):
    """Enumerates capture waiting modes."""
    POLL = 0
    """ Poll the interrupt """

    ASYNC = 1
    """ Use asyncio to wait on the interrupt"""

    SLEEP = 2
    """ Just sleep """


class PhotonTrigger(DefaultIP):
    bindto = ['mazinlab:mkidgen3:trigger:0.4', 'mazinlab:mkidgen3:fake_trigger:0.1']
    THRESHOFF_SLICE = slice(0x1000 // 4, 0x1000 // 4 + 1024)
    StatusTuple = namedtuple('PhotonTriggerStatus', ('thresholds', 'holdoffs'))

    def __init__(self, description):
        """
        The core uses an array of 512 values, each consisting of 4 pairs of values packed into a 64bit dword.
        These pairs of values consist of the threshold (a byte treated as an ap_fixed<8,1> number for comparison
        against the ap_fixed<16,1> phase) and a holdoff value (a byte that specified the number of samples over which to
        pick the minimal phase for a photon).

        The holdoff value is in units of the phase stream sample rate (nominally 1MHz). It must not exceed 254 as the
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

    def configuration(self, scaled=True):
        """
        Return named tuple of thresholds (in raw format or scaled format) and holdoffs assigned to each of the
        2048 resonator channels
        """
        x = np.frombuffer(np.array(self.mmio.array[self.THRESHOFF_SLICE]), dtype=np.uint8)
        return self.StatusTuple(thresholds=x[::2].astype(np.int8) / 128 if scaled else x[::2].astype(np.int8),
                                holdoffs=x[1::2])

    def configure(self, thresholds=None, holdoffs=None):
        """Thresholds shall be floating point numbers in [-1,1) and will be converted to ap_fixed<8,0>"""
        if thresholds is None and holdoffs is None:
            return

        _thresholds, _holdoffs = None, None
        if thresholds is not None:
            thresholds = (np.asarray(thresholds).clip(-1, 1) * 128).round().astype(int).clip(-128, 127)
            if len(thresholds) != 2048:
                raise ValueError('len(thresholds)!=2048')
        else:
            _thresholds, _holdoffs = self.configuration(scaled=False)

        if holdoffs is not None:
            holdoffs = np.asarray(holdoffs).astype(int).clip(0, 254)
            if len(holdoffs) != 2048:
                raise ValueError('len(holdoffs)!=2048')
        elif _holdoffs is None:
            _thresholds, _holdeffos = self.configuration(scaled=False)

        if thresholds is None:
            thresholds = _thresholds
        if holdoffs is None:
            holdoffs = _holdoffs + 1

        assert thresholds.size == holdoffs.size

        data = np.zeros((2048, 2), dtype=np.uint8)
        data[:, 0] = thresholds.astype(np.uint8)
        data[:, 1] = holdoffs
        self.mmio.array[self.THRESHOFF_SLICE] = np.frombuffer(data, dtype=np.uint32)


class PhotonPostageFilter(DefaultIP):
    ADDR_MONITOR_CHAN = tuple(range(0x10, 0x49, 0x8))
    bindto = ['mazinlab:mkidgen3:postage_filter:0.2', 'mazinlab:mkidgen3:postage_filter_w_interconn:0.1']

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
        monitor_channels = np.asarray(monitor_channels[:8], dtype=np.uint16).clip(0, 2047)
        for a, c in zip(self.ADDR_MONITOR_CHAN, monitor_channels):
            self.write(a, int(c))


class PhotonPostageMAXI(DefaultIP):
    POSTAGE_BUFFER_LEN = 1000 * 8  # Must be POSTAGE_BUFSIZE from the HLS
    N_CAPDATA = PHOTON_POSTAGE_WINDOW_LENGTH  # Must be 1 less than the HLS value
    MAX_CAPTURE_TIME_S = POSTAGE_BUFFER_LEN / 8 * (N_CAPDATA + 1) / LOWPASSED_IQ_SAMPLE_RATE
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

    def capture(self, max_events=None, wait=True):
        if max_events is None:
            max_events = self.POSTAGE_BUFFER_LEN
        if not self.register_map.CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf

        # Enable the interrupt
        self.register_map.IP_IER.CHAN0_INT_EN = 1
        self.register_map.GIER = 1
        self.read(0x0C)  # clear it
        self.write(0x2c, min(max(max_events, 1), self.POSTAGE_BUFFER_LEN))
        if self._buf is not None:
            getLogger(__name__).debug('Zeroing and reusing existing buffer')
            assert self._buf.shape==(self.POSTAGE_BUFFER_LEN, self.N_CAPDATA + 1, 2)
            assert self._buf.dtype==np.int16
            self._buf[:] = 0
        else:
            self._buf = buffer.allocate((self.POSTAGE_BUFFER_LEN, self.N_CAPDATA + 1, 2), dtype=np.int16)

        self._buf[:,0,0] = -1  # set a sentinal value
        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())
        self.register_map.CTRL.AP_START = 1
        if wait:
            do_asyncio_thing(self.interrupt.wait())
        return self._buf

    def get_postage(self, raw=False, scaled=True, rawbuffer=False):
        """
        Raw takes precedence

        Returns array of event resonator channels and array of [n_events,N_CAPDATA]

        get channels with events via set(ids)
        select events from a channel via events[ids==<your_channel>]
        """
        count = self.event_count
        data = np.array(self._buf[:count])
        if rawbuffer:
            return data
        ids = data[:, 0, 0].astype(np.uint16)
        events = data[:, 1:, :]
        if not raw:
            events = events[:, :, 0] + events[:, :, 1] * 1j
            if scaled:
                events /= 2 ** 14
        return ids, events

    @property
    def event_count(self):
        return self.read(0x1c)

    def configure(self, max_events=None):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture(max_events)


class PhotonMAXI(DefaultIP):
    N_PHOTON_BUFFERS = 2  # Must match HLS C
    PHOTON_BUFF_N = 102400  # Must match HLS C and include padding for the partial burst
    PHOTON_DTYPE = _PHOTON_DTYPE
    PHOTON_PACKED_DTYPE = np.uint64
    bindto = ['mazinlab:mkidgen3:photon_maxi:0.2']

    """
    The PhotonMAXI core operates in AP_CONTINUE mode, ending and auto-restarting each time it needs to rotate the buffer
    
    The core reads inbound photons and when it has enough for a maximal (512) burst it writes them to the buffer.
    every time it reads a photon it sets the number of photons written to that buffer (so this number will increment in
    steps of 512 and will generally report that its value is valid while the core is running. 
    
    It appears that during a burst the core continues to ingest photons.
    
    Once a photon is received that has a timestamp with high bits that do not equal the preceding photon OR the number 
    of photons written is at least the rotation quantity the core will burst out one final, partial burst, update the 
    count of photons in that buffer and restart using the next buffer.
    
    The `active_buffer` is the buffer presently being updated. So if active 
    """

    @staticmethod
    def pack_photons(x, out=None):
        ret = np.zeros(x.size, dtype=PhotonMAXI.PHOTON_PACKED_DTYPE) if out is None else out
        ret[:] = (((x['time'] << 12) | x['id']) << 16) | x['phase'].astype(np.uint16)
        return ret

    @staticmethod
    def unpack_photons(x, out=None, n=0):
        """
        Unpack packed photons, optionally accumulating them into an existing output array

        Args:
            x: an array of packed photons
            out: optional, an array of type PhotonMAXI.PHOTON_DTYPE with the shape of x, but see n
            n: optional, an index into out to insert unpacked photons, the first axis is used if >1d an
                IndexError is raised if there is insufficient space.

        Returns: the unpacked photon array

        """
        if out is None:
            n=0
        elif x.shape[0]+n>out.shape[0]:
            raise IndexError('Output array is too small')

        ret = np.zeros(x.shape, dtype=PhotonMAXI.PHOTON_DTYPE) if out is None else out
        sl=slice(n,n+x.shape[0])
        ret['phase'][sl] = x & 0xffff
        ret['time'][sl] = x >> 28
        ret['id'][sl] = (x >> 16) & 0xfff
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

    def get_photons(self, no_copy=False):
        """v0.4 uses 9c7ad901ef20bb356010109c83311bc5e49db81e of photon_maxi
        The active buffer is the index of the buffer being loaded and so the other index should be used
        the active buffer is written every time a photon comes in and so will very likely be valid
        """

        if not self.register_map.active_buffer_ctrl.active_buffer_ap_vld:
            getLogger(__name__).warning(f'active_buffer_ctrl not valid and getting, {repr(self.register_map)}')
            # return None  # No valid buffer
        tic=time.perf_counter()
        cnta, cntb, ab = self.mmio.array[8:11]  # about 66% faster that a pair of reads, about 64-140 us
        ab &= 0xff
        count = cnta if ab else cntb
        count &= 0x1ffff

        ret = self._buf[0 if ab else 1][:count]
        if not no_copy:
            ret = np.array(ret)

        if self.register_map.active_buffer_ctrl.active_buffer_ap_vld:
            toc=time.perf_counter()
            ab2 = self.mmio.array[10] & 0xff
            getLogger(__name__).warning(f'Buffer {ab}->{ab2} in {int((toc-tic)*1e6)}. Dropping {count} photons.')
            raise RuntimeError('Buffer changed during read')

        return ret

    def stop_capture(self):
        self.register_map.CTRL.AUTO_RESTART = 0

    def wait(self, method: WaitMode = WaitMode.POLL, interval=None):
        """
        Wait for a capture buffer to be ready, will clear interrupt before awaiting if using asyncio, may not want this

        interval required for WaitMode.SLEEP
        """
        if method == WaitMode.POLL:
            interval = interval or .000010
            while not self.register_map.IP_ISR.CHAN0_INT_ST:  # self.register_map.CTRL.INTERRUPT:
                time.sleep(interval)  # default to 10us
        elif method == WaitMode.SLEEP:
            assert interval, 'Interval required'
            time.sleep(interval)
        else:
            self.read(0x0C)  # clear interrupt
            do_asyncio_thing(self.interrupt.wait())
        return

    def photon_fountain(self, q: (zmq.Socket, queue.SimpleQueue), kill=None, copy_buffer=False, spawn=True):
        """
        Is both the target for and can spawn a thread of the same

        q may be a zma socket or a queue.Queue

        kill must be a threading.Event if spawn is not True

        returns an unstarted thread and a kill event to kill the thread

        photons will be fetched and sent over the q copying the buffer if copy_buffer is True

        thread MUST be started before .capture() is called to initiate capture

        """
        if spawn:
            stop = kill = threading.Event()
            fetcher_thread = threading.Thread(target=self.photon_fountain, name='photon fountain',
                                              args=(q, ), kwargs=dict(kill=kill, spawn=False, copy_buffer=copy_buffer))
            return fetcher_thread, stop

        assert isinstance(kill, threading.Event)

        log = getLogger(__name__)
        delts = []
        times=[time.perf_counter()]

        while True:
            times.append(time.perf_counter())
            if kill.is_set():
                break
            times.append(time.perf_counter())
            self.wait(WaitMode.ASYNC)
            times.append(time.perf_counter())
            try:
                x = self.get_photons(no_copy=not copy_buffer)

                # if x is None:
                #     times.append(time.time())
                #     dif = (np.diff(times) * 1e6).astype(int)
                #     delts.append(dif)
                #     x = datetime.utcnow().strftime('%M:%S.%f')
                #     na=('logmsg','zmqrecv','sleep','get')
                #     log.debug(f'Empty Fetch @ {x}, us intervals: {list(zip(na,dif))}')
                #     times = times[-1:]
                #     continue

            except RuntimeError:
                times.append(time.perf_counter())
                dif = (np.diff(times) * 1e6).astype(int)
                delts.append(dif)
                x = datetime.utcnow().strftime('%M:%S.%f')
                na = ('logmsg', 'zmqrecv', 'sleep', 'get')
                log.error(f'Drop Fetch @ {x}, us intervals: {list(zip(na, dif))}')
                # log.error(f"Dropping photons, couldn't keep up with buffer rotation")
                times = times[-1:]
                continue

            times.append(time.perf_counter())

            if hasattr(q, 'put'):
                q.put(x)
            else:
                q.send(x, copy=False)

            times.append(time.perf_counter())
            dif = (np.diff(times)*1e6).astype(int)
            delts.append(dif)
            x = datetime.utcnow().strftime('%M:%S.%f')
            na = ('logmsg', 'zmqrecv', 'sleep', 'get','inq')
            log.error(f'Fetch @ {x}, us intervals: {list(zip(na, dif))}')
            times=times[-1:]

        log.info(f'Time between waits: {delts}')

        if hasattr(q, 'put'):
            q.put(None)
        else:
            q.send(b'')
            q.close()

    @property
    def buffer_count_interval(self):
        return self.read(0x38) & 0x1ffff

    @property
    def buffer_interval(self):
        return 2 ** (self.read(0x40)&0x1f) / 1e3

    @buffer_interval.setter
    def buffer_interval(self, interval_ms):
        """Permissible range is about 0.5 - 1000 ms and will be clipped"""
        if not self.register_map.CTRL.AP_IDLE and self.register_map.CTRL.AUTO_RESTART:
            getLogger(__name__).warning('Buffer rotation interval will not take effect until core (auto) restart')

        l2_buffer_shift = np.round(np.log2(interval_ms * 1000))
        _buffer_time_ms = 2 ** l2_buffer_shift / 1000
        if l2_buffer_shift < 9 or l2_buffer_shift > 20:
            l2_buffer_shift = min(max(l2_buffer_shift, 9), 20)
            getLogger(__name__).warning(f'Requested rotation interval ({interval_ms:.2f} ms) unsupported, '
                                        f'using {2 ** l2_buffer_shift / 1000:.2f}')
        elif abs(_buffer_time_ms - interval_ms) >= .01:
            getLogger(__name__).info(f'Requested rotation interval {interval_ms:.2f} ms rounded to '
                                     f'{_buffer_time_ms:.2f} ms')
        else:
            getLogger(__name__).info(f'Setting rotation interval to {_buffer_time_ms:.2f} ms')

        self.write(0x40, int(l2_buffer_shift))

    def capture(self, n_photons_per_buffer=101888, buffer_time_ms=4.096, cacheable_buffer=True):
        """
        Initiate capture of photons. Any existing buffer will be freed.

        Note that proper functionality of the core requires that at least two photons arrive at least buffer_time_ms
        apart or more than n_photons_per_buffer are seen. If neither of those conditions is met the core will not
        complete. If fewer than 512 photons are seen AND they all arrive within one buffer interval no data will even
        be written into the buffer. Both of these scenarios are essentially impossible and can be resolved by setting
        an arbitrarily low event trigger threshold in the trigger core.

        The FPGA photon buffer is used as a C array of size uint128_t[N_PHOTON_BUFFERS][FLAT_PHOTON_BUFSIZE/2]
        FLAT_PHOTON_BUFSIZE 102400
        N_PHOTON_BUFFERS 2

        Photons are packed in uint64_t and paired for writing, hence the half size axis and 128b word length for
        the core. In the core:
        0,0 [photon0 photon1 buf1] 0,1 [photon2 photon3 buf1] .....
        1,0 [photon0 photon1 buf2] 1,1 [photon2 photon3 buf2] .....

        In python we want to allocate  (N_PHOTON_BUFFERS, FLAT_PHOTON_BUFSIZE)

        the photon count buffer is 17 bits and maxes out at 131071

        the core only increments the count in units of 512 however (a maximal burst of 4KiB of photons)
        then when a threshold is crossed (n_photons_per_buffer) or the high bits of two successive photons
        aren't the same we stop bursting, send off any remaining values, and swap buffers

        the active buffer (preburst) and photon count (post burst) are set once per burst, then the count updated with the
        numbers from the final partial burst (just before restart, buffer swap and zeroing of new buffer count)

        So each buffer must be n_photons_per_buffer + 256*2 photons deep


        Args:
            n_photons_per_buffer: Rotate the buffer approximately every this many photons, underset this by 512 if the
            actual rate is critical. Will be limited to about 100k. FLAT_PHOTON_BUFSIZE-512 (see HLS C code).
            buffer_time_ms: Rotate the buffer approximately every this many ms. Valid range is ~ 0.5 - 1000 us.

        Returns: The buffer. Don't write to this. Don't free this. Glances at the returned buffer should see new
        photons appear in batches of 512 and the buffer count increment in steps of 512, both until a final batch of up
        to 512. Then the other buffer should increment while the values in the inactive buffer stay the same.

        """
        if not self.register_map.CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf

        if self._buf is not None:
            x = self._buf
            self._buf = None
            x.freebuffer()

        self.register_map.IP_IER.CHAN0_INT_EN = 1  # enable the done interrupt
        self.register_map.GIER = 1

        self.read(0x0C)  # clear any existing interrupt (COR)
        self.read(0x28)  # clear the occupancy valid flag by reading the buffer occupancy (COR)

        self._buf = buffer.allocate((self.N_PHOTON_BUFFERS, self.PHOTON_BUFF_N),
                                    dtype=self.PHOTON_PACKED_DTYPE, cacheable=cacheable_buffer)

        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())

        self.buffer_interval = buffer_time_ms

        n_photons_per_buffer = int(n_photons_per_buffer)
        if n_photons_per_buffer > self.PHOTON_BUFF_N - 512:
            getLogger(__name__).warning(f'Requesting too large a buffer, buffer will be rotated every '
                                        f'={self.PHOTON_BUFF_N} photons')

        self.write(0x38, min(max(int(n_photons_per_buffer), 1), self.PHOTON_BUFF_N - 512))
        self.register_map.CTRL.AUTO_RESTART = 1
        self.register_map.CTRL.AP_START = 1
        return self._buf

    def configure(self):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture()


class PhotonPacketizer(DefaultIP):
    MAX_PHOTON_PACKET_N = 262144  # Must not exceed dma buffer size
    PHOTON_DTYPE = _PHOTON_DTYPE
    PHOTON_PACKED_DTYPE = np.uint64
    bindto = ['mazinlab:mkidgen3:photon_fifo_packetizer:0.1']

    @staticmethod
    def pack_photons(x, out=None):
        ret = np.zeros(x.size, dtype=PhotonMAXI.PHOTON_PACKED_DTYPE) if out is None else out
        ret[:] = (((x['time'] << 12) | x['id']) << 16) | x['phase'].astype(np.uint16)
        return ret

    @staticmethod
    def unpack_photons(x, out=None, n=0):
        """
        Unpack packed photons, optionally accumulating them into an existing output array

        Args:
            x: an array of packed photons
            out: optional, an array of type PhotonMAXI.PHOTON_DTYPE with the shape of x, but see n
            n: optional, an index into out to insert unpacked photons, the first axis is used if >1d an
                IndexError is raised if there is insufficient space.

        Returns: the unpacked photon array

        """
        if out is None:
            n = 0
        elif x.shape[0] + n > out.shape[0]:
            raise IndexError('Output array is too small')

        ret = np.zeros(x.shape, dtype=PhotonMAXI.PHOTON_DTYPE) if out is None else out
        sl = slice(n, n + x.shape[0])
        ret['phase'][sl] = x & 0xffff
        ret['time'][sl] = x >> 28
        ret['id'][sl] = (x >> 16) & 0xfff
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

        // 0x10 : Data signal of photons_per_buf
        //        bit 16~0 - photons_per_buf[16:0] (Read/Write)
        //        others   - reserved
        // 0x18 : Data signal of time_shift
        //        bit 4~0 - time_shift[4:0] (Read/Write)
        //        others  - reserved
        """
        super().__init__(description=description)

    @property
    def buffer_count_interval(self):
        return self.read(0x10) & 0x1ffff

    @property
    def buffer_interval(self):
        return 2 ** (self.read(0x18) & 0x1f) / 1e3

    @buffer_interval.setter
    def buffer_interval(self, interval_ms):
        """Permissible range is about 0.5 - 1000 ms and will be clipped"""
        l2_buffer_shift = np.round(np.log2(interval_ms * 1000))
        _buffer_time_ms = 2 ** l2_buffer_shift / 1000
        if l2_buffer_shift < 9 or l2_buffer_shift > 20:
            l2_buffer_shift = min(max(l2_buffer_shift, 9), 20)
            getLogger(__name__).warning(f'Requested rotation interval ({interval_ms:.2f} ms) unsupported, '
                                        f'using {2 ** l2_buffer_shift / 1000:.2f}')
        elif abs(_buffer_time_ms - interval_ms) >= .01:
            getLogger(__name__).info(f'Requested rotation interval {interval_ms:.2f} ms rounded to '
                                     f'{_buffer_time_ms:.2f} ms')
        else:
            getLogger(__name__).info(f'Setting rotation interval to {_buffer_time_ms:.2f} ms')

        self.write(0x18, int(l2_buffer_shift))

    def packetize(self, n_photons_per_buffer=100000, buffer_time_ms=4.096):
        """
        Initiate packetization of photons.

        Note that proper functionality of the core requires that at least two photons arrive at least buffer_time_ms
        apart or more than n_photons_per_buffer are seen. If neither of those conditions is met the core will not
        complete. If fewer than 512 photons are seen AND they all arrive within one buffer interval no data will even
        be written into the buffer. Both of these scenarios are essentially impossible and can be resolved by setting
        an arbitrarily low event trigger threshold in the trigger core.

        Args:
            n_photons_per_buffer: Rotate the buffer approximately every this many photons, underset this by 512 if the
            actual rate is critical. Will be limited to about 100k. FLAT_PHOTON_BUFSIZE-512 (see HLS C code).
            buffer_time_ms: Rotate the buffer approximately every this many ms. Valid range is ~ 0.5 - 1000 us.

        """
        self.buffer_interval = buffer_time_ms

        n_photons_per_buffer = int(n_photons_per_buffer)
        if n_photons_per_buffer > self.MAX_PHOTON_PACKET_N:
            getLogger(__name__).warning(f'Requesting too large a packet, using packet of '
                                        f'{self.MAX_PHOTON_PACKET_N} photons.')

        self.write(0x10, min(max(n_photons_per_buffer-2, 1), self.MAX_PHOTON_PACKET_N-2))

    def configure(self):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.packetize()
