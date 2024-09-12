import numpy as np
import pynq.buffer as buffer
from pynq import allocate, DefaultIP, DefaultHierarchy
import time
import asyncio
from logging import getLogger
from typing import Iterable

from ..mkidpynq import N_IQ_GROUPS, MAX_CAP_RAM_BYTES, PL_DDR4_ADDR, \
    check_description_for  # config, overlay details
from ..system_parameters import N_CHANNELS, N_IQ_GROUPS, N_PHASE_GROUPS, channel_to_iqgroup, channel_to_phasegroup, iqgroup_to_channel, phasegroup_to_channel, ADC_INPUT_WARN
from ..util import ps_ram_sane, format_bytes
from ..interrupts import ThreadedPLInterruptManager


class FilterIQ(DefaultIP):
    """
    """
    bindto = ['mazinlab:mkidgen3:filter_iq:0.1']
    ADDR_KEEP = 0x10  # 8 words
    ADDR_LASTGRP = 0x34  # 8 bits

    def __init__(self, description):
        super().__init__(description=description)

    @property
    def n_kept(self):
        """ Return the number of groups being preserved"""
        return len(self.keep)

    @property
    def keep(self):
        """
        Get the IQ groups that are being preserved. Returns an iterable of resonator channel group numbers (0-255).

        e.g. i in keep => resonator channels i*8 ... i*8+7 are preserved
        """
        ret = []
        for i in range(8):
            k = self.read(self.ADDR_KEEP + 0x4 * i)
            ret += [j + 32 * i for j in range(32) if k & (1 << j)]
        return ret

    def keep_channels(self, channels):
        """Like keep= but assume channels"""
        self.keep = channel_to_iqgroup(channels)

    @keep.setter
    def keep(self, groups: Iterable):
        """
        Tell the block to preserve IQ groups. Set to an iterable of resonator channel IQ group numbers (0-255). If
        iterable contains values > 255 it is assumed numbers indicate resonator channels and the necessary groups
        will be computed. 'all' may be used as a shortcut.

        IQs are processed in 256 groups of 8, so to capture the IQ values of resonator channels 0, 37,
        and 2047 groups must be set to either 'all' or should
        include e.g. (0, 4, 255). Note that this will cause IQs for channels 0-7, 32-39, and 2040-2047 to be captured.

        len(groups) must be a multiple of 2
        """

        # Determine keep, keep a group if all or if the group number is in groups
        keep = np.zeros(8, dtype=np.uint32)
        if isinstance(groups, str):
            if groups.lower() != 'all':
                raise ValueError("The only legal string for keep is 'all'")
            keep += 0xFFFFFFFF
            last = N_IQ_GROUPS-1
        else:
            groups = set(groups)

            if len(groups) % 2:
                groups.add(set(range(256)).difference(groups).pop())
            last = int(max(groups))
            if max(groups) > N_IQ_GROUPS-1 or min(groups) < 0:
                raise ValueError(f'Groups must be in range 0-{N_IQ_GROUPS-1}')

            for g in groups:
                i = g // 32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the correct bit

        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP + 0x4 * i, int(k))
        self.write(self.ADDR_LASTGRP, last)


class FilterPhase(DefaultIP):
    bindto = ['mazinlab:mkidgen3:filter_phase:0.2']
    ADDR_KEEP = 0x10  # 4 words
    ADDR_LASTGRP = 0x24  # 7 bits

    def __init__(self, description):
        super().__init__(description=description)

    @property
    def n_kept(self):
        """ Return the number of groups being preserved"""
        return len(self.keep)

    @property
    def keep(self):
        """
        Get the phase groups that are being preserved. Returns an iterable of resonator channel group numbers (0-127).

        e.g. i in keep => resonator channels i*16 ... i*16+15 are preserved
        """
        ret = []
        for i in range(4):
            k = self.read(self.ADDR_KEEP + 0x4 * i)
            ret += [j + 32 * i for j in range(32) if k & (1 << j)]
        return ret

    def keep_channels(self, channels):
        """Like keep= but assume channels"""
        self.keep = channel_to_phasegroup(channels)

    @keep.setter
    def keep(self, groups):
        """
        Tell the block to preserve phase groups. Set to an iterable of resonator channel phase group numbers (0-129). If
        iterable contains values > 127 it is assumed numbers indicate resonator channels and the necessary groups
        will be computed. 'all' may be used as a shortcut.

        Phases are captured in 128 groups of 16, so to capture the values of resonator channels 0, 37,
        and 2047 groups must be set to either 'all' or should
        include (0, 2, 127). Note that this will cause phases for channels 0-16, 32-48, and 2032-2047 to be captured.
        """

        # Determine keep, keep a group if all or if the group number is in groups
        keep = np.zeros(4, dtype=np.uint32)
        if isinstance(groups, str):
            if groups.lower() != 'all':
                raise ValueError("The only legal string for keep is 'all'")
            keep += 0xFFFFFFFF
            last = N_PHASE_GROUPS-1
        else:
            if len(groups) % 2:
                groups.add(set(range(512)).difference(groups).pop())
            last = int(max(groups))

            if max(groups) > N_PHASE_GROUPS-1 or min(groups) < 0:
                raise ValueError(f'Groups must be in range 0-{N_PHASE_GROUPS-1}')

            for g in groups:
                i = g // 32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the correct bit

        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP + 0x4 * i, int(k))
        self.write(self.ADDR_LASTGRP, last)


class CaptureHierarchy(DefaultHierarchy):
    """
    Capture Hierarchy.

    Driver supports capture at up to five locations. They are numbered 1-5 in the source map in accordance with their
    position on the axis switch. TODO: Need to standardize these keys and names
    switch port: tap_name: description:
    0: 'adc': connected to the output of the adc
    1: 'rawiq': connected to the output of bin to res, can be used to look at the OPFB bins before the DDC
    2: 'ddciq': connected to the output of the DDC before the lowpass
    3: 'filtphase': connected to the output of matched filters
    4: 'debugiq': placeholder for a filter_iq IP connected to 4th switch port
    5. 'debugpphase' placeholder for a filter_phase IP connected to 5th switch port
    """
    IQ_MAP = {'rawiq': 0, 'ddciq': 1, 'debugiq': 2}  #tap name MUST include 'iq' and shall not include 'phase'
    PHASE_MAP = {'filtphase': 0, 'debugphase': 1} #tap name MUST include 'phase' and shall not include 'iq'
    SOURCE_MAP = dict(adc=0, rawiq=1, ddciq=2, filtphase=3, debugiq=4, debugphase=5)
    USE_CACHEABLE_BUFFERS = True

    def __init__(self, description):
        super().__init__(description)
        self.filter_iq = {}
        self.filter_phase = {}

        switchloc = getattr(self, 'Switchboard', None) or getattr(self, 'switchboard', None) or self
        self.switch = getattr(switchloc, 'axis_switch_0', None) or getattr(switchloc, 'axis_switch', None)

        if not hasattr(self, 'axis2mm'):
            self.axis2mm = self.axis2mm_0

        self._fetch_filter_blocks()

    @staticmethod
    def checkhierarchy(description):
        found = check_description_for(description, ('xilinx.com:axis_switch', 'mazinlab:mkidgen3:filter_iq',
                                                    'xilinx.com:module_ref:axis2mm'))
        return description['fullpath'] == 'capture' and bool(len(found['xilinx.com:module_ref:axis2mm']))

    def _fetch_filter_blocks(self):
        i = 0
        while True:
            try:
                self.filter_iq[i] = getattr(self, f'filter_iq_{i}')
                i += 1
            except AttributeError:
                break
        if not i:
            getLogger(__name__).warning(f'Capture does not support iq capture')
        else:
            getLogger(__name__).debug(f'Found {i} iq capture taps')

        for n, i in self.IQ_MAP.items():
            self.filter_iq[n] = self.filter_iq.get(i, None)

        i = 0
        while True:
            try:
                self.filter_phase[i] = getattr(self, f'filter_phase_{i}')
                i += 1
            except AttributeError:
                break
        if not i:
            getLogger(__name__).warning(f'Capture does not support phase capture')
        else:
            getLogger(__name__).debug(f'Found {i} phase capture taps')

        for n, i in self.PHASE_MAP.items():
            self.filter_phase[n] = self.filter_phase.get(i, None)

        base = pbase = 1
        iqs = list(self.IQ_MAP.items())
        iqs.sort(key=lambda x: x[1])
        for k, v in iqs:
            if self.filter_iq.get(k, None):
                self.SOURCE_MAP[k] = v + base
                pbase = v + base

        ph = list(self.PHASE_MAP.items())
        ph.sort(key=lambda x: x[1])
        for k, v in ph:
            self.SOURCE_MAP[k] = v + pbase + 1

    def flush(self, n):
        """Capture n*64 bytes to flush a fifo"""
        buffer = allocate(n, dtype='u64', target=self.ddr4_0)
        self.axis2mm.addr = buffer.device_address
        self.axis2mm.len = 64 * n
        self.axis2mm.start(continuous=False, increment=True)
        del buffer

    def looping_capture(self, source: str, n: int, buffers: buffer.PynqBuffer, callback):
        getLogger(__name__).warning('Looping capture untested, exercise case')
        if self.switch is not None:
            self.switch.set_driver(slave=self.SOURCE_MAP[source], commit=True)
        if n % 64:
            raise ValueError('Can only capture in multiples of 64 bytes')
        if not self.axis2mm.ready:
            raise IOError("capture core not ready, this shouldn't happen."
                          " Try calling .axis2mm.abort() followed by .axis2mm.clear_error()"
                          " then try a small throwaway capture (data order may not be aligned in the first capture "
                          "after a reset).")

        i = 0
        self.axis2mm.addr = buffers[0].device_address
        self.axis2mm.len = n
        self.axis2mm.start(continuous=True, increment=True)

        while not self.terminate_looped_capture:
            self.wait()
            stat = self.cmd_ctrl_reg
            if stat['r_err']:
                getLogger(__name__).error(f'Aborting capture loop due to error: {stat}')
                break
            else:
                callback(buffers[i])
                i = i + 1 if i < len(buffers) else 0
                self.axis2mm.addr = buffers[i].device_address
                self.axis2mm.len = n
        self.abort()

    def _capture(self, source, n, buffer_addr):
        if self.switch is not None:
            self.switch.set_driver(slave=self.SOURCE_MAP[source], commit=True)
        if n % 64:
            raise ValueError('Can only capture in multiples of 64 bytes')
        if not self.axis2mm.ready:
            raise IOError("capture core not ready, this shouldn't happen."
                          " Try calling .axis2mm.abort() followed by .axis2mm.clear_error()"
                          " then try a small throwaway capture (data order may not be aligned in the first capture "
                          "after a reset).")

        self.axis2mm.addr = buffer_addr
        self.axis2mm.len = n
        _, e = ThreadedPLInterruptManager.get_monitor(self.axis2mm._interrupts['o_int']['fullpath'], id='capheir')
        e.clear()
        getLogger(__name__).debug(f'Starting capture of {format_bytes(n)} ({n // 64} beats) to address {hex(buffer_addr)} from '
                                  f'source {source}: \n Interrupt Status:' +
                                  str(ThreadedPLInterruptManager.get_status(self.axis2mm._interrupts['o_int']['fullpath'])))
        self.axis2mm.start(continuous=False, increment=True)

    def is_ready(self):
        return self.axis2mm.ready

    def keep_channels(self, tap, channels):
        if isinstance(channels, str):
            if channels.lower() == 'all':
                channels = list(range(N_CHANNELS))
            else:
                raise ValueError(f'Invalid channel type: {channels}')
        if tap in self.IQ_MAP:
            return self.filter_iq[tap].keep_channels(channels)
        elif tap in self.PHASE_MAP:
            return self.filter_phase[tap].keep_channels(channels)

    def kept_channels(self, tap):
        if tap in self.IQ_MAP:
            return iqgroup_to_channel(self.filter_iq[tap].keep)
        elif tap in self.PHASE_MAP:
            return phasegroup_to_channel(self.filter_phase[tap].keep)
        else:
            return (1,)

    def capture(self, n, tap, groups='all', wait=True):
        """
        Do a capture to PL Ram
        Args:
            n: the number of samples to capture, will be truncated to the nearest capturable amount.
            tap: an iq, phase, or adc capture tap location. 'adc' or see IQ_MAP and PHASE_MAP keys. Raises value error
            if invalid.
            groups: Which sample groups to capture (only relevant for iq and phase)
            wait: wait for the capture to complete. If not waiting it is necessary to call axis2mm.errors()
            to check for any capture errors. If there are errors axis2mm.clear_errors() will be called.

        Returns: The capture buffer or a dictionary of capture errors

        """
        try:
            assert (int(n)-n) == 0
            n = int(n)
        except:
            raise TypeError('n must be effectively an integer')

        if tap in self.IQ_MAP:
            buf = self.capture_iq(n, tap_location=tap, duration=False, groups=groups, wait=wait)
        elif tap in self.PHASE_MAP:
            buf = self.capture_phase(n, tap_location=tap, duration=False, groups=groups, wait=wait)
        elif tap == 'adc':
            buf = self.capture_adc(n, complex=False, duration=False, wait=wait)
        else:
            valid = ('adc',) + tuple(self.IQ_MAP.keys()) + tuple(self.PHASE_MAP.keys())
            raise ValueError(f'{tap} is not a valid capture location from {valid}')

        if wait:
            errors = self.axis2mm.errors()
            if errors:
                del buf
                self.axis2mm.clear_error()
                return errors

        return buf

    def capture_iq(self, n, groups='all', tap_location='ddciq', duration=False, wait=True):
        """
        potentially valid tap locations are the keys of CaptureHierarchy.IQ_MAP
        if buffer is None one will be allocated, if groups is None it will not be set
        """
        if duration:
            # n samples = t[ms] * 2e6[samples/sec]
            n = int(np.floor(n * 2e-3 * 1e6))

        if n <= 0:
            raise ValueError('Must request at least 1 sample')

        if self.filter_iq.get(tap_location, None) is None:
            raise ValueError(f'Unsupported IQ capture location: {tap_location}')

        if groups is not None:
            self.filter_iq[tap_location].keep = groups
        n_groups = len(self.filter_iq[tap_location].keep)

        # each group is 8 IQ (32 bytes)
        capture_bytes = n * n_groups * 32

        try:
            buffer = allocate((n, n_groups * 8, 2), dtype='i2', target=self.ddr4_0,
                              cacheable=self.USE_CACHEABLE_BUFFERS if wait else False)
        except RuntimeError:
            getLogger(__name__).warning(f'Insufficient space for requested samples.')
            raise RuntimeError('Insufficient free space')
        addr = buffer.device_address

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = capture_bytes / 1024 ** 2
        datarate_mbps = 32 * 512 * n_groups / 256
        captime = datavolume_mb / datarate_mbps

        msg = (f"Capturing {format_bytes(capture_bytes)} of data @ {datarate_mbps:.1f} MiBps. "
               f"ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        self._capture(tap_location, capture_bytes, addr)

        if wait:
            if isinstance(wait, bool):
                wait = {}
            if 'duration' not in wait:
                wait['duration'] = captime
            self.wait(**wait)

        buffer.invalidate()

        return buffer

    def capture_adc(self, n, duration=False, complex=False, wait=True):
        """
        samples are captured in multiples of 8 will be clipped as necessary

        Set complex to return a numpy complex array of the result. This implies a copy out of the buffer so care is
        required with memory sizes (i.e. will use 5x standard memory in PS DDR).

        """
        if complex and not wait:
            raise RuntimeError('Complex return not supported with immediate return.')

        if n <= 0:
            raise ValueError('Must request at least 1 sample')

        if duration:
            # n samples = t[ms] * 4.096e9[samples/sec]
            n = int(np.floor(n * 1e-3 * 4.096e9))

        n -= n % 8

        if n <= 0:
            n = 8

        # Compute capturesize, this is the number of adc samples to be written
        capture_bytes = n * 4

        if complex and not ps_ram_sane(n*8):
            raise RuntimeError('Not enough RAM to copy capture to complex')

        try:
            buffer = allocate((n, 2), dtype='i2', target=self.ddr4_0,
                              cacheable=self.USE_CACHEABLE_BUFFERS if wait else False)
        except RuntimeError:
            getLogger(__name__).warning(f'Insufficient space for requested samples.')
            raise RuntimeError('Insufficient free space')
        addr = buffer.device_address

        datavolume_mb = capture_bytes / 1024 ** 2
        datarate_mbps = 32 * 512
        captime = datavolume_mb / datarate_mbps

        msg = (f"Capturing {datavolume_mb:.1f} MB of data @ {datarate_mbps} MB/s. "
               f"ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        self._capture('adc', capture_bytes, addr)

        if wait or complex:
            if isinstance(wait, bool):
                wait = {}
            if 'duration' not in wait or (complex and wait['duration'] < captime):
                wait['duration'] = captime
            self.wait(**wait)

        buffer.invalidate()

        if complex:
            d = np.zeros(n, dtype=np.complex64)
            d.real[:] = buffer[:, 0]
            d.imag[:] = buffer[:, 1]
            del buffer
            return d
        else:
            return buffer

    def wait(self, duration=0, interrupt=False):
        """Wait for a capture to complete, use interrupt on axis2mm by default, otherwise sleep for duration"""
        if not interrupt:
            while not self.axis2mm.complete:
                time.sleep(duration / 5000)
        else:
            _, e = ThreadedPLInterruptManager.get_monitor(self.axis2mm._interrupts['o_int']['fullpath'], id='capheir')
            e.wait()
            # self.axis2mm.abort()
            # e.clear()

    def capture_phase(self, n, groups='all', duration=False, tap_location='filtphase', wait:(bool,dict)=True):
        """
        samples are captured in multiples of 16 will be clipped ad necessary
        groups is 0-127 or all, None will leave the filter unchanged

        wait is args to selt.wait
        """
        if self.filter_phase.get(tap_location, None) is None:
            getLogger(__name__).error(f'Bitstream does not support phase capture at tap {tap_location}')
            return None

        if duration:
            # n samples = t[ms] * 1e6[samples/sec]
            n = int(np.floor(n * 1e-3 * 1e6))

        if n <= 0:
            raise ValueError('Must request at least 1 sample')

        if groups is not None:
            self.filter_phase[tap_location].keep = groups
        n_groups = len(self.filter_phase[tap_location].keep)

        # each group is 16 phases (32 bytes)
        capture_bytes = n * 2 * n_groups * 16

        try:
            buffer = allocate((n, n_groups * 16), dtype='i2', target=self.ddr4_0,
                              cacheable=self.USE_CACHEABLE_BUFFERS if wait else False)
        except RuntimeError:
            getLogger(__name__).warning(f'Insufficient space for requested samples.')
            raise RuntimeError('Insufficient free space')
        addr = buffer.device_address

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = capture_bytes / 1024 ** 2
        datarate_mbps = 32 * 512 / 4 * n_groups / 128  # phases arrive 4@512 so the filter outputs every 4 clocks
        captime = datavolume_mb / datarate_mbps

        msg = (f"Capturing ~{datavolume_mb:.2f} MB of data @ {datarate_mbps:.1f} MBps. "
               f"ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        self._capture(tap_location, capture_bytes, addr)

        if wait:
            if isinstance(wait, bool):
                wait = {}
            if 'duration' not in wait:
                wait['duration'] = captime
            self.wait(**wait)

        buffer.invalidate()

        return buffer

    def ddc_compare_cap(self, n_points=1024):
        """A helper function to capture data just after bin2res and after the ddc"""
        x = self.capture_iq(n_points, 'all', tap_location='rawiq')
        riq = np.array(x)
        del x
        x = self.capture_iq(n_points, 'all', tap_location='ddciq')
        iq = np.array(x)
        del x
        riq = riq[..., 0] + riq[..., 1] * 1j
        iq = iq[..., 0] + iq[..., 1] * 1j
        return riq, iq

    def cap_cordic_compare(self, n_points=1024):
        """A helper function to capture data just after the ddc and after matched filters"""
        x = self.capture_iq(n_points, 'all', tap_location='ddciq')
        riq = np.array(x)
        del x
        x = self.capture_phase(n_points, 'all', tap_location='filtphase')
        phase = np.array(x)
        del x
        riq = riq[..., 0] + riq[..., 1] * 1j
        return riq, phase


class _AXIS2MM:
    @property
    def cmd_ctrl_reg(self):
        reg = self.read(0)
        names = ('r_busy', 'r_err', 'r_complete', 'r_continuous', 'r_increment_n',
                 'r_tlast_syncd_n', 'decode_error', 'slave_error', 'overflow_error',
                 'aborting', 'fifo_len', 'abort')
        bits = (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, (20, 16), (15, 8))

        def cvrt(x, bits):
            if isinstance(bits, int):
                return bool(1 & (reg >> bits))
            else:
                return (reg >> bits[1]) & (2 ** (bits[0] - bits[1] + 1) - 1)

        return {k: cvrt(reg, v) for k, v in zip(names, bits)}

    @property
    def addr(self):
        """5 bytes"""
        return self.read(0x10) | ((self.read(0x14) & 0xffff) << 32)

    @addr.setter
    def addr(self, x):
        #         x&=(2**35-1)
        #         self.write(0x10, x&0xffffffff)
        #         self.write(0x14, (x>>32)&0xffffffff)
        self.write(0x10, x.to_bytes(8, 'little'))

    @property
    def len(self):
        return self.read(0x18)

    @len.setter
    def len(self, x):
        """A number of bytes, not beats!"""
        self.write(0x18, x)

    @property
    def ready(self):
        stat = self.cmd_ctrl_reg
        return not stat['r_busy'] and not stat['r_err'] and not stat['aborting']

    @property
    def complete(self):
        return True if ((self.read(0) >> 29) & 1) else False

    def abort(self):
        #         self.write(0, (self.read(0)^0xff00)|0x2600)
        self.write(0, 0x26000000)

    def clear_error(self):
        self.write(0, 0x40000000)

    def errors(self):
        """Return a dictionary of errors if there are errors else None"""
        x = self.cmd_ctrl_reg
        if x['r_err']:
            return {k:x[k] for k in ('r_err', 'decode_error', 'slave_error', 'overflow_error')}
        else:
            return None

    def start(self, continuous=False, increment=True):
        if not self.ready:
            raise IOError('Not ready, need to abort and/or clear errors')
        x = 0x80000000  # start and do not clear error
        x |= continuous << 28
        x |= (not increment) << 27
        self.write(0, x)


class AXIS2MMIP(DefaultIP, _AXIS2MM):
    bindto = ['xilinx.com:module_ref:axis2mm: 1.0']

    def __init__(self, description):
        super().__init__(description=description)


class AXIS2MMHier(DefaultHierarchy, _AXIS2MM):
    def __init__(self, description):
        super().__init__(description)
        self._core = self.S_AXIL
        self.read = self._core.read
        self.mmio = self._core.mmio
        self.write = self._core.write

    @staticmethod
    def checkhierarchy(description):
        try:
            t = description['ip']['S_AXIL']['type']
        except KeyError:
            return False
        return t == 'xilinx.com:module_ref:axis2mm:1.0'
