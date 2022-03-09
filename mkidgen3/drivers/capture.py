import numpy as np
from pynq import allocate, DefaultIP, DefaultHierarchy
import time
import mkidgen3.mkidpynq
from mkidgen3.mkidpynq import N_IQ_GROUPS, MAX_CAP_RAM_BYTES, PL_DDR4_ADDR, check_description_for  # config, overlay details
from logging import getLogger


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

    @keep.setter
    def keep(self, groups):
        """
        Tell the block to preserve IQ groups. Set to an iterable of resonator channel IQ group numbers (0-255). If
        iterable contains values > 255 it is assumed numbers indicate resonator channels and the necessary groups
        will be computed. 'all' may be used as a shortcut.

        IQs are processed in 256 groups of 8, so to capture the IQ values of resonator channels 0, 37,
        and 2047 groups must be set to either 'all' or should
        include (0, 4, 255). Note that this will cause IQs for channels 0-7, 32-39, and 2040-2047 to be captured.
        """

        # Determine keep, keep a group if all or if the group number is in groups
        keep = np.zeros(8, dtype=np.uint32)
        if isinstance(groups, str):
            if groups.lower() != 'all':
                raise ValueError("The only legal string for keep is 'all'")
            keep += 0xFFFFFFFF
            last = 255
        else:

            if max(groups) > 255:
                groups = set([g // 8 for g in groups])  # convert channel to group
            last = max(groups)

            if len(groups) % 2:
                raise ValueError('Groups must be captured in multiples of two')

            if max(groups) > 255 or min(groups) < 0:
                raise ValueError(f'Groups must be in range 0-255')

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
            last = 127
        else:
            if max(groups) > 127:
                groups = set([g // 16 for g in groups])  # convert channel to group
            last = max(groups)

            if len(groups) % 2:
                raise ValueError('Groups must be captured in multiples of two')

            if max(groups) > 127 or min(groups) < 0:
                raise ValueError(f'Groups must be in range 0-127')

            for g in groups:
                i = g // 32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the correct bit

        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP + 0x4 * i, int(k))
        self.write(self.ADDR_LASTGRP, last)


class CaptureHierarchy(DefaultHierarchy):
    IQ_MAP = {'rawiq': 0, 'iq': 1, 'ddciq': 2}
    SOURCE_MAP = dict(adc=0, iq=2, rawiq=1, phase=3)

    def __init__(self, description):
        super().__init__(description)

        self.filter_iq = {}
        for n, i in self.IQ_MAP.items():
            try:
                dev = getattr(self, f'filter_iq_{i}')
            except AttributeError:
                dev = None
                getLogger(__name__).debug(f'Capture of {n} not supported')
            self.filter_iq[n] = dev

        self.switch = getattr(self,'axis_switch_0', None) or getattr(self,'axis_switch', None)
        try:
            self.filter_phase = self.filter_phase_0
        except AttributeError:
            self.filter_phase = None
        self.axis2mm = self.axis2mm_0

    @staticmethod
    def checkhierarchy(description):
        found = check_description_for(description, ('xilinx.com:axis_switch', 'mazinlab:mkidgen3:filter_iq',
                                                    'xilinx.com:module_ref:axis2mm'))
        return description['fullpath'] == 'capture' and bool(len(found['xilinx.com:module_ref:axis2mm']))

    def flush(self, n):
        """Capture n*64 bytes to flush a fifo"""
        buffer = allocate(n, dtype='u64', target=self.ddr4_0)
        self.axis2mm.addr = buffer
        self.axis2mm.len = 64*n
        self.axis2mm.start(continuous=False, increment=True)
        buffer.freebuffer()

    def _capture(self, source, n, buffer):
        if self.switch is not None:
            self.switch.set_driver(slave=self.SOURCE_MAP[source], commit=True)
        if n % 64:
            raise ValueError('Can only capture in multiples of 64 bytes')
        if not self.axis2mm.ready:
            raise IOError("capture core not ready, this shouldn't happen."
                          " Try calling .axis2mm.abort() followed by .axis2mm.clear_error()"
                          " then try a small throwaway capture (data order may not be aligned in the first capture "
                          "after a reset).")
        getLogger(__name__).debug(f'Starting capture of {n} bytes ({n // 64} beats) to address {hex(buffer)} from '
                                  f'source {source}')
        self.axis2mm.addr = buffer
        self.axis2mm.len = n
        self.axis2mm.start(continuous=False, increment=True)

    def capture_iq(self, n, groups='all', tap_location='iq', duration=False):
        """
        potentially valid tap locations are the keys of CaptureHierarchy.IQ_MAP
        if buffer is None one will be allocated
        """
        if duration:
            # n samples = t[ms] * 2e6[samples/sec]
            n = int(np.floor(n * 2e-3 * 1e6))

        if n <= 0:
            raise ValueError('Must request at least 1 sample')

        if self.filter_iq.get(tap_location, None) is None:
            raise ValueError(f'Unsupported IQ capture location: {tap_location}')
        self.filter_iq[tap_location].keep = groups
        n_groups = len(self.filter_iq[tap_location].keep)

        # each group is 8 IQ (32 bytes)
        capture_bytes = n * n_groups * 32

        try:
            buffer = allocate((n, n_groups * 8, 2), dtype='i2', target=self.ddr4_0)
        except RuntimeError:
            getLogger(__name__).warning(f'Insufficient space for requested samples.')
            raise RuntimeError('Insufficient free space')
        addr = buffer.device_address

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = capture_bytes / 1024 ** 2
        datarate_mbps = 32 * 512 * n_groups / 256
        captime = datavolume_mb / datarate_mbps

        msg = (f"Capturing {datavolume_mb:.1f} MB of data @ {datarate_mbps:.1f} MBps. "
               f"ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        self._capture(tap_location, capture_bytes, addr)
        time.sleep(captime)
        return buffer

    def capture_adc(self, n, duration=False):
        """
        samples are captured in multiples of 8 will be clipped as necessary
        """
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

        try:
            buffer = allocate((n, 2), dtype='i2', target=self.ddr4_0)
        except RuntimeError:
            getLogger(__name__).warning(f'Insufficient space for requested samples.')
            raise RuntimeError('Insufficient free space')
        addr = buffer.device_address

        datavolume_mb = capture_bytes / 1024 ** 2
        datarate_mbps = 32 * 512
        captime = datavolume_mb / datarate_mbps

        msg = (f"Capturing {datavolume_mb} MB of data @ {datarate_mbps} MB/s. "
               f"ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        self._capture('adc', capture_bytes, addr)
        time.sleep(captime)
        return buffer

    def capture_phase(self, n, groups='all', duration=False):
        """
        samples are captured in multiples of 16 will be clipped ad necessary
        groups is 0-127 or all
        """
        if self.filter_phase is None:
            getLogger(__name__).error('Bitstream does not support phase capture')
            return None

        if duration:
            # n samples = t[ms] * 1e6[samples/sec]
            n = int(np.floor(n * 1e-3 * 1e6))

        if n <= 0:
            raise ValueError('Must request at least 1 sample')

        self.filter_phase.keep = groups
        n_groups = len(self.filter_phase.keep)

        # each group is 16 phases (32 bytes)
        capture_bytes = n * 2 * n_groups * 16

        try:
            buffer = allocate((n, n_groups * 16), dtype='i2', target=self.ddr4_0)
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

        self._capture('phase', capture_bytes, addr)
        time.sleep(captime)

        return buffer


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

    def abort(self):
        #         self.write(0, (self.read(0)^0xff00)|0x2600)
        self.write(0, 0x26000000)

    def clear_error(self):
        self.write(0, 0x40000000)

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
