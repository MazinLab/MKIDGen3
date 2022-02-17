import numpy as np
from pynq import allocate, DefaultIP, DefaultHierarchy
import time
import mkidgen3.mkidpynq
from mkidgen3.mkidpynq import N_IQ_GROUPS, MAX_CAP_RAM_BYTES, PL_DDR4_ADDR, check_description_for  # config, overlay details
from logging import getLogger


class IQCapture(DefaultIP):
    """
    // control
    // 0x00 : Control signals
    //        bit 0  - ap_start (Read/Write/COH)
    //        bit 1  - ap_done (Read/COR)
    //        bit 2  - ap_idle (Read)
    //        bit 3  - ap_ready (Read/COR)
    //        bit 7  - auto_restart (Read/Write)
    //        others - reserved
    // 0x04 : Global Interrupt Enable Register
    //        bit 0  - Global Interrupt Enable (Read/Write)
    //        others - reserved
    // 0x08 : IP Interrupt Enable Register (Read/Write)
    //        bit 0  - enable ap_done interrupt (Read/Write)
    //        bit 1  - enable ap_ready interrupt (Read/Write)
    //        others - reserved
    // 0x0c : IP Interrupt Status Register (Read/TOW)
    //        bit 0  - ap_done (COR/TOW)
    //        bit 1  - ap_ready (COR/TOW)
    //        others - reserved
    // 0x10 : Data signal of keep
    //        bit 31~0 - keep[31:0] (Read/Write)
    // 0x14 : Data signal of keep
    //        bit 31~0 - keep[63:32] (Read/Write)
    // 0x18 : Data signal of keep
    //        bit 31~0 - keep[95:64] (Read/Write)
    // 0x1c : Data signal of keep
    //        bit 31~0 - keep[127:96] (Read/Write)
    // 0x20 : Data signal of keep
    //        bit 31~0 - keep[159:128] (Read/Write)
    // 0x24 : Data signal of keep
    //        bit 31~0 - keep[191:160] (Read/Write)
    // 0x28 : Data signal of keep
    //        bit 31~0 - keep[223:192] (Read/Write)
    // 0x2c : Data signal of keep
    //        bit 31~0 - keep[255:224] (Read/Write)
    // 0x30 : reserved
    // 0x34 : Data signal of total_capturesize
    //        bit 31~0 - total_capturesize[31:0] (Read/Write)
    // 0x38 : Data signal of total_capturesize
    //        bit 2~0 - total_capturesize[34:32] (Read/Write)
    //        others  - reserved
    // 0x3c : reserved
    // 0x40 : Data signal of capturesize
    //        bit 26~0 - capturesize[26:0] (Read/Write)
    //        others   - reserved
    // 0x44 : reserved
    // 0x48 : Data signal of iqout
    //        bit 31~0 - iqout[31:0] (Read/Write)
    // 0x4c : Data signal of iqout
    //        bit 31~0 - iqout[63:32] (Read/Write)
    // 0x50 : reserved
    """
    bindto = ['mazinlab:mkidgen3:iq_capture:1.34']
    ADDR_KEEP = 0x10  # 8 words
    ADDR_TOTALCAPTURESIZE = 0x34  # 2 words (only 34 bits) #This is the number of IQ groups that must be ingested to
    # grab the desired number of samples
    ADDR_CAPTURESIZE = 0x40  # 1 word (only 27 bits)  #This is the number of IQ groups written to PLDDR4
    ADDR_OUT = 0x48  # 2 words

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, n, groups=tuple(range(N_IQ_GROUPS)), start=True, addr=None, device_addr=None):
        """
        Tell the block to capture n samples of the specified resonator IQ groups. If addr is None an addr will be
        assigned. addr is the number of 4KB pages into the PLDDR4 to start at.

        The amount of data to be captured is n*len(set(groups))*32 bytes, but n will be constrained to
        <= floor(MAX_CAP_RAM_BYTES/32/n_groups). Note that anything at the destination memory will be overwritten.

        groups - An iterable of resonator channel IQ group numbers or 'all'. IQs processed in 256 groups of 8,
        so to capture the IQ values of resonator channels 0, 37, and 2047 groups must be set to either 'all' or should
        include (0, 4, 255). Note that this will cause IQs for channels 0-7, 32-39, and 2040-2047 to be captured.
        """
        # Determine keep, keep a group if all or if the group number is in groups
        keep = np.zeros(8, dtype=np.uint32)
        if isinstance(groups, str):
            n_groups = 256
            if groups.lower() == 'all':
                keep += 0xFFFFFFFF
            else:
                raise ValueError("The only legal string option for groups is 'all'")
        else:
            n_groups = 0
            for g in groups:
                if not 0 <= g < 256:
                    raise ValueError(f'Groups {g} not in range 0-255')
                n_groups += 1
                i = g // 32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the right bit

        # Compute capturesize, this is the number of IQ groups to be written to DDR4
        # TODO we should probably drop N so that we aren't capturing a partial set
        cap_size = int(min(n * n_groups, MAX_CAP_RAM_BYTES // 32))
        # Compute the total capture size, this is the number of IQ groups that must be ingested to capture the data
        total_capturesize = (256 - n_groups + 1) * cap_size

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = cap_size * 32 / 1024 ** 2
        datarate_mbps = 32 * 512 * n_groups / 256

        msg = (f"Will capture ~{datavolume_mb} MB "
               f"of data @ {datarate_mbps} MBps. ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)
        captime = datavolume_mb / datarate_mbps
        print(msg)

        # Set the output address. TODO Support addr
        if addr is None:
            addr = device_addr or PL_DDR4_ADDR
        else:
            addr = PL_DDR4_ADDR + addr * 2 ** 12

        # Set registers
        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP + 0x4 * i, int(k))
        self.write(self.ADDR_CAPTURESIZE, cap_size)
        self.write(self.ADDR_TOTALCAPTURESIZE, total_capturesize & 0xffffffff)
        self.write(self.ADDR_TOTALCAPTURESIZE + 0x4, total_capturesize >> 32)
        self.write(self.ADDR_OUT, int(addr) & 0xffffffff)
        self.write(self.ADDR_OUT + 0x4, int(addr) >> 32)

        if start:
            self.start_capture()

        return captime

    def start(self):
        self.register_map.CTRL.AP_START = 1


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

            if max(groups) > 255:
                raise ValueError(f'Groups must be  not in range 0-255')
            for g in groups:
                if not 0 <= g < 256:
                    raise ValueError(f'Groups {g} not in range 0-255')
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

            if max(groups) > 127:
                raise ValueError(f'Groups must be  not in range 0-127')

            for g in groups:
                if not 0 <= g < 128:
                    raise ValueError(f'Groups {g} not in range 0-127')
                i = g // 32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the correct bit

        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP + 0x4 * i, int(k))
        self.write(self.ADDR_LASTGRP, last)


class WriteAXI256(DefaultIP):
    bindto = ['mazinlab:mkidgen3:write_axi256:0.1']
    ADDR_CAPTURESIZE = 0x10  # 1 27bit word
    ADDR_OUT = 0x18  # 2 words

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, size, addr):
        """
        Capture a number of 256bit words to address, size will one less if odd.

        Address may need to be a multiple of 4096 bytes.

        No bounds checking is performed.
        """
        addr = int(addr)
        size = int(size)

        if size % 2:
            size -= 1

        if size < 2:
            getLogger(__name__).debug("Requested capture of zero samples, ignoring request")
            return

        size = size & (2 ** 28 - 1)
        datavolume_mb = size * 32 / 1024 ** 2

        msg = f"Capturing ~{datavolume_mb} MB of data."
        getLogger(__name__).debug(msg)

        self.write(self.ADDR_CAPTURESIZE, size)
        self.write(self.ADDR_OUT, addr & 0xffffffff)
        self.write(self.ADDR_OUT + 0x4, addr >> 32)

        self.start()

    def start(self):
        self.register_map.CTRL.AP_START = 1


class ADCCapture(DefaultIP):
    """
    // control
    // 0x00 : Control signals
    //        bit 0  - ap_start (Read/Write/COH)
    //        bit 1  - ap_done (Read/COR)
    //        bit 2  - ap_idle (Read)
    //        bit 3  - ap_ready (Read)
    //        bit 7  - auto_restart (Read/Write)
    //        others - reserved
    // 0x04 : Global Interrupt Enable Register
    //        bit 0  - Global Interrupt Enable (Read/Write)
    //        others - reserved
    // 0x08 : IP Interrupt Enable Register (Read/Write)
    //        bit 0  - enable ap_done interrupt (Read/Write)
    //        bit 1  - enable ap_ready interrupt (Read/Write)
    //        others - reserved
    // 0x0c : IP Interrupt Status Register (Read/TOW)
    //        bit 0  - ap_done (COR/TOW)
    //        bit 1  - ap_ready (COR/TOW)
    //        others - reserved
    // 0x10 : Data signal of capturesize
    //        bit 31~0 - capturesize[31:0] (Read/Write)
    // 0x14 : reserved
    // 0x18 : Data signal of iqout
    //        bit 31~0 - iqout[31:0] (Read/Write)
    // 0x1c : Data signal of iqout
    //        bit 31~0 - iqout[63:32] (Read/Write)
    // 0x20 : reserved
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
    """
    ADDR_CAPTURESIZE = 0x10
    ADDR_OUT = 0x18
    bindto = ['mazinlab:mkidgen3:adc_capture:0.7', 'mazinlab:mkidgen3:adc_capture:0.6']

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, n=65536, t_ms=None, start=False, addr=None):
        """
        Capture raw ADC Samples. IQ samples are captured in chunks of 8 sequential samples.

        n = number of IQ samples // 8 to capture from the ADC stream.
        default n is 65536 which corresponds to 1 full DAC replay table 2**19 samples / 8 samples per data transfer.
        max_n = 2**32//32 = 134217728.
        max_n corresponds to 2*32 = 4 GiB of data (PL DRAM size)
        divided by 32 bytes (8  IQ samples) per data transfer from the ADC to PL DRAM.

        Optional:
        t_ms  = time in miliseconds to capture the ADC output for.
        max_t = 262.1 ms  = max_n * 8 [samples/transfer] / 4.096e9 [IQ  samples/sec]
        t_ms is rounded down to the nearest number of IQ samples // 8.
        t_ms takes precedence over n.
        """

        if t_ms is not None:
            # n transactions = t[sec] * 4.096e9[samples/sec]*1[transaction]/8[samples]
            n = int(np.floor(t_ms * 10e-3 * 4.096e9 / 8))

        # Set capturesize, this sets the number of chunks of 8 sequential IQ samples that are captured.
        max_n = MAX_CAP_RAM_BYTES // 32
        cap_size = int(min(n, max_n))
        self.register_map.capturesize = cap_size

        # Set the output address.
        addr = PL_DDR4_ADDR if addr is None else PL_DDR4_ADDR + int(addr) * 2 ** 12
        self.register_map.iqout_1 = addr & 0xffffffff  # set low address
        self.register_map.iqout_2 = addr >> 32  # set high address

        if start:
            self.start()

    def start(self):
        self.register_map.CTRL.AP_START = 1


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

    def _capture(self, source, n, buffer):
        if self.switch is not None:
            self.switch.set_driver(slave=self.SOURCE_MAP[source], commit=True)
        if not self.axis2mm.ready:
            raise IOError("capture core not ready, this shouldn't happen."
                          " Try calling .axis2mm.abort() followed by .axis2mm.clear_error()"
                          " then try a small throwaway capture (data order may not be aligned in the first capture "
                          "after a reset).")
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
        x = 0xc0000000  # start and clear error
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
