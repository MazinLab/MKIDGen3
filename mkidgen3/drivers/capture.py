import numpy as np
from pynq import allocate, DefaultIP, DefaultHierarchy

import mkidgen3.mkidpynq
from mkidgen3.mkidpynq import N_IQ_GROUPS, MAX_CAP_RAM_BYTES, PL_DDR4_ADDR  # config, overlay details
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
            ret += [j + 32*i for j in range(32) if k & (1 << j)]
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
                groups = set([g//8 for g in groups])  # convert channel to group
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


class WriteAXI256(DefaultIP):
    bindto = ['mazinlab:mkidgen3:write_axi256:0.1']
    ADDR_CAPTURESIZE = 0x10  # 1 27bit word
    ADDR_OUT = 0x18 # 2 words

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

        size = size & (2**28-1)
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
            n = int(np.floor(
                t_ms * 10e-3 * 4.096e9 / 8))  # n transactions = t[sec] * 4.096e9[samples/sec]*1[transaction]/8[samples]

        # Set capturesize, this sets the number of chunks of 8 sequential IQ samples that are captured.
        max_n = MAX_CAP_RAM_BYTES // 32
        cap_size = int(min(n, max_n))
        self.register_map.capturesize = cap_size

        # Set the output address.
        addr = PL_DDR4_ADDR if addr is None else PL_DDR4_ADDR + int(addr) * 2 ** 12
        self.register_map.iqout_1 = addr & (0xffffffff)  # set low  address
        self.register_map.iqout_2 = addr >> 32  # set high address

        if start:
            self.start()

    def start(self):
        self.register_map.CTRL.AP_START = 1


class CaptureHierarchy(DefaultHierarchy):
    def __init__(self, description):
        super().__init__(description)
        # self.adc_cap = self.adc_capture_0
        # self.raw_iq_cap = self.iq_capture_0
        # self.dds_iq_cap = self.iq_capture_1
        self.filter_iq = self.filter_iq_0
        self.axi256 = self.write_axi256_0

    @staticmethod
    def checkhierarchy(description):
        for k in ('write_axi256_0', 'filter_iq_0'):
            if k not in description['ip']:
                return False
        return True

    def captureiq(self, n, buffer, groups='all'):
        """ Capture n samples of each specified iq into buffer, returning the time it will take"""
        if not isinstance(buffer, int):
            space = buffer.size * buffer.dtype.itemsize
            addr = buffer.device_address
        else:
            addr = buffer
            space = 2**32-1 - (buffer-mkidgen3.mkidpynq.PL_DDR4_ADDR)

        if addr%4096 != 0:
            getLogger(__name__).warning('Address is not 4K aligned, may cause issues.')

        self.filter_iq.keep = groups
        keep = self.filter_iq.keep
        n_groups = keep

        # Compute capturesize, this is the number of IQ groups to be written to DDR4
        capturesize = n * n_groups

        if capturesize*32 > space:
            capturesize = space // 32
            getLogger(__name__).warning(f'Insufficient space for requested, truncating to {capturesize/n_groups} '
                                        f'samples.')

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = capturesize * 32 / 1024 ** 2
        datarate_mbps = 32 * 512 * n_groups / 256
        captime = datavolume_mb / datarate_mbps

        msg = (f"Will capture ~{datavolume_mb} MB "
               f"of data @ {datarate_mbps} MBps. ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        #Set the switch if needed
        # self.axi256_switch.select(1)

        self.axi256.capture(capturesize, addr)
        return captime

    def captureraw(self, n, buffer):
        if not isinstance(buffer, int):
            space = buffer.size  #TODO convert to number of bytes
            addr = buffer.device_address
        else:
            addr = buffer
            space = 2**32-1 - (buffer-mkidgen3.mkidpynq.PL_DDR4_ADDR)

        if addr%4096 != 0:
            getLogger(__name__).warning('Address is not 4K aligned, may cause issues.')

        capturesize = n

        if capturesize*32 > space:
            capturesize = space // 32
            getLogger(__name__).warning(f'Insufficient space for requested, truncating to {capturesize} '
                                        f'samples.')

        # NB this ignores the final bit from a non-128 multiple
        datavolume_mb = capturesize * 32 / 1024 ** 2
        datarate_mbps = 32 * 512
        captime = datavolume_mb / datarate_mbps

        msg = (f"Will capture ~{datavolume_mb} MB "
               f"of data @ {datarate_mbps} MBps. ETA {datavolume_mb / datarate_mbps * 1000:.0f} ms")
        getLogger(__name__).debug(msg)

        #Set the switch if needed
        # self.axi256_switch.select(0)

        self.axi256.capture(capturesize, addr)
        return captime
