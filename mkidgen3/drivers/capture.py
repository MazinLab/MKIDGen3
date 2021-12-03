import numpy as np
from pynq import DefaultIP
from mkidgen3.mkidpynq import N_IQ_GROUPS, MAX_CAP_RAM_BYTES, PL_DDR4_ADDR #config, overlay details


class IQCapture(DefaultIP):
    """
    // ==============================================================
    // Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.1.1 (64-bit)
    // Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
    // ==============================================================
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
    // 0x10 - 0x2f : Data signal of keep
    //        bit 31~0 - keep[31:0] keep[63:32] keep[95:64] .... (Read/Write)
    // 0x30 : reserved
    // 0x34 : Data signal of capturesize  max value 2^28-1
    //        bit 31~0 - capturesize[31:0] (Read/Write)
    // 0x38 : reserved
    // 0x3c : Data signal of iqout
    //        bit 31~0 - iqout[31:0] (Read/Write)
    // 0x40 : Data signal of iqout
    //        bit 31~0 - iqout[63:32] (Read/Write)
    // 0x44 : reserved
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

    #define XIQ_CAPTURE_CONTROL_ADDR_AP_CTRL          0x00
    #define XIQ_CAPTURE_CONTROL_ADDR_GIE              0x04
    #define XIQ_CAPTURE_CONTROL_ADDR_IER              0x08
    #define XIQ_CAPTURE_CONTROL_ADDR_ISR              0x0c
    #define XIQ_CAPTURE_CONTROL_ADDR_KEEP_DATA        0x10
    #define XIQ_CAPTURE_CONTROL_BITS_KEEP_DATA        256
    #define XIQ_CAPTURE_CONTROL_ADDR_CAPTURESIZE_DATA 0x34
    #define XIQ_CAPTURE_CONTROL_BITS_CAPTURESIZE_DATA 32
    #define XIQ_CAPTURE_CONTROL_ADDR_IQOUT_DATA       0x3c
    #define XIQ_CAPTURE_CONTROL_BITS_IQOUT_DATA       64
    """
    bindto = ['mazinlab:mkidgen3:iq_capture:0.7']
    ADDR_KEEP = 0x10  # 8 words
    ADDR_CAPTURESIZE = 0x34
    ADDR_OUT = 0x3c  # 2 words

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, n, groups=tuple(range(N_IQ_GROUPS)), start=True, addr=None):
        """
        Tell the block to capture n samples of IQ groups from only the specified resonator groups. If addr is None an
        addr will be assigned.

        n will be constrained to <= floor(MAX_CAP_RAM_BYTES/32/n_groups)

        Note that anything in memory from [addr, addr+32*(n*n_groups+1) ) will be overwritten.

        Note (TODO) addr must be in the PL DDR4 space and must be 4kB aligned

        groups - an iterable of IQ group numbers or 'all'. IQs are processed in 256 groups of 8, so to capture the IQ
        values of IQs 0, 37, and 2047 groups must be set to either 'all' or should include (0, 4, 255). Note that this
        will cause IQs 0-7, 32-39, and 2040-2047 to be captured.
        """

        # Set keep
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
                i = g//32  # (0-7)
                keep[i] |= (1 << (g % 32))  # set the right bit

        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP+0x4*i, int(k))

        # Set capturesize, this sets the number of groups of 8 that are captured
        # if it isn't a multiple of
        max_n = int(MAX_CAP_RAM_BYTES/32)
        cap_size = int(min(n*n_groups, max_n))
        self.write(self.ADDR_CAPTURESIZE, cap_size)

        #NB this ignores the final bit from a non-128 multiple
        datavolume_mb = cap_size * 32 / 1024 ** 2
        datarate_mbps = 32 * 512 * n_groups / 256
        captime_ms = cap_size / 2 / 1024 ** 2 * 1000

        print(f"Will capture ~{datavolume_mb} MB "
              f"of data @ {datarate_mbps} MBps. ETA {datavolume_mb/datarate_mbps*1000:.0f} ms")

        # Set the output address. TODO Support addr
        if addr is None:
            addr = PL_DDR4_ADDR
        else:
            addr = PL_DDR4_ADDR+addr*2**12
        self.write(self.ADDR_OUT, int(addr)&0xffffffff)
        self.write(self.ADDR_OUT+0x4, int(addr)>>32)

    #     if start:
    #         self.start_capture()
    #
    # def halt_capture(self):
    #     self.write(self.ADDR_CONFIGURE, 1)
    #
    # def start_capture(self):
    #     self.write(self.ADDR_CONFIGURE, 1)
    #     self.write(self.ADDR_CONFIGURE, 0)


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
            n = int(np.floor(t_ms*10e-3*4.096e9/8)) # n transactions = t[sec] * 4.096e9[samples/sec]*1[transaction]/8[samples]

        # Set capturesize, this sets the number of chunks of 8 sequential IQ samples that are captured.
        max_n = MAX_CAP_RAM_BYTES//32
        cap_size = int(min(n, max_n))
        self.register_map.capturesize = cap_size

        # Set the output address.
        addr = PL_DDR4_ADDR if addr is None else PL_DDR4_ADDR+int(addr)*2**12
        self.register_map.iqout_1 = addr & (0xffffffff)  # set low  address
        self.register_map.iqout_2 = addr >> 32  # set high address

        if start:
            self.start()

    def start(self):
        self.register_map.CTRL.AP_START = 1