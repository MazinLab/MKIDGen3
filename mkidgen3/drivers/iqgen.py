from pynq import allocate, DefaultIP


class IQGen(DefaultIP):
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
    // 0x10 : Data signal of max
    //        bit 31~0 - max[31:0] (Read/Write)
    // 0x14 : reserved
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
    """
    ADDR_CAPTURESIZE = 0x10
    bindto = ['mazinlab:mkidgen3:iq_gen:0.1']

    def __init__(self, description):
        super().__init__(description=description)

    def generate(self, n, start=False):
        """
        Tell the block to generate n sets of 8 IQ samples.
        """
        self.write(self.ADDR_CAPTURESIZE, n)
        if start:
            self.start()

    def start(self):
        self.register_map.CTRL.AP_START=1
