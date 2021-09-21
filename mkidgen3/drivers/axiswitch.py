class AxisSwitch(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['xilinx.com:ip:axis_switch:1.1']

    def set_master(self, master, slave=0, disable=False):
        """Set the slave for the master"""
        cfg = 0x80000000 if disable else (slave & 0b111)
        self.write(0x0040 + master * 4, cfg)

    def status(self):
        self.read(0x0040)

    def commit(self):
        """Commit config, triggers a soft 16 cycle reset"""
        self.write(0x0000, 0x2)