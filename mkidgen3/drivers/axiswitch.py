from pynq import allocate, DefaultIP


class AxisSwitch(DefaultIP):
    bindto = ['xilinx.com:ip:axis_switch:1.1']

    def __init__(self, description):
        super().__init__(description=description)

    def set_driver(self, slave=0, master=0, disable=False, commit=True):
        """Set the slave for the master"""
        cfg = slave & 0b111
        master = min(max(int(master), 0), 15)
        if disable:
            cfg |= 0x80000000
        self.write(0x0040 + master * 4, cfg)
        if commit:
            self.commit()

    def is_disabled(self, master=0):
        return (self.read(0x0040 + min(max(int(master), 0), 15) * 4) & 0xf0000000) > 0

    def driver_for(self, master=0):
        return self.read(0x0040 + min(max(int(master), 0), 15) * 4)

    def commit(self):
        """Commit config, triggers a soft 16 cycle reset"""
        self.write(0x0000, 0x2)
