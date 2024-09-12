from pynq import DefaultIP


class AxisSwitch(DefaultIP):
    bindto = ['xilinx.com:ip:axis_switch:1.1']

    def __init__(self, description):
        super().__init__(description=description)

    def set_driver(self, slave=0, master=0, disable=False, commit=True):
        """Set the slave for the master, committing the change triggers a 16 cycle soft reset of the switch"""
        cfg = slave & 0b111
        master = min(max(int(master), 0), 15)
        if disable:
            cfg |= 0x80000000
        self.write(0x0040 + master * 4, cfg)
        if commit:
            self.commit()

    def disable(self, master=0):
        """Disable the master, do not change the driver, will commit and soft-reset the core"""
        master = min(max(int(master), 0), 15)
        addr = 0x0040 + master * 4
        self.write(addr, self.read(addr) | 0x80000000)
        self.commit()

    def enable(self, master=0):
        """Enable the master, do not change the driver, will commit and soft-reset the core"""
        master = min(max(int(master), 0), 15)
        addr = 0x0040 + master * 4
        self.write(addr, self.read(addr) ^ 0x80000000)
        self.commit()

    def is_disabled(self, master=0):
        """True iff the master is disabled"""
        return (self.read(0x0040 + min(max(int(master), 0), 15) * 4) & 0xf0000000) != 0

    def driver_of(self, master=0):
        """Return the driver of the specified master, master may be disabled"""
        return self.read(0x0040 + min(max(int(master), 0), 15) * 4)

    def commit(self):
        """Commit config, triggers a soft 16 cycle reset"""
        self.write(0x0000, 0x2)


class SwitchOnLast(DefaultIP):
    bindto = ['mazinlab:mkidgen3:switch_on_last:0.1']

    def __init__(self, description):
        super().__init__(description=description)

    def set_driver(self, slave=0, disable=False):
        """Set the slave for the output"""
        slave = max(min(slave, 5), 0)
        if disable:
            self.register_map.enable = False
        self.register_map.stream = slave

    def disable(self):
        """Disable the output"""
        self.register_map.enable = False

    def enable(self):
        """Enable the output"""
        self.register_map.enable = True

    def is_disabled(self):
        """True iff the output is disabled"""
        return self.register_map.enable.enable

    def driver_of(self):
        """Return the driver for the output, output may be disabled"""
        return self.register_map.stream.stream
