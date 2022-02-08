from pynq import DefaultHierarchy
import xrfclk
from logging import getLogger


def status():
    import mkidgen3
    rfdc = mkidgen3._gen3_overlay.usp_rf_data_converter_0

    regmap = {'Restart Power-On State Machine': 0x0004,
              'Restart State': 0x0008,
              'Current State': 0x000C,
              'Reset Count': 0x0038,
              'Interrupt Status': 0x0200,
              'Tile Common Status': 0x0228,
              'Tile Disable': 0x0230}
    tilemap = [(f'ADC{i}', v) for i, v in enumerate((0x14000, 0x18000, 0x1C000, 0x20000))]
    tilemap += [(f'DAC{i}', v) for i, v in enumerate((0x04000, 0x08000))]  # , 0x0C000, 0x10000))]
    tilemap = dict(tilemap)
    print(rfdc.read(0x0008))
    for t, taddr in tilemap.items():
        print(t)
        for k, r in regmap.items():
            print(f'  {k}:  {rfdc.read(taddr + r)}')


def reset():
    import mkidgen3
    rfdc = mkidgen3._gen3_overlay.usp_rf_data_converter_0
    rfdc.write(0x0004, 0x00000001)


class RFDCHierarchy(DefaultHierarchy):
    def __init__(self, description):
        super().__init__(description)
        self.rfdc = self.usp_rf_data_converter_0
        try:
            self.switch = self.axis_switch_0
        except AttributeError:
            self.switch = None
            getLogger(__name__).info('RFDCHierarchy does not support switching ADCs')

    def start_clocks(self):
        xrfclk.set_ref_clks()

    def reset(self):
        self.rfdc.write(0x0004, 0x00000001)

    def select_adc(self, adc='single_ended'):
        if self.switch is None:
            raise RuntimeError('RFDCHierarchy does not support switching ADCs')
        self.switch.set_driver(slave=int(adc is 'single_ended'))

    @property
    def active_adc(self):
        if self.switch is None:
            raise RuntimeError('RFDCHierarchy does not support switching ADCs')
        if self.switch.is_disabled():
            return 'none'
        elif self.switch.driver_for() & 1:
            return 'single_ended (ADC0 0,2)'
        else:
            return 'differential (ADC2 0,2)'

    def rfdc_status(self, tell=False):
        regmap = {'Restart Power-On State Machine': 0x0004,
                  'Restart State': 0x0008,
                  'Current State': 0x000C,
                  'Reset Count': 0x0038,
                  'Interrupt Status': 0x0200,
                  'Tile Common Status': 0x0228,
                  'Tile Disable': 0x0230}
        tilemap = [(f'ADC{i}', v) for i, v in enumerate((0x14000, 0x18000, 0x1C000, 0x20000))]
        tilemap += [(f'DAC{i}', v) for i, v in enumerate((0x04000, 0x08000))]  # , 0x0C000, 0x10000))]
        tilemap = dict(tilemap)

        ret = {'global': self.rfdc.read(0x0008)}
        for t, taddr in tilemap.items():
            ret[t] = {k: self.rfdc.read(taddr + r) for k, r in regmap.items()}
        if tell:
            print('Global Status: ' + hex(ret['global']))
            for t in ([f'ADC{i}' for i in range(4)]+[f'DAC{i}' for i in range(2)]):
                print(f'{t}:')
                for k,v in ret[t].items():
                    print(f'  {k}: {v}')
        return ret

    @staticmethod
    def checkhierarchy(description):
        if 'usp_rf_data_converter_0' not in description['ip']:
            return False
        return True
