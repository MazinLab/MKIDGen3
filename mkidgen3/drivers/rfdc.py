from pynq import DefaultHierarchy
from logging import getLogger

from mkidgen3.clocking import start_clocks


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

    def start_clocks(self, external_10mhz=False):
        start_clocks(external_10mhz=external_10mhz)

    def reset(self):
        self.rfdc.write(0x0004, 0x00000001)

    def select_adc(self, adc='single_ended'):
        if self.switch is None:
            raise RuntimeError('RFDCHierarchy does not support switching ADCs')
        self.switch.set_driver(slave=int(adc == 'single_ended'))

    @property
    def active_adc(self):
        if self.switch is None:
            return 'Switching not supported. Please add driver introspection to determine active adc'  #TODO
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

    def set_gain(self, gain=1.0, qmc_settings=None):
        settings = {'EnableGain': 1, 'EnablePhase': 0, 'EventSource': 0, 'GainCorrectionFactor': gain,
                    'OffsetCorrectionFactor': 0, 'PhaseCorrectionFactor': 0.0}

        if qmc_settings is not None:
            for k,v in settings.items():
                settings[k] = qmc_settings.get(k,v)

        import xrfdc

        self.rfdc.dac_tiles[0].blocks[0].QMCSettings = settings
        self.rfdc.dac_tiles[0].blocks[0].UpdateEvent(xrfdc.EVENT_QMC)
        self.rfdc.dac_tiles[0].blocks[1].QMCSettings = settings
        self.rfdc.dac_tiles[0].blocks[1].UpdateEvent(xrfdc.EVENT_QMC)

        self.rfdc.dac_tiles[1].blocks[2].QMCSettings = settings
        self.rfdc.dac_tiles[1].blocks[2].UpdateEvent(xrfdc.EVENT_QMC)
        self.rfdc.dac_tiles[1].blocks[3].QMCSettings = settings
        self.rfdc.dac_tiles[1].blocks[3].UpdateEvent(xrfdc.EVENT_QMC)

    def set_qmc(self, adc=None, dac=None, gain=0.0, offset=0, phase=0.0):
        """
        Sets the quadrature error modulation circuit for the specified adc/dac.

        adc: tuple describing (adc tile, adc block) allowed values: 0,1,2,3
        dac: tuple describing (dac tile, dac block) allowed values: 0,1,2,3
        gain: number between 0 and 2 describing data converter gain.
        offset: xxxxx
        phase: xxxxxx

        Example Usage: set_qmc(adc=(0,1), gain=1.5)
        """

        settings = {'EnableGain': 1 if gain else 0, 'EnablePhase': 1 if phase else 0, 'EventSource': 0, 'GainCorrectionFactor': gain,'OffsetCorrectionFactor': offset, 'PhaseCorrectionFactor': phase}

        import xrfdc

        if adc is not None:
            self.rfdc.adc_tiles[adc[0]].blocks[adc[1]].QMCSettings = settings
            self.rfdc.adc_tiles[adc[0]].blocks[adc[1]].UpdateEvent(xrfdc.EVENT_QMC)
            print(f"Setting ADC Tile {adc[0]}, Block {adc[1]}")

        if dac is not None:
            self.rfdc.dac_tiles[dac[0]].blocks[dac[1]].QMCSettings = settings
            self.rfdc.dac_tiles[dac[0]].blocks[dac[1]].UpdateEvent(xrfdc.EVENT_QMC)
            print(f"Setting DAC Tile {dac[0]}, Block {dac[1]}")
        return(settings)
