import pynq
from pynq import DefaultHierarchy
from logging import getLogger

from mkidgen3.drivers.rfdcclock import configure
from mkidgen3.mkidpynq import get_board_name


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


# This does not appear to actually be necessary for MTS but the example does do it.
def reset_clockwizard(overlay:pynq.Overlay):
    CLOCKWIZARD_LOCK_ADDRESS = 0x0004
    CLOCKWIZARD_RESET_ADDRESS = 0x0000
    CLOCKWIZARD_RESET_TOKEN = 0x000A
    mmcm = overlay.Clocktree.RF_CLKGEN
    mmcm.mmio.write_reg(CLOCKWIZARD_RESET_ADDRESS, CLOCKWIZARD_RESET_TOKEN)


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
        configure(external_10mhz)

    def reset(self):
        self.rfdc.write(0x0004, 0x00000001)

    def select_adc(self, adc='single_ended'):
        if self.switch is None:
            raise RuntimeError('RFDCHierarchy does not support switching ADCs')
        self.switch.set_driver(slave=int(adc == 'single_ended'))

    @property
    def active_adc(self):
        if self.switch is None:
            return 'Switching not supported. Please add driver introspection to determine active adc'
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
            for t in ([f'ADC{i}' for i in range(4)] + [f'DAC{i}' for i in range(2)]):
                print(f'{t}:')
                for k, v in ret[t].items():
                    print(f'  {k}: {v}')
        return ret

    @staticmethod
    def checkhierarchy(description):
        if 'usp_rf_data_converter_0' not in description['ip']:
            return False
        return True

    def enable_mts(self, dac=True, adc=False):
        """Synchronizes all active ADC and DAC tiles

        Parameters
        ----------
        dac : Boolean
            Enables MTS on the DAC
        adc : Boolean
            Enables MTS on the ADC
        """
        if not (dac or adc):
            return

        if get_board_name() == "RFSoC4x2":
            self.ACTIVE_DAC_TILES = 0b0101
            self.ACTIVE_ADC_TILES = 0b0101
            self.MAX_DAC_TILES = 4
            self.MAX_ADC_TILES = 4
            self.DAC_REF_TILE = 2
            self.ADC_REF_TILE = 2
        else:
            raise NotImplementedError(f"{get_board_name()} MTS Not Supported")

        if dac:
            self.rfdc.mts_dac_config.RefTile = self.DAC_REF_TILE
        if adc:
            self.rfdc.mts_adc_config.RefTile = self.ADC_REF_TILE

        self.init_tile_sync(dac=dac, adc=adc)
        self.sync_tiles(dac=dac, adc=adc)

    def init_tile_sync(self, reset_clockwiz=False, dac=True, adc=True):
        """Initilizes the ADCs and DACs for MTS

        This resets all the DACs and ADCs, initilizes the MTS engine in the tiles with the CLK
        inputs and turns the rest of the tiles back on

        Parameters
        ----------
        reset_clockwiz : Boolean
            Resets the clockwizard driving the design, this does not appear to be required
        dac : Boolean
            Enables MTS on the DAC
        adc : Boolean
            Enables MTS on the ADC
        """
        import time

        # It's gross but I'm not going to mess with the structure here because there is some
        # weirdness around the dac config some of which seems to come from the ADC. Unfortunately
        # none of the MTS regs are documented and I haven't had the time to debug said weirdness
        if dac:
            self.rfdc.mts_dac_config.Tiles = 0b0001  # turn only one tile on first
        if adc:
            self.rfdc.mts_adc_config.Tiles = 0b0001
        # Start the SYSREF capture engine in the first tiles
        if dac:
            self.rfdc.mts_dac_config.SysRef_Enable = 1
            self.rfdc.mts_dac_config.Target_Latency = -1
        if adc:
            self.rfdc.mts_adc_config.SysRef_Enable = 1
            self.rfdc.mts_adc_config.Target_Latency = -1
        #
        if dac:
            self.rfdc.mts_dac()
        if adc:
            self.rfdc.mts_adc()
        # Reset MTS ClockWizard MMCM - refer to PG065
        if reset_clockwiz:
            reset_clockwizard(self._overlay)
        time.sleep(0.1)
        # Reset only user selected DAC tiles
        if dac:
            bitvector = self.ACTIVE_DAC_TILES
            for n in range(self.MAX_DAC_TILES):
                if (bitvector & 0x1):
                    self.rfdc.dac_tiles[n].Reset()
                bitvector = bitvector >> 1
        # Reset ADC FIFO of only user selected tiles - restarts MTS engine
        if adc:
            for toggleValue in range(0, 1):
                bitvector = self.ACTIVE_ADC_TILES
                for n in range(self.MAX_ADC_TILES):
                    if (bitvector & 0x1):
                        self.rfdc.adc_tiles[n].SetupFIFOBoth(toggleValue)
                    bitvector = bitvector >> 1

    def sync_tiles(self, dac_target=-1, adc_target=-1, dac=True, adc=True):
        """Synchronize all the active ADC and DAC tiles in the design

        Parameters
        ----------
        dac_target : int
            Set a target latency for the DAC tiles between 0 and 127 cycles passing -1 allows the
            MTS engine to select a latency
        adc_target : int
            Set a target latency for the ADC tiles between 0 and 127 cycles passing -1 allows the
            MTS engine to select a latency
        dac : Boolean
            Enables MTS sync on the DAC
        adc : Boolean
            Enables MTS sync on the ADC
        """
        self._sync_tiles(dac_target, adc_target, dac, adc)
        import mkidgen3.quirks
        if mkidgen3.quirks.MTS.double_sync:
            self._sync_tiles(dac_target, adc_target, dac, adc)

    def _sync_tiles(self, dac_target=-1, adc_target=-1, dac=True, adc=True):
        """ Configures RFSoC MTS alignment"""
        # Set which RF tiles use MTS and turn MTS off
        if self.ACTIVE_ADC_TILES > 0 and adc:
            self.rfdc.mts_adc_config.Tiles = self.ACTIVE_ADC_TILES
            self.rfdc.mts_adc_config.SysRef_Enable = 1
            self.rfdc.mts_adc_config.Target_Latency = adc_target
            self.rfdc.mts_adc()
        elif self.ACTIVE_ADC_TILES == 0:
            self.rfdc.mts_adc_config.Tiles = 0x0
            self.rfdc.mts_adc_config.SysRef_Enable = 0
        if self.ACTIVE_DAC_TILES > 0 and dac:
            self.rfdc.mts_dac_config.Tiles = self.ACTIVE_DAC_TILES  # group defined in binary 0b1111
            self.rfdc.mts_dac_config.SysRef_Enable = 1
            self.rfdc.mts_dac_config.Target_Latency = dac_target
            self.rfdc.mts_dac()
        elif self.ACTIVE_DAC_TILES == 0:
            self.rfdc.mts_dac_config.Tiles = 0x0
            self.rfdc.mts_dac_config.SysRef_Enable = 0

    def set_gain(self, adc_gains=(1.0, 1.0), dac_gains=(1.0,1.0)):
        getLogger(__name__).info('Pretending to set gain, not currently supported')
        return

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
        getLogger(__name__).info('Pretending to set QMC, not currently supported')
        return

        settings = {'EnableGain': 1 if gain else 0, 'EnablePhase': 1 if phase else 0, 'EventSource': 0,
                    'GainCorrectionFactor': gain, 'OffsetCorrectionFactor': offset, 'PhaseCorrectionFactor': phase}

        import xrfdc

        if adc is not None:
            self.rfdc.adc_tiles[adc[0]].blocks[adc[1]].QMCSettings = settings
            self.rfdc.adc_tiles[adc[0]].blocks[adc[1]].UpdateEvent(xrfdc.EVENT_QMC)
            print(f"Setting ADC Tile {adc[0]}, Block {adc[1]}")

        if dac is not None:
            self.rfdc.dac_tiles[dac[0]].blocks[dac[1]].QMCSettings = settings
            self.rfdc.dac_tiles[dac[0]].blocks[dac[1]].UpdateEvent(xrfdc.EVENT_QMC)
            print(f"Setting DAC Tile {dac[0]}, Block {dac[1]}")
        return (settings)
