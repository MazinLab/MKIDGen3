import mkidgen3.clocking
import mkidgen3.drivers.rfdc
from pynq import Overlay
from logging import getLogger
from mkidgen3.mkidpynq import DummyOverlay
from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.server.feedline_objects import FeedlineConfig, FeedlineConfigManager

DEFAULT_BIT_FILE='/home/xilinx/gen3_top.bit'


class FeedlineHardware:
    def __init__(self, bitstream, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=True, download=False, program_clock=True):

        self.config_manager = FeedlineConfigManager()
        self._clock_source = clock_source
        try:
            self._ol = Overlay(bitstream, download=download, ignore_version=ignore_version)
        except RuntimeError as e:
            if 'No Devices Found' in str(e):
                getLogger(__name__).warning('No PL device found, is BOARD set? This is expected on a laptop')
                self._ol = DummyOverlay(bitstream)
            else:
                raise

        self._if_board = IFBoard(if_port, connect=False)
        self._ignore_version = ignore_version
        if program_clock:
            import mkidgen3.drivers.rfdc
            mkidgen3.clocking.start_clocks(clock_source)

    def reset(self):
        self._if_board.power_off(save_settings=False)
        mkidgen3.clocking.start_clocks(self._clock_source)
        self._ol = Overlay(self._ol.bitfile_name, ignore_version=self._ignore_version, download=True)
        self._if_board.power_on()

    def status(self):
        """

        Returns: a JSON Serializable status object per the schema fully describing the state of the feedline hardware

        """
        return 'hardware status'

    def bequiet(self, stop_dacs=True, poweroff_if=False):
        """

        Args:
            stop_dacs: Stop the DACs from replaying any values
            poweroff_if: Power down the IF board (implies `stop_if`)

        Returns: None
        """
        if stop_dacs:
            self._ol.dac_table.quiet()
        if poweroff_if:
            self._if_board.power_off(save_settings=False)

    def config_compatible_with(self, config: FeedlineConfig):
        return self.config_manager.required() < config

    def derequire_config(self, id):
        """True iff the required settings changed as a result"""
        try:
            return self.config_manager.pop(id)
        except KeyError:
            return False

    def apply_config(self, id, config: FeedlineConfig):
        """Takes and applies a config to the hardware, updates and tracks the effective set of settings"""

        # Add the config to the pot and get the effective config
        fl_setup = self.config_manager.add(id, config)

        # IF Board
        if fl_setup.if_config is not None:
            getLogger(__name__).debug(f'Configure IF Board with {fl_setup.if_config.settings_dict()}')
            self._if_board.configure(**fl_setup.dac_config.settings_dict())

        # DAC
        if fl_setup.dac_config is not None:
            getLogger(__name__).debug(f'Configure DAC with {fl_setup.dac_config.settings_dict()}')
            self._ol.dac_replay.configure(**fl_setup.dac_config.settings_dict())

        # ADC
        if fl_setup.adc_config is not None:
            getLogger(__name__).debug(f'Configure ADC with {fl_setup.adc_config.settings_dict()}')
            # self._ol.dac_replay.configure(**fl_setup.dac_setup.settings_dict())

        # Photon Pipe
        if fl_setup.pp_config is not None:
            # Channel assignments
            if fl_setup.pp_config.chan_config is not None:
                self._ol.photon_pipe.reschan.bin_to_res.configure(**fl_setup.pp_config.chan_config.settings_dict())
            # DDC
            if fl_setup.pp_config.ddc_config is not None:
                self._ol.photon_pipe.reschan.ddc.configure(**fl_setup.pp_config.ddc_config.settings_dict())
            # Matched Filters
            if fl_setup.pp_config.filter_config is not None:
                self._ol.photon_pipe.phasematch.configure(**fl_setup.pp_config.filter_config.settings_dict())
            # Matched Filters
            if fl_setup.pp_config.trig_config is not None:
                self._ol.photon_pipe.phasematch.configure(**fl_setup.pp_config.trig_config.settings_dict())
