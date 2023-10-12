import mkidgen3.drivers.rfdcclock
import mkidgen3 as g3
import mkidgen3.drivers.rfdc
from pynq import Overlay
from logging import getLogger
from mkidgen3.mkidpynq import DummyOverlay
from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.server.feedline_config import FeedlineConfig, FeedlineConfigManager

DEFAULT_BIT_FILE='/home/xilinx/gen3_top.bit'


class FeedlineHardware:
    def __init__(self, bitstream=DEFAULT_BIT_FILE, clock_source='4.096GSPS_MTS_dualloop', if_port='dev/ifboard', ignore_version=False,
                 program_clock=True, mts=True, download=False):

        self.config_manager = FeedlineConfigManager()
        self._clock_source = clock_source
        try:
            self._ol = g3.configure(bitstream, ignore_version=ignore_version, clocks=program_clock,
                                    programming_key=clock_source, mts=mts, download=download)
        except RuntimeError as e:
            if 'No Devices Found' in str(e):
                getLogger(__name__).warning('No PL device found, is BOARD set? This is expected on a laptop')
                self._ol = DummyOverlay(bitstream)
            else:
                raise

        self._if_board = IFBoard(if_port, connect=False)
        self._ignore_version = ignore_version
#        if program_clock:
#            import mkidgen3.drivers.rfdc
#            mkidgen3.clocking.configure(clock_source)

    def reset(self):
        raise NotImplementedError('Reset settings not implemented')
        self._if_board.power_off(save_settings=False)
        mkidgen3.drivers.rfdcclock.configure(self._clock_source)
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

        # TODO: we need some sort of conditional apply for BitstreamConfig, RFDCClockingConfig, and RFDCConfig
        # RFDC Clocking
        if fl_setup.rfdc_clk is not None:
            getLogger(__name__).debug(f'Requesting update to RFDC clocking configuration.')
            #.configure?

        # Bitstream
        if fl_setup.bitstream is not None:
            getLogger(__name__).debug(f'Requesting update to bitstream.')
            #.configure?

        # RFDC
        if fl_setup.rfdc is not None:
            getLogger(__name__).debug(f'Requesting update to RFDC configuration.')
            #.configure?

        # IF Board
        if fl_setup.if_board is not None:
            getLogger(__name__).debug(f'Requesting update to IF Board configuration.')
            self._if_board.configure(**fl_setup.if_board.settings_dict())

        # DAC Replay
        if fl_setup.waveform is not None:
            getLogger(__name__).debug(f'Configure DAC with {fl_setup.waveform.settings_dict()}')
            self._ol.dac_replay.configure(**fl_setup.waveform.settings_dict())

        if fl_setup.chan is not None:
            self._ol.photon_pipe.reschan.bin_to_res.configure(**fl_setup.chan.settings_dict())
            # DDC
        if fl_setup.ddc is not None:
            self._ol.photon_pipe.reschan.ddc.configure(**fl_setup.ddc.settings_dict())
            # Matched Filters
        if fl_setup.filter is not None:
            self._ol.photon_pipe.phasematch.configure(**fl_setup.filter.settings_dict())
            # Matched Filters
        if fl_setup.trig is not None:
            self._ol.photon_pipe.phasematch.configure(**fl_setup.trig.settings_dict())
