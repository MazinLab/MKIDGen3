import copy

import mkidgen3.drivers.rfdcclock
import mkidgen3 as g3
import mkidgen3.drivers.rfdc
from pynq import Overlay
from logging import getLogger
from mkidgen3.mkidpynq import DummyOverlay
from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.server.feedline_config import (FeedlineConfig, FeedlineConfigManager,
                                             BitstreamConfig, RFDCClockingConfig, RFDCConfig)
from mkidgen3.server.feedline_client_objects import CaptureAbortedException, CaptureRequest
import zmq
from mkidgen3.server.misc import zpipe
import threading
import pynq
import time

DEFAULT_BIT_FILE='/home/xilinx/gen3_top.bit'


class FeedlineHardware:
    def __init__(self, bitstream=DEFAULT_BIT_FILE, rfdcclock='4.096GSPS_MTS_dualloop', if_port='dev/ifboard', ignore_version=False,
                 program_clock=True, rfdc: RFDCConfig| None =None, download=False, clock_source='external'):
        """

        Args:
            bitstream:
            rfdcclock:
            if_port:
            ignore_version:
            program_clock:
            rfdc: Enable MTS. Cannot be enabled unless until overlay is downloaded.
            download:
            clock_source:
        """

        self._default_bitstream = BitstreamConfig(bitstream=bitstream, ignore_version=ignore_version)
        self._default_rfdc_clocking = RFDCClockingConfig(programming_key=rfdcclock, clock_source=clock_source)
        self._default_rfdc = RFDCConfig(dac_mts=True, adc_mts=True) if rfdc is None else rfdc

        self.config_manager = FeedlineConfigManager()

        if program_clock:
            mkidgen3.drivers.rfdcclock.configure(**self._default_rfdc_clocking.settings_dict())
            time.sleep(0.5)  # allow clocks to stabilize before loading overlay  TODO: is this necessary

        try:
            self._ol = pynq.Overlay(self._default_bitstream.bitstream,
                                    ignore_version=self._default_bitstream.ignore_version, download=download)
            mkidgen3.quirks.Overlay(self._ol).post_configure()

            if download:
                self._ol.rfdc.enable_mts(*self._default_rfdc.mts)

        except RuntimeError as e:
            if 'No Devices Found' in str(e):
                getLogger(__name__).warning('No PL device found, is BOARD set? This is expected on a laptop')
                self._ol = DummyOverlay(bitstream)
            else:
                raise

        self._if_board = IFBoard(if_port, connect=False)

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

        if fl_setup.rfdc_clk is not None:
            getLogger(__name__).debug(f'Requesting update to RFDC clocking configuration.')
            mkidgen3.drivers.rfdcclock.configure(**fl_setup.rfdc_clk.settings_dict())

        if fl_setup.bitstream is not None or fl_setup.rfdc_clk is not None:
            getLogger(__name__).debug(f'Requesting update to bitstream.')
            if fl_setup.bitstream is not None:
                x = copy.deepcopy(self._default_bitstream)
                x.merge_with(fl_setup.bitstream)

            self._ol = Overlay(x.bitstream, ignore_version=x.ignore_version, download=True)

        # RFDC
        if fl_setup.rfdc is not None:
            getLogger(__name__).debug(f'Requesting update to RFDC configuration.')
            self._ol.rfdc.enable_mts(dac=fl_setup.rfdc.dac_mts, adc=fl_setup.rfdc.adc_mts)
            self._ol.rfdc.set_qmc(adc_gains=fl_setup.rfdc.adc_gains, dac_gains=fl_setup.rfdc.dac_gains)

        # IF Board
        if fl_setup.if_board is not None:
            getLogger(__name__).debug(f'Requesting update to IF Board configuration.')
            try:
                self._if_board.configure(**fl_setup.if_board.settings_dict())
            except Exception as e:
                getLogger(__name__).warning(f"Unable to connect to IF board: {e}, proceeding without IF board.")

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

    def plram_cap(self, pipe, cr: CaptureRequest, context=None):
        """

        Args:
            pipe: An inproc
            context:
            cr: A CaptureRequest object

        Returns: None

        """
        failmsg = ''
        try:
            assert cr.type == 'engineering', 'Incorrect capture request type'
            assert self._ol.capture.ready(), 'Capture Subsystem is busy'
        except AssertionError as e:
            failmsg = str(e)

        try:
            cr.establish(context=context)
        except zmq.ZMQError as e:
            failmsg = f"Unable to establish capture {cr.id} due to {e}, dropping request."

        if failmsg:
            getLogger(__name__).error(failmsg)
            cr.fail(failmsg, raise_exception=False)
            return
        CHUNKING_THRESHOLD = 256*1024**3
        nchunks = cr.size_bytes // CHUNKING_THRESHOLD
        partial = cr.size_bytes - CHUNKING_THRESHOLD * nchunks
        chunks = [CHUNKING_THRESHOLD] * nchunks
        if partial:
            chunks.append(partial)

        try:
            for i, csize in enumerate(chunks):
                try:
                    abort = pipe.recv(zmq.NOBLOCK)
                    raise CaptureAbortedException(abort)
                except zmq.ZMQError as e:
                    if e.errno != zmq.EAGAIN:
                        raise
                data = self._ol.capture.capture(csize, tap=cr.tap, wait=True)
                cr.send_data(data, status=f'{i + 1}/{len(chunks)}', copy=False)
                data.free_buffer()
            cr.finish()
        except CaptureAbortedException as e:
            cr.abort(e)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            cr.fail(f'Aborted due to {e}', raise_exception=False)
        finally:
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
        pipe.close()

    def photon_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context=None):
        """
        #TODO should this be a instance method of FeedlineHardware?
        pipe: a zme pair pipe to detect abort
        cr: the capture request
        ol: the overlay
        """
        failmsg = ''
        photon_maxi = self._ol.photon_pipe.trigger_system.photon_maxi
        try:
            assert cr.type == 'photon', 'Incorrect capture request type'
        except AssertionError as e:
            failmsg = str(e)

        try:
            cr.establish(context=context)
        except zmq.ZMQError as e:
            failmsg = f"Unable to establish capture {cr.id} due to {e}, dropping request."

        if failmsg:
            getLogger(__name__).error(failmsg)
            try:
                cr.fail(failmsg)
                cr.destablish()
            except zmq.ZMQError as ez:
                getLogger(__name__).warning(f'Failed to send abort/destablish for {cr} due to {ez}')
            return

        q, q_other = zpipe(zmq.Context.instance())
        # q = queue.SimpleQueue()  #an alternative
        fountain, stop = photon_maxi.photon_fountain(q_other, spawn=True, copy_buffer=False)

        def photon_sender(q: zmq.Socket, cr, unpack=False):
            log = getLogger(__name__)
            try:
                while True:
                    log.info(f'iter start')
                    photons = q.recv_pyobj()
                    log.info(f'received')
                    if photons is None:
                        cr.finish()
                        break
                    cr.send_data(photon_maxi.unpack_photons(photons) if unpack else photons, copy=False)
            except Exception as e:
                cr.abort(f'Uncaught exception: {e}')
                q.close()
                raise e
            log.info(f'done')

        sender = threading.Thread(target=photon_sender, args=(q, cr))

        try:
            sender.start()
            fountain.start()
            photon_maxi.capture(buffer_time_ms=cr.nsamp)  # todo: add support for setting the latency via the request?
            while not cr.completed:
                try:
                    abort = pipe.recv(zmq.NOBLOCK)
                    raise CaptureAbortedException(abort)
                except zmq.ZMQError as e:
                    if e.errno != zmq.EAGAIN:
                        raise
        except CaptureAbortedException as e:
            stop.set()  # sender will finish up the CR
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            stop.set()
            cr.fail(f'Failed due to {e}')
        finally:
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
            self._ol.photon_pipe_trigger_system.photon_maxi.stop_capture()
            stop.set()
            pipe.close()
            fountain.join()
            sender.join()
            if isinstance(q, zmq.Socket):
                q.close()

    def stamp_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context: zmq.Context=None):
        failmsg = ''
        postage_maxi = self._ol.photon_pipe.trigger_system.postage_maxi
        try:
            assert cr.type == 'postage', 'Incorrect capture request type'
            assert postage_maxi.register_map.AP_CTRL_AP_IDLE == 1, 'Postage MAXI is busy'
        except AssertionError as e:
            failmsg = str(e)

        try:
            cr.establish(context=context)
        except zmq.ZMQError as e:
            failmsg = f"Unable to establish capture {cr.id} due to {e}, dropping request."

        if failmsg:
            getLogger(__name__).error(failmsg)
            cr.fail(failmsg, raise_exception=False)
            return

        try:
            postage_maxi.capture()
            while not postage_maxi.interrupt.is_set():
                try:
                    abort = pipe.recv(zmq.NOBLOCK)
                    raise CaptureAbortedException(abort)
                except zmq.ZMQError as e:
                    if e.errno != zmq.EAGAIN:
                        raise
                time.sleep(min(postage_maxi.MAX_CAPTURE_TIME_S / 10, .1))
            cr.send_data(postage_maxi.get_postage(raw=False, scaled=True), copy=False)
            cr.finish()
        except CaptureAbortedException as e:
            cr.abort(e, raise_exception=False)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            cr.fail(f'Failed due to {e}', raise_exception=False)
        finally:
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
            pipe.close()
