import asyncio
import copy
from mkidgen3.rfsocmemory import determine_max_chunk, memfree_mib
import mkidgen3.drivers.rfdcclock
import mkidgen3.drivers.rfdc
from pynq import Overlay
from logging import getLogger
from mkidgen3.mkidpynq import DummyOverlay
from mkidgen3.system_parameters import ADC_MAX_INT
from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.server.feedline_config import (FeedlineConfig, FeedlineConfigManager,
                                             BitstreamConfig, RFDCClockingConfig, RFDCConfig)
from mkidgen3.server.captures import CaptureRequest
from mkidgen3.util import check_zmq_abort_pipe, AbortedException, format_bytes, compute_max_val, format_time
from ..interrupts import ThreadedPLInterruptManager
import zmq
from mkidgen3.server.misc import zpipe
import threading
import pynq
import time
import numpy as np
from mkidgen3.mkidpynq import unpack_photons
DEFAULT_BIT_FILE = '/home/xilinx/gen3_top_final.bit'


class FeedlineHardware:
    def __init__(self, bitstream=DEFAULT_BIT_FILE, rfdcclock='4.096GSPS_MTS_dualloop', if_port='dev/ifboard',
                 ignore_version=False,
                 program_clock=True, rfdc: RFDCConfig | None = None, download=False, clock_source='external'):
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
                ThreadedPLInterruptManager.get_manager()
                self._ol.rfdc.enable_mts(dac=self._default_rfdc.dac_mts, adc=self._default_rfdc.adc_mts)

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
        return self.config_manager.required().compatible_with(config)

    def derequire_config(self, id):
        """True iff the required settings changed as a result"""
        try:
            return self.config_manager.pop(id)
        except KeyError:
            return False

    def apply_config(self, id, config: FeedlineConfig):
        """

        Args:
            id: an identifier to associate with the config, identifiers should be unique to each config.
            config: a feedline config to apply, only updates will actually be sent to hardware

        Returns: None

        Raises: ValueError may be raised if a non-unique ID is used. ValueError will be raised if the config is not
        compatible with currently required configs or it contains settings that are hashed and
        have not been seen before.

        """

        # Add the config to the pot and get the effective config
        fl_setup = self.config_manager.add(id, config)

        clock_set = False
        if fl_setup.rfdc_clk:
            getLogger(__name__).debug(f'Requesting update to RFDC clocking configuration.')
            mkidgen3.drivers.rfdcclock.configure(**fl_setup.rfdc_clk.settings_dict())
            #time.sleep(0.5)
            clock_set = True

        if clock_set or fl_setup.bitstream:
            getLogger(__name__).debug(f'Requesting update to bitstream.')
            if fl_setup.bitstream:
                x = copy.deepcopy(self._default_bitstream)
                x.merge_with(fl_setup.bitstream)
            else:
                x = self._default_bitstream

            self._ol = Overlay(x.bitstream, ignore_version=x.ignore_version, download=True)
            mkidgen3.quirks.Overlay(self._ol).post_configure()
            ThreadedPLInterruptManager.get_manager()

        if fl_setup.rfdc:
            getLogger(__name__).debug(f'Requesting update to RFDC configuration.')
            self._ol.rfdc.enable_mts(dac=fl_setup.rfdc.dac_mts, adc=fl_setup.rfdc.adc_mts)
            self._ol.rfdc.set_gain(adc_gains=fl_setup.rfdc.adc_gains, dac_gains=fl_setup.rfdc.dac_gains)

        from mkidgen3.drivers.ppssync import PPSMode, PPSSource
        self._ol.pps_synchronization.pps_synchronizer_con_0.start_engine(
                mode=PPSMode.FORCE_START,
                start_second=None,
                skew=None,
                lockout=None,
                rollover_thresh=None,
                resync=False,
                pps_source=PPSSource.PPS0,
                load_time=None,
                clk_period_ns=1.953125,
                timeout=5,
                poll=1000*1000*1000)

        #NB these if comparisons must not be "is None" as an empty config can't be used to trigger reconfiguration

        if fl_setup.if_board:
            getLogger(__name__).debug(f'Requesting update to IF Board configuration.')
            try:
                self._if_board.configure(**fl_setup.if_board.settings_dict())
            except Exception as e:
                getLogger(__name__).warning(f"Unable to connect to IF board: {e}, proceeding without IF board.")

        if fl_setup.waveform:
            getLogger(__name__).debug(f'Configure DAC with {fl_setup.waveform.settings_dict()}')
            self._ol.dac_table.configure(**fl_setup.waveform.settings_dict())

        if fl_setup.chan:
            self._ol.photon_pipe.reschan.bin_to_res.configure(**fl_setup.chan.settings_dict())

        if fl_setup.ddc:
            self._ol.photon_pipe.reschan.ddccontrol_0.configure(**fl_setup.ddc.settings_dict())

        if fl_setup.filter:
            self._ol.photon_pipe.phasematch.configure(**fl_setup.filter.settings_dict())

        if fl_setup.trig:
            self._ol.trigger_system.trigger_1.configure(**fl_setup.trig.settings_dict())

        if fl_setup.if_board:
            self._if_board.settle()

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
            assert self._ol.capture.is_ready(), 'Capture Subsystem is busy'
        except AssertionError as e:
            failmsg = str(e)
        except AttributeError as e:
            failmsg = f'Something is fundamentally wrong. Check your bitstream. ({str(e)})'

        try:
            cr.establish(context=context)
        except zmq.ZMQError as e:
            failmsg += f"Unable to establish capture {cr.id} due to {e}, dropping request."

        if not failmsg and cr.fail_saturation_frac:  # check for ADC saturation
            buf = self._ol.capture.capture_adc(2**19, complex=False)
            max_val = np.abs(buf).max()
            del buf
            if max_val >= ADC_MAX_INT*cr.fail_saturation_frac:
                failmsg = (f"ADC levels ({max_val}) above saturation failure "
                           f"level ({ADC_MAX_INT*cr.fail_saturation_frac:.0f})")

        if failmsg:
            getLogger(__name__).error(failmsg)
            cr.fail(failmsg, raise_exception=False)
            try:
                pipe.close()
            except zmq.ZMQError:
                pass
            return

        self._ol.capture.keep_channels(cr.tap, cr.channels if cr.channels else 'all')
        hw_channels = tuple(sorted(self._ol.capture.kept_channels(cr.tap)))

        capture_atom_bytes = len(hw_channels)*cr.dwid
        hw_size_bytes = capture_atom_bytes*cr.nsamp
        demands = None
        chunking_thresh = determine_max_chunk('ps', demands=demands,
                                              assume_compression=not cr.data_endpoint.startswith('file://'))
        nchunks = hw_size_bytes // chunking_thresh
        partial = hw_size_bytes - chunking_thresh * nchunks
        chunks = [chunking_thresh // capture_atom_bytes] * nchunks
        if partial:
            chunks.append(partial // capture_atom_bytes)

        if len(chunks) > 1 and cr.numpy_metric:
            failmsg = ('Reducing captures via numpy metrics is not supported chunked captures. '
                       'Decrease the capture size or compute the metric client-side.')
            getLogger(__name__).error(failmsg)
            cr.fail(failmsg, raise_exception=False)
            try:
                pipe.close()
            except zmq.ZMQError:
                pass
            return

        channel_sel = None
        ps_buf = None
        if cr.channels is not None:  # channel-specific capture
            usr_channels = cr.channels
            if hw_channels != usr_channels:  # strip off extra channels required by hardware capture
                getLogger(__name__).debug(f'Requested channel capture of {format_bytes(cr.size_bytes)} '
                                          f'requires {format_bytes(hw_size_bytes)}')
                getLogger(__name__).debug(f'Requested channel subset of hardware capture '
                                              f'will copy {format_bytes(cr.size_bytes)} '
                                              f'PL buffer to PS RAM.')
                channel_sel = np.where(np.in1d(hw_channels, usr_channels))[0]
                buf_shape = (chunks[0], len(usr_channels))
                if 'iq' in cr.tap:
                    buf_shape += (2,)
                ps_buf = np.empty(buf_shape,  dtype=np.int16)

        getLogger(__name__).debug(f'Beginning plram capture loop of {len(chunks)} chunk(s) at {cr.tap}')

        try:
            for i, csize in enumerate(chunks):
                times = [time.perf_counter()]

                check_zmq_abort_pipe(pipe)

                times.append(time.perf_counter())
                data = self._ol.capture.capture(csize, cr.tap, groups=None, wait=True)
                times.append(time.perf_counter())
                if isinstance(data, dict):
                    raise RuntimeError(f'PL axis2mm capture failed: {data}. '
                                       f'Note decode errors are indicative of a memory address and '
                                       f'should never happen issue.')
                if channel_sel is None:
                    data_to_send = data
                else:
                    data_to_send = np.take(data, channel_sel, axis=1, out=ps_buf[:csize])
                times.append(time.perf_counter())

                zmqtmp = zmq.COPY_THRESHOLD
                zmq.COPY_THRESHOLD = 0
                tracker = cr.send_data(data_to_send, status=f'{i + 1}/{len(chunks)}', copy=False, compress=True)
                times.append(time.perf_counter())

                if tracker is not None:
                    tracker.wait()
                zmq.COPY_THRESHOLD = zmqtmp
                times.append(time.perf_counter())

                del data
                times.append(time.perf_counter())
                times_str = [format_time(x) for x in np.diff((times))]
                labels = ['check abort','execute capture','optionally slice','send data','wait for tracker','free buffer']
                getLogger(__name__).debug(list(zip(labels, times_str)))
            cr.finish()
        except AbortedException as e:
            cr.abort(e)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            cr.fail(f'Aborted due to {e}', raise_exception=False)
        finally:
            cr.destablish()
            del ps_buf
            getLogger(__name__).debug(f'plram cap complete')

    def photon_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context=None):
        """
        pipe: a zmq pair pipe to detect abort
        cr: the capture request
        """
        failmsg = ''
        photon_maxi = self._ol.trigger_system.photon_maxi_0
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
        # q = q_other = queue.SimpleQueue()  #an alternative

        fountain, stop = photon_maxi.photon_fountain(q_other, spawn=True, copy_buffer=True)

        def photon_sender(q: zmq.Socket, cr, unpack=False):
            log = getLogger(__name__)
            from datetime import datetime
            toc = 0
            delts = []
            try:
                cr.establish(context=context)
                while True:
                    tic = datetime.utcnow()
                    if toc:
                        delt = tic - toc
                        delts.append(delt.seconds*1e6+delt.microseconds)
                        x = tic.strftime('%M:%S.%f')
                        log.debug(f'Prep send @ {x}, since last wait ended {delt.seconds}.{delt.microseconds:06}s')

                    x = q.recv()
                    if x == b'':
                        toc = datetime.utcnow()
                        cr.finish()
                        break
                    photons = np.frombuffer(x, dtype=photon_maxi.PHOTON_PACKED_DTYPE)
                    toc = datetime.utcnow()

                    cr.send_data(unpack_photons(photons) if unpack else photons, copy=False, compress=True)
            except Exception as e:
                cr.abort(f'Uncaught exception in photon sender: {e}')
                raise e
            finally:
                cr.destablish()
                q.close()
            log.info(f'Sending done. Time between send waits: {delts} us')

        cr.destablish()
        sender = threading.Thread(target=photon_sender, args=(q, cr), name='Photon Sender')

        try:
            sender.start()
            fountain.start()
            time.sleep(.2)
            photon_maxi.capture(buffer_time_ms=cr.nsamp)
            while not cr.completed:
                check_zmq_abort_pipe(pipe)
        except AbortedException as e:
            getLogger(__name__).error(f'Aborting photon capture {cr} due user request.')
            stop.set()
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            stop.set()
        finally:
            self._ol.trigger_system.photon_maxi_0.stop_capture()
            fountain.join()
            sender.join()
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
            if isinstance(q, zmq.Socket):
                q.close()
                q_other.close()

    def postage_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context: zmq.Context = None):
        failmsg = ''
        postage_filt = self._ol.trigger_system.postage_filter_0
        postage_maxi = self._ol.trigger_system.postage_maxi_0
        try:
            assert cr.type == 'postage', 'Incorrect capture request type'
            assert postage_maxi.register_map.CTRL.AP_IDLE == 1, 'Postage MAXI is busy'
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
            postage_filt.configure(monitor_channels=cr.channels)
            _, postdone = ThreadedPLInterruptManager.get_monitor(postage_maxi, id='frs_postage_cap')
            postdone.clear()
            postage_maxi.capture(max_events=cr.nsamp, wait=False)

            while not postdone.is_set():
                check_zmq_abort_pipe(pipe)
                time.sleep(min(postage_maxi.MAX_CAPTURE_TIME_S / 10, .1))
            cr.send_data(postage_maxi.get_postage(rawbuffer=True), copy=False, compress=True)
            cr.finish()
        except AbortedException as e:
            cr.abort(e, raise_exception=False)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            cr.fail(f'Failed due to {e}', raise_exception=False)
        finally:
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
