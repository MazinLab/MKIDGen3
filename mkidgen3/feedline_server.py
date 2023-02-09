import json
import logging

import mkidgen3.clocking
from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.schema import validate
from mkidgen3.status_keeper import StatusKeeper
from logging import getLogger
import pynq
import mkidgen3.drivers.rfdc
import mkidgen3 as g3
from mkidgen3.objects import CaptureRequest, CaptureAbortedException, FeedlineConfig, FeedlineStatus, DACStatus, \
    DDCStatus, \
    FLPhotonBuffer
from typing import List
import zmq
import blosc2
import time
import threading
from datetime import datetime
import argparse
import binascii
import os
import numpy as np

COMMAND_LIST = ('reset', 'capture', 'bequiet', 'status')

CHUNKING_THRESHOLD = 1000


def zpipe(ctx):
    """build inproc pipe for talking to threads
    mimic pipe used in czmq zthread_fork.
    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b


class FeedlineConfigSet:
    def __init__(self):
        self._dict = {}

    def effective(self) -> FeedlineConfig:
        """Return a FLSettings (or ducktype) resulting from settings in the set"""
        return FeedlineConfig()

    def pop(self, id) -> bool:
        """
        Remove settings from the set

        Args:
            id: settings id to remove, raises KeyError if object isn't in the set

        Returns: True iff the effective settings changed as a result of the pop
        """
        x = self.effective
        self._dict.pop(id)
        return self.effective != x

    def add(self, id, config: FeedlineConfig) -> FeedlineConfig:
        """

        Args:
            id: an id for the settings (for later removal from the set)
            config: a feedline config at any level of specificity

        Returns: A feedline config with the settings that need updating populated and the rest None

        """
        self._dict[id] = config
        return FeedlineConfig()


class DummyOverlay:
    class DummyBuffer(np.ndarray):
        def freebuffer(self):
            pass

    class DummyCap:
        @staticmethod
        def capture(csize, *args, **kwargs):
            n = csize*2
            return np.random.uniform(low=-10000, high=10000, size=n).astype(np.int16).view(DummyOverlay.DummyBuffer)

        @staticmethod
        def ready():
            return True

    def __init__(self, bitstream, *args, **kwargs):
        self.capture = DummyOverlay.DummyCap()


class FeedlineHardware:
    def __init__(self, bitstream, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False, download=False, program_clock=True):

        self._config_pot = FeedlineConfigSet()  # a set of FeedlineStettings
        self._clock_source = validate(clock_source=clock_source, error=True)
        try:
            self._ol = pynq.Overlay(bitstream, download=download, ignore_version=ignore_version)
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
            mkidgen3.clocking.start_clocks(external_10mhz=clock_source == 'external_10mhz')

    def reset(self):
        self._if_board.power_off(save_settings=False)
        mkidgen3.clocking.start_clocks(external_10mhz=self._clock_source == 'external_10mhz')
        self._ol = pynq.Overlay(self._ol.bitfile_name, ignore_version=self._ignore_version, download=True)
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
        return self._config_pot.effective()==config

    def derequire_config(self, id):
        try:
            self._config_pot.pop(id)
        except KeyError:
            pass

    def apply_config(self, id, config: FeedlineConfig):
        """Takes and applies a config to the hardware, updates and tracks the effective set of settings"""
        # IF board
        fl_setup = self._config_pot.add(id, config)

        # IF Board
        if fl_setup.if_setup is not None:
            getLogger(__name__).debug(f'Configure IF Board with {fl_setup.if_setup.settings_dict()}')
            self._if_board.configure(**fl_setup.dac_setup.settings_dict())

        # DAC
        if fl_setup.dac_setup is not None:
            getLogger(__name__).debug(f'Configure DAC with {fl_setup.dac_setup.settings_dict()}')
            self._ol.dac_replay.configure(**fl_setup.dac_setup.settings_dict())

        # ADC
        if fl_setup.adc_setup is not None:
            getLogger(__name__).debug(f'Configure ADC with {fl_setup.adc_setup.settings_dict()}')
            # self._ol.dac_replay.configure(**fl_setup.dac_setup.settings_dict())

        # Photon Pipe
        if fl_setup.pp_setup is not None:
            # Channel assignments
            if fl_setup.pp_setup.chan_config is not None:
                self._ol.photon_pipe.reschan.bin_to_res.configure(**fl_setup.pp_setup.chan_config.settings_dict())
            # DDC
            if fl_setup.pp_setup.ddc_config is not None:
                self._ol.photon_pipe.reschan.ddc.configure(**fl_setup.pp_setup.ddc_config.settings_dict())
            # Matched Filters
            if fl_setup.pp_setup.filter_config is not None:
                self._ol.photon_pipe.phasematch.configure(**fl_setup.pp_setup.filter_config.settings_dict())
            # Matched Filters
            if fl_setup.pp_setup.trig_config is not None:
                self._ol.photon_pipe.phasematch.configure(**fl_setup.pp_setup.trig_config.settings_dict())


class FeedlineReadout:
    def __init__(self, bitstream, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False):
        self.hardware = FeedlineHardware(bitstream, clock_source=clock_source, if_port=if_port,
                                         ignore_version=ignore_version, download=True, program_clock=True)

        # self.status_keeper = StatusKeeper(status_port)  #TODO

    def status(self):
        """

        Returns: Dictionary of status information

        """
        status = {'hardware': self.hardware.status(),
                  'running_captures': self._running_captures(),
                  'pending_captures': self._pending_captures()}

        return status

    @staticmethod
    def plram_cap(pipe, cr, ol: pynq.Overlay, context=None):
        """

        Args:
            context:
            cr: A CaptureRequest object
            ol: A pynq.Overlay with the firmware bitstream loaded, assumed to be thread safe

        Returns: None

        """
        failmsg = ''
        try:
            assert cr.type == 'engineering', 'Incorrect capture request type'
            assert ol.capture.ready(), 'Capture Subsystem is busy'
        except AssertionError as e:
            failmsg = str(e)

        # try:
        #     abort = context.socket(
        #         zmq.SUB)  # TODO Use of subscribe will probably result in missed abort messages, REQ/REP?
        #     abort.setsockopt(zmq.SUBSCRIBE, id)
        #     abort.connect('inproc://cap_abort')
        # except zmq.EFAULT:
        #     failmsg = f"Unable to establish abort socket {cr.id}, dropping request."

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

        try:
            nchunks = cr.size // CHUNKING_THRESHOLD
            partial = cr.size - CHUNKING_THRESHOLD * nchunks
            chunks = [CHUNKING_THRESHOLD] * nchunks
            if partial:
                chunks.append(partial)
            for i, csize in enumerate(chunks):
                try:
                    abort = pipe.recv(zmq.NOBLOCK)
                    raise CaptureAbortedException(abort)
                except zmq.ZMQError as e:
                    if e.errno != zmq.EAGAIN:
                        raise
                data = ol.capture.capture(csize, tap=cr.tap, wait=True)
                # data = np.random.uniform(-10000,10000, size=csize*2).astype(np.int16)
                cr.add_data(data, status=f'Captured {i} of {len(chunks)}')
                data.free_buffer()
            cr.finish()
        except CaptureAbortedException as e:
            cr.abort(e)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            try:
                cr.fail(f'Aborted due to {e}')
            except zmq.EFAULT as ez:
                getLogger(__name__).warning(f'Failed to send abort message {cr} due to {ez}')
        finally:
            del cr

    @staticmethod
    def photon_cap(pipe, context, cr, ol):
        try:
            cr.establish(context)
        except Exception:
            getLogger(__name__).error(f"Unable to establish capture {cr.id}, dropping request.")
            return

        abort = context.socket(zmq.SUB)
        abort.setsockopt(zmq.SUBSCRIBE, id)
        abort.connect('inproc://cap_abort')

        buffers = []
        try:
            buffers = [FLPhotonBuffer() for _ in range(2)]
            buf = None
            while not cr.aborted() and abort.poll(1) == 0:
                to_send = buf
                buf = buffers.pop(0)
                buffers.append(buf)
                ol.photon_capture.capture(buf.buffer)
                if to_send:
                    cr.add_data(to_send.buffer)
                while not buf.full and not abort.poll(5):
                    pass
            ol.photon_capture.stop()
            cr.add_data(buf.buffer)
            cr.finish()
        except Exception as e:
            getLogger(__name__).error(f'Terminating capture {id} due to {e}')
            cr.fail(f'Aborted due to {str(e)}')
        finally:
            ol.photon_capture.stop()
            del cr
            while buffers:
                b = buffers.pop()
                del b
            abort.close()

    @staticmethod
    def stamp_cap(pipe, context, cr, ol):
        pass

    def main(self, pipe: zmq.Socket, context: zmq.Context = None):
        """
        Enqueue a list of capture requests for future handling. Invalid requests are dealt with immediately and not
        enqueued.

        Args:
            zpipe: a pipe for receiving capture requests
            conext: zmq.Context

        Returns: None

        """
        context = context or zmq.Context().instance()

        class TapThread:
            def __init__(self, thread, pipe, other_pipe, request):
                self.thread = thread
                self.request = request
                self.pipe = pipe
                self._other_pipe = other_pipe

            def __del__(self):
                self.pipe.close()
                self._other_pipe.close()

        tap_threads = {k: None for k in ('photon', 'stamp', 'engineering')}

        to_check = []
        checked = []
        getLogger(__name__).info('Main thread starting')
        while True:
            # Check to see if any capture threads have finished
            complete = [k for k, t in tap_threads.items() if t is not None and not t.thread.is_alive()]
            # for each finished capture thread remove its settings from the requirements pot and cleanup

            if bool(complete):  # need to check up to the size of the queue if anything finished
                to_check.extend(checked)
                checked = []

            for k in complete:
                self.hardware.derequire_config(tap_threads[k].request.id)
                del tap_threads[k]
                tap_threads[k] = None

            running_by_id = {tt.request.id: tt for tt in tap_threads.values() if tt is not None}

            # check for any incoming info: CapRequest, ABORT id|all, EXIT
            cmd, data = '', ''
            try:
                cmd, data = pipe.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    for cr in checked + to_check:
                        cr.abort('Keyboard interrupt')  # signal that captures will never happen
                    for v in running_by_id.values():
                        v.pipe.send('abort')
                    break
                elif e.errno != zmq.EAGAIN:
                    raise e  # real error
            else:
                if cmd not in ('exit', 'abort', 'capture'):
                    getLogger(__name__).error(f'Received invalid command "{cmd}"')
                    cmd, data = '', ''

            def abort_all():
                for cr in checked + to_check:
                    cr.abort('Abort all')  # signal that captures will never happen
                for v in running_by_id.values():  # stop any running tap threads
                    try:
                        v.pipe.send('abort')  # TODO what happens to pipe when thread ends?
                    except zmq.ZMQError:
                        getLogger(__name__).critical('Error sending abort to worker thread. Exiting')
                        raise

            def abort_by_id(id):
                aborted = False
                if id in running_by_id:
                    aborted = True
                    running_by_id[id].pipe.send('abort')  # TODO what happens to pipe when thread ends?
                for cr in filter(lambda x: x.id == id, checked):
                    aborted = True
                    checked.pop(checked.index(cr))
                    cr.abort('Abort by id')
                for cr in filter(lambda x: x.id == id, to_check):
                    aborted = True
                    to_check.pop(to_check.index(cr))
                    cr.abort('Abort by id')

                if not aborted:
                    getLogger(__name__).info(f'Capture request {id} is unknown and cannot be aborted.')

            if cmd == 'exit':
                abort_all()
                break

            if cmd == 'abort':
                if data == 'all':
                    abort_all()
                    checked, to_check = [], []
                else:
                    abort_by_id(data)

            cr = None  # CR is the capture request that will be ckicked off this iteration of the loop
            if cmd == 'capture':
                if (not to_check and tap_threads[data.type] is None and
                        self.hardware.config_compatible_with(data.feedline_config)):
                    cr = data  # this can be run and nothing else, so it will be done below
                else:
                    q = to_check if to_check else checked
                    try:
                        data.set_status('queued', f'Queued')
                        q.append(data)
                    except zmq.ZMQError as e:
                        getLogger(__name__).error(f'Unable to update status due to {e}. Silently dropping request'
                                                  f' {data.id}')

                # cant be run because there might be something more important (we check anyway),
                # the tap is in use (we check when the tap finishes)
                # settings aren't compatible (we will check when something finishes)

            if not cr:
                try:
                    cr = to_check.pop(0)
                except IndexError:
                    continue

            assert isinstance(cr, CaptureRequest)

            try:
                if tap_threads[cr.type] is not None:
                    cr.set_status('queued', f'tap location in use by: {tap_threads[cr.type].request.id}')
                    checked.append(cr)
                    continue
                elif not self.hardware.config_compatible_with(cr.feedline_config):
                    cr.set_status('queued', f'incompatible with one or more of: {running_by_id.keys()}')
                    checked.append(cr)
                    continue
            except zmq.ZMQError as e:
                getLogger(__name__).error(f'Unable to update status due to {e}. Silently aborting request {cr}.')
                continue

            try:
                self.hardware.apply_config(cr.id, cr.feedline_config)
            except Exception as e:
                getLogger(__name__).critical(f'Hardware settings failure: {e}. Aborting all requests and dying.')
                for cr in checked + to_check:
                    cr.abort('Hardware settings failure')  # signal that captures will never happen
                for v in running_by_id.values():  # stop any running tap threads
                    try:
                        v.pipe[0].send('abort')  # TODO what happens to pipe when thread ends?
                    except zmq.ZMQError:
                        getLogger(__name__).critical(f'Error sending abort to worker thread {v}.')
                break

            cap_runners = {'engineering': self.plram_cap, 'photon': self.photon_cap, 'stamp': self.stamp_cap}
            target = cap_runners[cr.type]
            a, b = zpipe(context)
            cr.set_status('running', f'Started at UTC {datetime.utcnow()}')
            cr.destablish()
            t = threading.Thread(target=target, name=f"CapThread: {cr.id}",
                                 args=(b, cr, self.hardware._ol), kwargs=dict(context=context))
            t.start()
            tap_threads[cr.type] = TapThread(t, a, b, cr)

        getLogger(__name__).info('Capture thread exiting')
        # context.term()


def parse_cl():
    parser = argparse.ArgumentParser(description='Feedline Readout Server', add_help=True)
    parser.add_argument('-p', '--port', dest='port', action='store', required=False, type=int,
                        help='Server port', default='8888')
    parser.add_argument('--cap_port', dest='capture_port', action='store', required=False, type=int,
                        help='Capture Data Port', default='8889')
    parser.add_argument('--sta_port', dest='status_port', action='store', required=False, type=int,
                        help='Capture Status Port', default='8890')
    parser.add_argument('--clock', dest='clock', action='store', required=False, type=str,
                        help='Clock Source', default='external_10mhz')
    parser.add_argument('-b', '--bitstream', dest='bitstream', action='store', required=False, type=str,
                        help='bitstream file', default='')
    parser.add_argument('--if', dest='if_board', action='store', required=False, type=str,
                        help='IF Board device', default='/dev/if_board')
    parser.add_argument('--iv', dest='ignore_fpga_driver_version', action='store_true', required=False,
                        help='Ignore FPGA driver version checks', default=False)
    return parser.parse_args()


def zpipe(ctx):
    """
    build an inproc pipe for talking to threads
    mimic pipe used in czmq zthread_fork.
    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    args = parse_cl()
    context = zmq.Context.instance()
    context.linger = 0


    fr = FeedlineReadout(args.bitstream, clock_source=args.clock, if_port=args.if_board,
                         ignore_version=args.ignore_fpga_driver_version)

    capture_port = args.capture_port
    status_port = args.status_port
    command_port = args.port
    from zmq.devices import ThreadDevice

    cap_addr = f'tcp://*:{capture_port}'
    stat_addr = f'tcp://*:{status_port}'
    cmd_addr = f"tcp://*:{command_port}"
    # Set up a proxy for routing all the capture requests
    dtd = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
    dtd.setsockopt_in(zmq.LINGER, 0)
    dtd.setsockopt_out(zmq.LINGER, 0)
    dtd.bind_in('inproc://cap_data.xsub')
    dtd.bind_out(cap_addr)
    dtd.daemon = True
    dtd.start()
    getLogger(__name__).info(f'Publishing capture data to {cap_addr}')

    std = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.PUB)
    std.bind_in('inproc://cap_status.xsub')
    std.bind_out(stat_addr)
    std.setsockopt_in(zmq.LINGER, 0)
    std.setsockopt_out(zmq.LINGER, 0)
    dtd.daemon = True
    std.start()
    getLogger(__name__).info(f'Publishing capture status information to {stat_addr}')

    # Set up a command port

    socket = context.socket(zmq.REP)
    socket.bind(cmd_addr)
    getLogger(__name__).info(f'Accepting commands on {cmd_addr}')

    cap_pipe, cap_pipe_thread = zpipe(zmq.Context.instance())

    main = threading.Thread(name='CaptureHandler', target=fr.main, args=(cap_pipe_thread,),
                            kwargs={'context': context})
    main.daemon = False
    main.start()

    while True:
        try:
            cmd, arg = socket.recv_pyobj()
        except zmq.ZMQError as e:
            getLogger(__name__).error(f'Caught {e}, aborting and shutting down')
            cap_pipe.send_pyobj(('exit', None))
            break
        except KeyboardInterrupt:
            getLogger(__name__).error(f'Keyboard Interrupt aborting and shutting down')
            cap_pipe.send_pyobj(('exit', None))
            break
        else:
            if not main.is_alive():
                getLogger(__name__).critical(f'Capture thread has died prematurely. All existing captures will '
                                             f'never complete. Exiting.')
                socket.send_json('ERROR')
                break
        getLogger(__name__).debug(f'Recieved command "{cmd}" with args {arg}')
        if cmd == 'reset':
            cap_pipe.send_pyobj(('exit', None))
            main.join()
            fr.hardware.reset()
            main = threading.Thread(name='CaptureHandler', target=fr.main, args=(cap_pipe_thread,),
                                    kwargs={'context': context})
            main.daemon = False
            main.start()
            socket.send_json('OK')
        elif cmd == 'status':
            try:
                status = fr.status()  # this might take a while and fail
            except Exception as e:
                status = {'hardware': str(e)}
            status['id'] = f'FRS {args.fl_id} @ {args.port}/{args.cap_port}'
            socket.send_json(status)
        elif cmd == 'bequiet':
            cap_pipe.send_pyobj(('abort', 'all'))
            try:
                fr.hardware.bequiet(**json.loads(arg))  # This might take a while and fail
                socket.send_json('OK')
            except Exception as e:
                socket.send_json(f'ERROR: {e}')
        elif cmd == 'capture':
            cap_pipe.send_pyobj(('capture', arg))
            socket.send_json('OK')

    main.join()
    socket.close()
    cap_pipe.close()
    cap_pipe_thread.close()
    context.term()
