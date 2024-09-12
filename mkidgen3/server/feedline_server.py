import logging
import pickle
import time

from logging import getLogger

from mkidgen3.server.captures import CaptureRequest
from mkidgen3.server.feedline_config import RFDCConfig
from mkidgen3.server.fpga_objects import FeedlineHardware, DEFAULT_BIT_FILE
from mkidgen3.server.misc import zpipe
from mkidgen3.util import check_active_jupyter_notebook
import asyncio
import zmq
import threading
from mkidgen3.util import setup_logging
from datetime import datetime
import argparse

COMMAND_LIST = ('reset', 'capture', 'bequiet', 'status')


class TapThread:
    def __init__(self, target, cr:CaptureRequest):
        context = zmq.Context().instance()
        a, b = zpipe(context)
        t = threading.Thread(target=target, name=f"TapThread: {cr.tap}:{cr.id}", args=(b, cr), kwargs=dict(context=context))
        self.thread = t
        self.request = cr
        self._pipe = a
        self._other_pipe = b
        t.start()

    def __repr__(self):
        a = 'running' if self.thread.is_alive() else 'stopped'
        return f'<{self.thread.name} ({a})>'

    def abort(self):
        try:
            # Abort the thread, not the request, the thread will handle the abort of the request if necessary!
            getLogger(__name__).debug(f'Sending abort to worker: {self}')
            self._pipe.send(b'abort')
        except zmq.ZMQError:
            getLogger(__name__).critical(f'Error sending abort to worker thread {self.thread}')
            raise

    def __del__(self):
        try:
            self._pipe.close()
            self._other_pipe.close()
        except zmq.error.ContextTerminated:
            pass

class FeedlineReadoutServer:
    def __init__(self, bitstream, clock_source='4.096GSPS_MTS_dualloop', if_port='dev/ifboard', ignore_version=False,
                 program_clock=True, mts=True, download=False):
        self.hardware = FeedlineHardware(bitstream, clock_source=clock_source, if_port=if_port,
                                         ignore_version=ignore_version, program_clock=program_clock,
                                         rfdc=RFDCConfig(dac_mts=mts, adc_mts=False), download=download)

        self._tap_threads = {k: None for k in ('photon', 'postage', 'engineering')}
        self._to_check = []
        self._checked = []
        self._cap_pipe=None

    def status(self):
        """

        Returns: Dictionary of status information

        """
        status = {'hardware': self.hardware.status(),
                  'running_captures': self._running_captures(),
                  'pending_captures': self._pending_captures()}

        return status

    def _running_captures(self):
        return tuple([tt.request.id for tt in list(self._tap_threads.values())])

    def _pending_captures(self):
        return tuple([cr.id for cr in self._checked+self._to_check])

    def _abort_all(self, join=False, reason='Abort all', raisezmqerror=True, also:tuple|list|CaptureRequest=tuple()):
        toabort=self._checked + self._to_check
        if isinstance(also, CaptureRequest):
            also = [also]
        toabort.extend(also)
        for cr in toabort:
            cr.establish()
            cr.abort(reason)  # signal that captures will never happen
            cr.destablish()
        self._checked, self._to_check = [], []
        # stop any running tap threads
        for tt in (tt for tt in self._tap_threads.values() if tt is not None):
            try:
                tt.abort()
            except zmq.ZMQError as e:
                if raisezmqerror:
                    raise e
        if join:
            for tt in self._tap_threads.values():
                if tt:
                    tt.thread.join()

    def _abort_by_id(self, id):
        aborted = False
        running_by_id = {tt.request.id: tt for tt in self._tap_threads.values() if tt is not None}
        if id in running_by_id:
            tt = running_by_id[id]
            aborted = True
            getLogger(__name__).debug(f'Found request {id} being serviced in {tt}. Aborting servicer.')
            tt.abort()
        for cr in filter(lambda x: x.id == id, self._checked):
            aborted = True
            getLogger(__name__).debug(f'Found request {id} in list of checked pending CR. Aborted')
            self._checked.pop(self._checked.index(cr))
            cr.abort('Abort by id')
        for cr in filter(lambda x: x.id == id, self._to_check):
            aborted = True
            getLogger(__name__).debug(f'Found request {id} in list of pending CR to be checked. Aborted')
            self._to_check.pop(self._to_check.index(cr))
            cr.abort('Abort by id')

        if not aborted:
            getLogger(__name__).info(f'Capture request {id} is unknown and cannot be aborted.')

    def create_capture_handler(self, start=True, daemon=False, context: zmq.Context = None):
        self._cap_pipe, self._cap_pipe_thread = zpipe(context or zmq.Context.instance())

        thread = threading.Thread(name='FRS CR Handler', target=self._main, args=(self._cap_pipe_thread,),
                                  kwargs={'context': context}, daemon=daemon)
        if start:
            thread.start()

        return thread

    def terminate_capture_handler(self):
        if self._cap_pipe:
            self._cap_pipe.send_pyobj(('exit', None))
            self._cap_pipe.close()

    def __del__(self):
        try:
            self._cap_pipe_thread.close()
        except:
            pass

    def abort_all(self):
        if self._cap_pipe:
            self._cap_pipe.send_pyobj(('abort', 'all'))

    def abort(self, id):
        if self._cap_pipe:
            self._cap_pipe.send_pyobj(('abort', id))

    def capture(self, capture_request):
        if self._cap_pipe:
            self._cap_pipe.send_pyobj(('capture', capture_request))

    def _cleanup_completed(self):
        """ Return true iff the effective config requirements changed """
        # Check to see if any capture threads have finished
        complete = [k for k, t in self._tap_threads.items() if t is not None and not t.thread.is_alive()]
        # for each finished capture thread remove its settings from the requirements pot and cleanup

        if bool(complete):  # need to check up to the size of the queue if anything finished
            # if the effective settings didn't change on completion we don't need to recheck stuff
            #  ignore this optimization for now
            self._to_check.extend(self._checked)
            self._checked = []

        effective_changed = False
        for k in complete:
            effective_changed |= self.hardware.derequire_config(self._tap_threads[k].request.id)
            del self._tap_threads[k]
            self._tap_threads[k] = None

        return effective_changed

    def _main(self, pipe: zmq.Socket, context: zmq.Context = None):
        """
        Enqueue a list of capture requests for future handling. Invalid requests are dealt with immediately and not
        enqueued.

        Args:
            pipe: a pipe for receiving capture requests
            context: zmq.Context

        Returns: None

        """
        context = context or zmq.Context().instance()

        #THIS LOOP SEEMS TO ACTUALLY be getting used ONLY to be deleted with the pynq.UIO reader deletion
        #when the threaded interrupt manager starts stuff up and nixes pynqs reader
        # (that apparently exists in this thread)
        try:
            aio_eloop = asyncio.get_running_loop()
        except RuntimeError:
            aio_eloop = asyncio.new_event_loop()
            getLogger(__name__).warning('Creating but not starting a thread that really should be axed and '
                                        'optimized away')
            t=threading.Thread(daemon=True, target=aio_eloop.run_forever, name='plramcap_eloop')

        asyncio.set_event_loop(aio_eloop)

        getLogger(__name__).info('Main thread starting')
        while True:

            effective_changed = self._cleanup_completed()

            running_by_id = {tt.request.id: tt for tt in self._tap_threads.values() if tt is not None}

            cr = None  # CR is the capture request that will be ckicked off this iteration of the loop
            # check for any incoming info: CapRequest, ABORT id|all, EXIT
            cmd, data = '', ''
            try:
                cmd, data = pipe.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    self._abort_all(reason='Keyboard interrupt')
                    if e.errno == zmq.ETERM:
                        break
                    else:
                        raise e  # real error

            if cmd not in ('exit', 'abort', 'capture', ''):
                getLogger(__name__).error(f'Received invalid command "{cmd}"')
                cmd, data = '', ''

            if cmd == 'exit':
                self._abort_all(join=True)
                break
            elif cmd == 'abort':
                if data == 'all':
                    self._abort_all(join=True)
                else:
                    self._abort_by_id(data)
            elif cmd == 'capture':
                self.hardware.config_manager.learn(data.feedline_config)
                unknown = self.hardware.config_manager.unlearned_hashes(data.feedline_config)
                if unknown:  # We've never been sent the full config necessary
                    try:
                        data.establish()
                        #TODO this code seems to imply json "{'resp': 'ERROR', 'data': unknown}"
                        data.fail(f'ERROR: Full FeedlineConfig never sent: {unknown}')
                    except zmq.ZMQError as e:
                        getLogger(__name__).error(f'Unable to fail request with hashed config due to {e}. '
                                                  f'Silently dropping request {data.id}')
                elif (not self._to_check and self._tap_threads[data.type] is None and
                      self.hardware.config_compatible_with(data.feedline_config)):
                    cr = data  # this can be run and nothing else, so it will be done below
                else:
                    q = self._to_check if self._to_check else self._checked
                    try:
                        data.set_status('queued', f'Queued', destablish=True)
                        q.append(data)
                    except zmq.ZMQError as e:
                        getLogger(__name__).error(f'Unable to update status due to {e}. Silently dropping request'
                                                  f' {data.id}')

                # cant be run because there might be something more important (we check anyway),
                # the tap is in use (we check when the tap finishes)
                # settings aren't compatible (we will check when something finishes)

            if not cr:
                try:
                    cr = self._to_check.pop(0)
                except IndexError:
                    continue

            assert isinstance(cr, CaptureRequest)

            try:
                if self._tap_threads[cr.type] is not None:
                    cr.set_status('queued', f'tap location in use by: {self._tap_threads[cr.type].request.id}')
                    self._checked.append(cr)
                    continue
                else:
                    if not self.hardware.config_compatible_with(cr.feedline_config):
                        cr.set_status('queued', f'incompatible with one or more of: {running_by_id.keys()}')
                        self._checked.append(cr)
                        continue
            except zmq.ZMQError as e:
                getLogger(__name__).error(f'Unable to update status due to {e}. Silently aborting request {cr}.')
                continue

            cr.destablish()  # ensure nothing lingers from any status messages

            try:
                self.hardware.apply_config(cr.id, cr.feedline_config)
            except Exception as e:
                self.hardware.derequire_config(id)  # Not  necessary as we are dying, but let's die in a clean house
                getLogger(__name__).critical(f'Hardware settings failure: {e}. Aborting all requests and dying.')
                self._abort_all(reason='Hardware settings failure', raisezmqerror=False, join=False, also=cr)
                break

            self.start_tap_thread(cr)

        aio_eloop.close()
        getLogger(__name__).info('Capture thread exiting')

    def start_tap_thread(self, cr):
        """
        Start a thread to service a capture request

        Args:
            cr: a capture request.

        Returns:

        """
        assert self._tap_threads.get(cr.type, None) is None, 'Only one TapThread per location may be created at a time'
        cap_runners = {'engineering': self.hardware.plram_cap, 'photon': self.hardware.photon_cap,
                       'postage': self.hardware.postage_cap}
        target = cap_runners[cr.type]


        cr.set_status('running', f'Started at UTC {datetime.utcnow()}', destablish=True)

        self._tap_threads[cr.type] = TapThread(target, cr)


def parse_cl():
    parser = argparse.ArgumentParser(description='Feedline Readout Server', add_help=True)
    parser.add_argument('-p', '--port', dest='port', action='store', required=False, type=int,
                        help='Server port', default='8888')
    parser.add_argument('--cap_port', dest='capture_port', action='store', required=False, type=int,
                        help='Capture Data Port', default='8889')
    parser.add_argument('--sta_port', dest='status_port', action='store', required=False, type=int,
                        help='Capture Status Port', default='8890')
    parser.add_argument('--clock', dest='clock', action='store', required=False, type=str,
                        help='Clock Source', default='default')
    parser.add_argument('-b', '--bitstream', dest='bitstream', action='store', required=False, type=str,
                        help='bitstream file',
                        default=DEFAULT_BIT_FILE)
    parser.add_argument('--if', dest='ifboard', action='store', required=False, type=str,
                        help='IF Board device', default='/dev/ifboard')
    parser.add_argument('--iv', dest='ignore_fpga_driver_version', action='store_true', required=False,
                        help='Ignore FPGA driver version checks', default=False)
    return parser.parse_args()


def start_zmq_devices(cap_addr, stat_addr):
    from zmq.devices import ThreadDevice

    cap_addr_internal = 'inproc://cap_data.xsub'
    stat_addr_internal = 'inproc://cap_stat.xsub'
    # Set up a proxy for routing all the capture requests
    dtd = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
    dtd.setsockopt_in(zmq.LINGER, 0)
    dtd.setsockopt_out(zmq.LINGER, 0)
    dtd.bind_in(cap_addr_internal)
    dtd.bind_out(cap_addr)
    dtd.daemon = True
    dtd.start()
    getLogger(__name__).info(f'Publishing capture data to {cap_addr} from relay @ {cap_addr_internal}')

    std = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
    std.setsockopt_in(zmq.LINGER, 0)
    std.setsockopt_out(zmq.LINGER, 0)
    std.bind_in(stat_addr_internal)
    std.bind_out(stat_addr)
    std.daemon = True
    std.start()
    getLogger(__name__).info(f'Publishing capture status information to {stat_addr} from relay @ {stat_addr_internal}')

    return dtd, std


if __name__ == '__main__':

    import os, time
    os.environ['TZ'] = 'right/UTC'
    time.tzset()
    setup_logging('feedlinereadoutserver')

    if check_active_jupyter_notebook():
        raise RuntimeError('Jupyter notebooks are running, shut them down first.')

    args = parse_cl()

    context = zmq.Context.instance(io_threads=2)
    context.linger = 1

    fr = FeedlineReadoutServer(args.bitstream, clock_source=args.clock, if_port=args.ifboard,
                               ignore_version=args.ignore_fpga_driver_version, program_clock=True, mts=True,
                               download=False)

    # Set up proxies for routing all the capture data and status
    cap_addr = f'tcp://*:{args.capture_port}'
    stat_addr = f'tcp://*:{args.status_port}'
    start_zmq_devices(cap_addr, stat_addr)

    # Set up a command port
    command_port = args.port
    cmd_addr = f"tcp://*:{command_port}"
    socket = context.socket(zmq.REP)
    socket.bind(cmd_addr)

    # Start up the main thread
    thread = fr.create_capture_handler(context=context, start=True, daemon=False)

    getLogger(__name__).info(f'Accepting commands on {cmd_addr}')

    while True:
        try:
            cmd, arg = socket.recv_pyobj()
        except zmq.ZMQError as e:
            getLogger(__name__).error(f'Caught {e}, aborting and shutting down')
            fr.terminate_capture_handler()
            break
        except KeyboardInterrupt:
            getLogger(__name__).error(f'Keyboard Interrupt, aborting and shutting down')
            fr.terminate_capture_handler()
            break
        except pickle.UnpicklingError:
            socket.send_pyobj('ERROR: Ignoring unpicklable command')
            getLogger(__name__).error(f'Ignoring unpicklable command')
            continue
        else:
            if not thread.is_alive():
                getLogger(__name__).critical(f'Capture thread has died prematurely. All existing captures will '
                                             f'never complete. Exiting.')
                socket.send_pyobj('ERROR')
                break

        getLogger(__name__).debug(f'Received command "{cmd}" with args {arg}')

        if cmd == 'reset':
            socket.send_pyobj('OK')
            fr.terminate_capture_handler()
            thread.join()
            fr.hardware.reset()
            thread = fr.create_capture_handler(context=zmq.Context.instance(), start=True, daemon=False)

        elif cmd == 'status':
            try:
                status = fr.status()  # this might take a while and fail
            except Exception as e:
                status = {'hardware': str(e)}
            status['id'] = f'FRS {args.fl_id} @ {args.port}/{args.cap_port}'
            socket.send_pyobj(status)

        elif cmd == 'bequiet':
            fr.abort_all()
            try:
                fr.hardware.bequiet(**arg)  # This might take a while and fail
                socket.send_pyobj('OK')
            except Exception as e:
                socket.send_pyobj(f'ERROR: {e}')

        elif cmd == 'capture':
            fr.capture(arg)
            socket.send_pyobj({'resp': 'OK', 'code': 0})

        elif cmd == 'abort':
            fr.abort(arg)
            socket.send_pyobj({'resp': 'OK', 'code': 0})

        else:
            socket.send_pyobj({'resp': 'ERROR', 'code': 0})

    thread.join()
    socket.close()
    context.term()
