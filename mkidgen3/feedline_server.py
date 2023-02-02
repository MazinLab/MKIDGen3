import json

from mkidgen3.drivers.ifboard import IFBoard
from mkidgen3.schema import validate
from mkidgen3.status_keeper import StatusKeeper
from logging import getLogger
import pynq
import mkidgen3.drivers.rfdc
import mkidgen3 as g3
from objects import CaptureRequest, CaptureAbortedException, FeedlineSetup, FeedlineStatus, DACStatus, DDCStatus, \
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


class FLSettingsSet:
    def __init__(self):
        self._dict = {}

    def effective(self) -> FeedlineSetup:
        """Return a FLSettings (or ducktype) resulting from settings in the set"""
        pass

    def pop(self, flsettings: FeedlineSetup) -> bool:
        """
        Remove settings from the set

        Args:
            flsettings: settings to remove, raises KeyError if object isn't in the set

        Returns: True iff the effective settings changed as a result of the pop

        """
        x = self.effective
        self._dict.pop(flsettings.id)
        return self.effective != x

    def add(self, flsettings: FeedlineSetup):
        self._dict[flsettings.id] = flsettings


class FeedlineHardware:
    def __init__(self, bitstream, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False, download=True, start_clock=True):
        self._clock_source = validate(clock_source=clock_source, error=True)
        self._ol = pynq.Overlay(bitstream, download=download, ignore_version=ignore_version)
        self._if_board = IFBoard(if_port, connect=False)
        self._ignore_version = ignore_version
        if start_clock:
            import mkidgen3.drivers.rfdc
            mkidgen3.drivers.rfdc.start_clocks(external_10mhz=clock_source == 'external_10mhz')

    def reset(self):
        self._if_board.power_off(save_settings=False)
        mkidgen3.drivers.rfdc.start_clocks(external_10mhz=self._clock_source == 'external_10mhz')
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


class FeedlineReadout:
    def __init__(self, bitstream, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False):
        self.hardware = FeedlineHardware(bitstream, clock_source=clock_source, if_port=if_port,
                                         ignore_version=ignore_version, download=True, start_clock=True)

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
    def plram_cap(cr, ol: pynq.Overlay, context=None):
        """

        Args:
            context:
            cr: A CaptureRequest object
            ol: A pynq.Overlay with the firmware bitstream loaded, assumed to be thread safe

        Returns: None

        """

        # TODO these are fatal errors and the function should never have been called.
        # CR should be aborted though
        failmsg = ''
        try:
            assert cr.type == 'engineering', 'Incorrect capture request type'
            assert ol.capture.ready(), 'Capture Subsystem is busy'
        except AssertionError as e:
            failmsg = str(e)

        try:
            abort = context.socket(
                zmq.SUB)  # TODO Use of subscribe will probably result in missed abort mesages, REQ/REP?
            abort.setsockopt(zmq.SUBSCRIBE, id)
            abort.connect('inproc://cap_abort')
        except zmq.EFAULT:
            failmsg = f"Unable to establish abort socket {cr.id}, dropping request."

        try:
            cr.establish(context)
        except zmq.EFAULT:
            failmsg = f"Unable to establish capture {cr.id}, dropping request."

        if failmsg:
            getLogger(__name__).error(failmsg)
            try:
                cr.fail(failmsg)
                cr.destablish()
            except zmq.EFAULT as ez:
                getLogger(__name__).warning(f'Failed to send abort for {cr} due to {ez}')
            return

        try:
            nchunks = cr.size // CHUNKING_THRESHOLD
            partial = cr.size - CHUNKING_THRESHOLD * nchunks
            chunks = [CHUNKING_THRESHOLD] * nchunks
            if partial:
                chunks.appned(partial)
            for i, csize in enumerate(chunks):
                if abort.poll(1) != 0:
                    raise CaptureAbortedException
                data = ol.capture(csize, tap=cr.tap, wait=True)
                cr.add_data(data, status=f'Captured {i} of {len(chunks)}')
                data.free_buffer()
            cr.finish()
        except CaptureAbortedException:
            cr.fail(f'Aborted')
        except Exception as e:
            getLogger(__name__).error(f'Terminating capture {id} due to {e}')
            try:
                cr.fail(f'Aborted due to {e}')
            except zmq.EFAULT as ez:
                getLogger(__name__).warning(f'Failed to send abort message {cr} due to {ez}')
        finally:
            del cr
            abort.close()

    @staticmethod
    def photon_cap(context, cr, ol):
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

    def main(self, pipe: zmq.Socket, context: zmq.Context = None):
        """
        Enqueue a list of capture requests for future handling. Invalid requests are dealt with immediately and not
        enqueued.

        Args:
            zpipe: a pipe for receiving capture requests
            conext: zmq.context

        Returns: None

        """
        context = context or zmq.Context().instance()

        class TapThread:
            def __init__(self, thread, pipe, request):
                self.thread = thread
                self.request = request
                self.pipe = pipe

        tap_threads = {k: None for k in ('photon', 'stamp', 'engineering')}
        active_settings = None
        settings_set = FLSettingsSet()  # a set of FeedlineStettings

        to_check = []
        checked = []
        while True:
            # Check to see if any capture threads have finished
            complete = [k for k, t in tap_threads.items() if t is not None and not t.thread.is_alive()]
            # for each finished capture thread remove its settings from the requirements pot and cleanup

            if bool(complete):  # need to check up to the size of the queue if anything finished
                to_check.extend(checked)
                checked = []

            for k in complete:
                settings_set.pop(tap_threads[k].request.settings)
                tap_threads[k] = None

            running_by_id = {tt.request.id: tt for tt in tap_threads.values() if tt is not None}

            # check for any incoming info: CapRequest, ABORT id|all, EXIT
            try:
                cmd, null, data = pipe.recv(zmq.NOBLOCK)
                assert null == b''
            except zmq.EAGAIN:
                cmd = ''
                data = ''

            if cmd not in ('exit', 'abort', 'capture'):
                pipe.send(f'ERROR: Invalid command "{cmd}"')
                cmd = ''
                data = ''

            if 'cmd' == 'exit':
                cmd = 'abort'
                data = 'all'

            if cmd == 'abort':
                if data == 'all':
                    for cr in checked + to_check:
                        cr.set_status('aborted')  # signal that captures will never happen
                    checked = []
                    to_check = []
                    for v in running_by_id.values():  # stop any running tap threads
                        try:
                            v.pipe.send('abort')  # TODO what happens to pipe when thread ends?
                        except zmq.EFAULT:
                            getLogger(__name__).critical('Error sending abort to worker thread. Exiting')
                            raise
                else:
                    aborted = False
                    if data in running_by_id:
                        aborted = True
                        running_by_id[data].pipe.send('abort')  # TODO what happens to pipe when thread ends?
                    for cr in filter(lambda x: x.id == data, checked):
                        aborted = True
                        checked.pop(checked.index(cr))
                        cr.set_status('aborted')
                    for cr in filter(lambda x: x.id == data, to_check):
                        aborted = True
                        to_check.pop(to_check.index(cr))
                        cr.set_status('aborted')

                    if not aborted:
                        getLogger(__name__).info(f'Capture request {data} is unknown and can not be aborted.')

            cr = None  # CR is the capture request that will be ckicked off this iteration of the loop
            if cmd == 'capture':
                if (not to_check and data.type not in tap_threads and
                        settings_set.effective().compatible_with(data.feedline_setup)):
                    cr = data  # this can be run and nothing else, so it will be done below
                else:
                    q = to_check if to_check else checked
                    try:
                        data.set_status('queued', f'Queued')
                        q.append(data)
                    except zmq.EFAULT:
                        getLogger(__name__).error(f'Unable to update status. Aborted request {data.id}')

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
                if cr.type in tap_threads:
                    cr.set_status('queued', f'tap location in use by: {tap_threads[cr.type].request.id}')
                    checked.append(cr)
                    continue
                elif not settings_set.effective().compatible_with(cr.feedline_setup):
                    cr.set_status('queued', f'incompatible with one or more of: {running_by_id.keys()}')
                    checked.append(cr)
                    continue
            except zmq.EFAULT:
                getLogger(__name__).error(f'Unable to update status. Aborted request {cr.id}')
                continue

            changed_settings = settings_set.add(cr.id, cr.feedline_setup)
            self.hardware.apply_settings(changed_settings)

            cap_runners = {'engineering': self.plram_cap, 'photon': self.photon_cap, 'stamp': self.stamp_cap}
            target = cap_runners[cr.type]
            a, b = zpipe(context)
            cr.set_status('running', f'Started at UTC {datetime.utcnow()}')
            cr.destablish()
            t = threading.Thread(target=target, name=f"CapThread: {cr.id}", args=(context, b, cr, self._ol))
            t.start()
            tap_threads[cr.type] = TapThread(t, a, cr)


def parse_cl():
    parser = argparse.ArgumentParser(description='Feedline Readout Server', add_help=True)
    parser.add_argument('-p', '--port', dest='port', action='store', required=False, type=int,
                        help='Server port', default='8888')
    parser.add_argument('--cap_port', dest='cap_port', action='store', required=False, type=int,
                        help='Capture Data Port', default='8889')
    parser.add_argument('--clock', dest='clock', action='store', required=False, type=str,
                        help='Clock Source', default='external_10mhz')
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
    args = parse_cl()

    fr = FeedlineReadout(args.bitstream, clock_source=args.clock, if_port=args.if_board,
                         ignore_version=args.ignore_fpga_driver_version)

    capture_port = args.capture_port
    command_port = args.port
    from zmq.devices import ProcessDevice

    # Set up a proxy for routing all the capture requests
    pd = zmq.devices.ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
    pd.bind_in('inproc://cap_data')
    pd.bind_out(f'tcp://*:{capture_port}')
    pd.setsockopt_in(zmq.XPUB_VERBOSE, 'ROUTER')
    pd.start()

    # Set up a command port
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{command_port}")

    cap_pipe, cap_pipe_thread = zpipe(zmq.Context.instance())

    main = threading.Thread(target=fr.main, args=(cap_pipe_thread,), kwargs={'context': context})
    main.daemon = False
    main.start()

    while True:
        cmd, args = socket.recv_multipart()

        if cmd == 'reset':
            cap_pipe.send(['abort', 'all'])
            fr.hardware.reset()
            socket.send_json('OK')

        elif cmd == 'status':
            status = fr.status()  # this might take a while and fail
            status['id'] = f'FRS {args.fl_id} @ {args.port}/{args.cap_port}'
            socket.send_json(status)
        elif cmd == 'bequiet':
            cap_pipe.send(['abort', 'all'])
            fr.hardware.bequiet(**json.loads(args))  # This might take a while and fail
            socket.send_json('OK')
        elif cmd == 'capture':
            # determine if capture is possible
            # determine if capture compatible with current actions
            # fire function to apply necessary pl settings and start thread to deal with capture
            cr = args
            needed_settings = cr.settings
            cap_pipe.send_multipart(['capture'] + args)
            socket.send_json(cap_pipe.recv())
