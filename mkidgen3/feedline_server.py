from .drivers.ifboard import IFBoard
from .schema import validate
from .status_keeper import StatusKeeper
from logging import getLogger
import pynq
import mkidgen3.drivers.rfdc
from objects import CaptureRequest, CaptureAbortedException, FeedlineSetup
from typing import List
import zmq
import blosc
import time
import threading
from datetime import datetime

COMMAND_LIST = ('reset', 'capture', 'bequiet', 'status')

CHUNKING_THRESHOLD = 1000


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


class FeedlineReadout:
    def __init__(self, name, bitstream, port=8000, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False, status_port=None):
        self._name = name
        self._bitstream = bitstream
        self._port = port
        self._ignore_version = ignore_version
        self._clock_source = validate(clock_source=clock_source, error=True)
        self._ol = None
        self._if_board = IFBoard(if_port, connect=False)
        # self.status_keeper = StatusKeeper(status_port)  #TODO

    @property
    def id(self):
        return f"FRS {self._name} @ {self._port}"

    def reset(self):
        """
        Reset the system via powering off the IF board, (re)starting the clocks, and (re)downloading the
        bitstreeam.

        Returns: None

        """
        self._if_board.power_off(save_settings=False)
        mkidgen3.drivers.rfdc.start_clocks(external_10mhz=self._clock_source == 'external_10mhz')
        self._ol = pynq.Overlay(self._bitstream, ignore_version=self._ignore_version, download=True)
        self._if_board.power_on()
        # self.status_keeper.update(self.id, **self.status()) #TODO

    def status(self):
        """

        Returns: Dictionary of status information

        """
        if self._ol is None:
            ol_status = FeedlineStatus(bitstream=None)
        else:
            ol_status = FeedlineStatus(dac=DACStatus.from_core(self._ol.dac_table),
                                       ddc=DDCStatus.from_core(self._ol.photon_pipe.reschan.ddc),
                                       )

        status = {'name': self._name,
                  'id': self.id,
                  'fpga': ol_status,
                  'if_board': self._if_board.status(),
                  'running_captures': self._running_captures(),
                  'pending_captures': self._pending_captures()}

        return status

    def bequiet(self, stop_dacs=True, poweroff_if=False):
        """

        Args:
            stop_dacs: Stop the DACs from replaying any values
            stop_if: Stop the PLLs on the IF board
            poweroff_if: Power down the IF board (implies `stop_if`)

        Returns: None
        """
        if stop_dacs:
            self._ol.dac_table.quiet()
            # self.status_keeper.update(self.id, dacs=self._ol.dac_table.status())
        if poweroff_if:
            self._if_board.power_off(save_settings=False)
            # self.status_keeper.update(self.id, if_board=self.if_board.status())

    @staticmethod
    def plram_cap(context, cr, ol):
        try:
            cr.establish(context)
        except Exception:
            getLogger(__name__).error(f"Unable to establish capture {cr.id}, dropping request.")
            return
        abort = context.socket(zmq.SUB)
        abort.setsockopt(zmq.SUBSCRIBE, id)
        abort.connect('inproc://cap_abort')
        try:
            nchunks = cr.size // CHUNKING_THRESHOLD
            partial = cr.size - CHUNKING_THRESHOLD * nchunks
            chunks = [CHUNKING_THRESHOLD] * nchunks
            if partial:
                chunks.appned(partial)
            for i, csize in enumerate(chunks):
                if cr.aborted() or abort.poll(1) != 0:
                    raise CaptureAbortedException
                data = ol.capture(csize, tap=cr.tap, wait=True)
                cr.add_data(data, status=f'Captured {i} of {len(chunks)}')
            cr.finish()
        except CaptureAbortedException:
            cr.fail(f'Aborted.')
        except Exception as e:
            getLogger(__name__).error(f'Terminating capture {id} due to {e}')
            cr.fail(f'Aborted due to {str(e)}')
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
            buffers = [PhotonBuffer() for _ in range(2)]
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
            for b in buffers:
                b.buffer.freebuffer()
            abort.close()

    def _capture_main(self, zpipe: zmq.Socket, context: zmq.Context = None, CHUNKING_THRESHOLD=1024 ** 2 * 100):
        """
        Enqueue a list of capture requests for future handling. Invalid requests are dealt with immediately and not
        enqueued.

        Args:
            zpipe: a pipe for receiving capture requests
            capreqs: one or more CaptureRequest objects

        Returns: None

        """
        context = context or zmq.Context().instance()

        requests = context.socket(zmq.REP)
        requests.connect('inproc://cap_request')

        class TapThread:
            def __init__(self, thread, pipe, request):
                self.thread = thread
                self.request = request
                self.pipe = pipe

        tap_threads = {k: None for k in ('photon', 'stamp', 'engineering')}
        active_settings = None
        settings_set = FLSettingsSet()  # a set of FeedlineStettings
        import queue
        crqueue = queue.Queue()

        check_queue = False
        while True:
            # Check to see if any capture threads have finished
            complete = [k for k, t in tap_threads.items() if t is not None and not t.thread.is_alive()]
            # for each finished capture thread remove its settings from the requirements pot and cleanup

            for k in complete:
                check_queue |= settings_set.pop(tap_threads[k].request.settings)
                tap_threads[k] = None

            running_by_id = {tt.request.id: tt for tt in tap_threads.values() if tt is not None}

            # check for any incoming info: CapRequest, ABORT id|all, EXIT
            try:
                cmd, null, data = zpipe.recv(zmq.NOBLOCK)
                assert null == b''
            except zmq.EAGAIN:
                cmd = ''
                data = ''

            if 'cmd' == 'exit':
                cmd = 'abort'
                data = 'all'

            if cmd == 'abort':
                if data == 'all':
                    while crqueue:
                        cr = crqueue.pop()
                        cr.abort()
                    for v in running_by_id.values():
                        v.pipe.send('abort')  # TODO what happens to pipe when thread ends?
                else:
                    if id in running_by_id:
                        running_by_id[id].pipe.send('abort')  # TODO what happens to pipe when thread ends?
                    elif data in crqueue:
                        cr = crqueue.pop(data)
                        cr.abort()
                    else:
                        getLogger(__name__).info(f'Capture request {data} is unknown and can not be aborted.')

            if cmd == 'capture':
                cr = data

            if (not check_queue and data.type not in tap_threads and
                    settings_set.effective().compatible_with(cr.feedline_setup)):
                # this can be run and nothing else, so do it
                pass
            else:
                crqueue.push(cr)
                # cant be run because there might be something more important (we check anyway),
                # the tap is in use (we check when the tap finishes)
                # settings arent compatible (we will check when something finishes)
                # So don't touch check_queue
                cr = None

            if check_queue and crqueue:
                cr = crqueue.pop()

            if not cr:
                continue

            assert isinstance(cr, CaptureRequest)

            if cr.type in tap_threads:
                cr.update_status('queued', f'tap location in use by: {tap_threads[cr.type].request.id}')
                crqueue.push(cr)
                continue
            elif settings_set.compatible_with(cr.feedline_setup):
                cr.update_status('queued', f'incompatible with one or more of: {running_by_id.keys()}')
                crqueue.push(cr)
                continue

            settings_set.add(cr.id, cr.feedline_setup)
            apply_fl_settings(cr.feedline_setup, self._ol)

            cap_runners = {'engineering': self.plram_cap, 'photon': self.photon_cap, 'stamp': self.stamp_cap}
            target = cap_runners[cr.type]
            a, b = zpipe(context)
            cr.set_status('running', f'Started at UTC {datetime.utcnow()}')
            t = threading.Thread(target=target, name=f"CapThread: {cr.id}", args=(context, b, cr, self._ol))
            t.start()
            tap_threads[cr.type] = TapThread(t, a, cr)

    def main(self, context: zmq.Context = None):

        context = context or zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self._port}")

        while True:
            cmd, args = socket.recv_multipart()

            if cmd == 'reset':
                # TODO abort all captures
                self.reset()  # This might take a while and fail
            elif cmd == 'status':
                status = self.status()  # this might take a while and fail
            elif cmd == 'bequiet':
                # TODO should we abort all captures
                # TODO extract bequiet kwargs from args and pass
                self.bequiet()  # This might take a while and fail
            elif cmd == 'capture':
                # determine if capture is possible
                # determine if capture compatible with current actions
                # fire function to apply necessary pl settings and start thread to deal with capture
                cr = args
                needed_settings = cr.settings

                self.active_capture_taps()
                active_taps = tuple(t for t, v in tap_threads.items() if v is not None)

                request_blocked = needed_settings.compatible_with(active_settings) or cr.tap in active_taps

                if request_blocked:
                    try:
                        cr.set_status('submitted', 'Request not compatible with running captures. '
                                                   f'Resubmitting to queue at {datetime.utcnow()}', context=context)
                    except Exception:
                        getLogger(__name__).error(f"Unable to update capture status for {cr.id}, dropping request.")
                        continue

                    try:
                        request_resubmit.send_pyobj(cr)
                    except Exception:
                        cr.fail('Resubmission failed.', context=context)
                        getLogger(__name__).error(f"Resubmission failed for {cr.id}, dropping request.")

                else:

                    active_settings = g3.apply_fl_settings(cr.settings)
                    if cr.tap in ('iq', 'phase', 'adc'):
                        target = self.plram_cap
                    elif cr.tap == 'photon':
                        target = self.photon_cap
                    else:
                        target = self.stamp_cap
                    t = threading.Thread(target=target, name=f"CapThread: {cr.id}", args=(context, cr, self._ol))
                    tap_threads[cr.tap] = t
                    t.start()
            # TODO sort out how to reply with the result of the command


"""
Test proceedure:
1) Start FRS
"""

import argparse


def parse_cl():
    parser = argparse.ArgumentParser(description='Feedline Readout Server', add_help=True)
    parser.add_argument('-d', '--dir', dest='dir',
                        action='store', required=False, type=str,
                        help='source dir for files', default='./')
    parser.add_argument('--cr', dest='do_cosmic', default=False,
                        action='store_true', required=False,
                        help='Do cosmic ray rejection')
    parser.add_argument('--num', dest='num', default='',
                        type=str, required=False,
                        help='#,#,... files to do. Default to all')
    parser.add_argument('-o', dest='outdir', default='./out/',
                        action='store', required=False, type=str,
                        help='Out directory')
    parser.add_argument('--sigclip', dest='sigclip', default=15.0,
                        action='store', required=False, type=float,
                        help='CR Sigmaclip (sigma limit for flagging as CR)')
    parser.add_argument('--sigfrac', dest='sigfrac', default=0.3,
                        action='store', required=False, type=float,
                        help='CR Sigmafrac (sigclip fract for neighboring pix)')
    parser.add_argument('--objlim', dest='objlim', default=1.4,
                        action='store', required=False, type=float,
                        help='CR Object Limit (raise if normal data clipped)')
    parser.add_argument('--criter', dest='criter', default=5,
                        action='store', required=False, type=int,
                        help='CR Iteration Limit')
    parser.add_argument('--overwrite', dest='overwrite', default=False,
                        action='store_true', required=False,
                        help='overwrite existing output')
    return parser.parse_args()


import binascii
import os


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


if __name__ == '__main__':
    args = parse_cl()

    fr = FeedlineReadout(args.fl_id, args.bitstream, port=args.command_port,
                         clock_source=args.clock, if_port=args.if_board,
                         ignore_version=args.ignore_fpga_driver_version, status_port=None)

    context = zmq.Context()
    server = context.socket(zmq.XSUB)
    server.bind(f'tcp:\\*:{args.cap_port}')
    captures = context.socket(zmq.XPUB)
    captures.bind(f'inproc:\\captures')

    threading.Thread(target=server.main, args=(context,))
    zmq.proxy(server, captures)
