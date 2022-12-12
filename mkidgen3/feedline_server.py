from .drivers.ifboard import IFBoard
from .schema import validate
from .status_keeper import StatusKeeper
from logging import getLogger
import pynq
import mkidgen3.drivers.rfdc
from objects import CaptureRequest, CaptureAbortedException
from typing import List
import zmq
import blosc
import time
import threading
from datetime import datetime

COMMAND_LIST = ('reset', 'capture', 'bequiet', 'status')

CHUNKING_THRESHOLD = 1000


class FeedlineReadoutServer:
    def __init__(self, name, bitstream, port=8000, clock_source="external_10mhz", if_port='dev/ifboard',
                 ignore_version=False, status_port=None):
        self._name = name
        self._bitstream = bitstream
        self._port = port
        self._ignore_version = ignore_version
        self._clock_source = validate(clock_source=clock_source, error=True)
        self._ol = None
        self.if_board = IFBoard(if_port, connect=False)
        self.status_keeper = StatusKeeper(status_port)

    @property
    def id(self):
        return f"FRS {self._name} @ {self._port}: {self._bitstream} clk={self._clock_source}"

    def reset(self):
        """
        Reset the system via powering off the IF board, (re)starting the clocks, and (re)downloading the
        bitstreeam.

        Returns: None

        """
        self.if_board.power_off(save_settings=False)
        mkidgen3.drivers.rfdc.start_clocks(external_10mhz=self._clock_source=='external_10mhz')
        self._ol = pynq.Overlay(self._bitstream, ignore_version=self._ignore_version, download=True)
        self.if_board.power_on()
        self.status_keeper.update(self.id, **self.status())
        return 'Reset Complete'

    def status(self):
        """

        Returns: Dictionary of status information

        """
        if self._ol is None:
            ol_status = {'dac': NoBitStream()}
        else:
            ol_status = {'dac': self._ol.dac_table.status(),}

        status = {'name': self._name, 'id': self._id,
                  'running': self._running_captures(),
                  'pending': self._pending_captures()}
        status.update(ol_status)
        status.update({'if_board': self.if_board.status()})
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
            self.status_keeper.update(self.id, dacs=self._ol.dac_table.status())
        if poweroff_if:
            self.if_board.power_off(save_settings=False)
            self.status_keeper.update(self.id, if_board=self.if_board.status())


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
                if cr.aborted() or abort.poll(1)!=0:
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
            while not cr.aborted() and abort.poll(1)==0:
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

    def _capture_main(self, context: zmq.Context, CHUNKING_THRESHOLD=1024**2*100):
        """
        Enqueue a list of capture requests for future handling. Invalid requests are dealt with immediately and not
        enqueued.

        Args:
            capreqs: one or more CaptureRequest objects

        Returns: None

        """
        requests = context.socket(zmq.REP)
        requests.connect('inproc://cap_request')

        request_resubmit = context.socket(zmq.REQ)

        tap_threads = {k:None for k in ('photon','stamp', 'iq', 'phase', 'adc') }
        active_settings = None

        try:
            while True:
                cr = requests.recv_pyobj()
                assert isinstance(cr, CaptureRequest)
                needed_settings = cr.settings

                active_taps = tuple(t for t,v in tap_threads.items() if v is not None)

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

                    active_settings = apply_fl_settings(cr.settings)
                    if cr.tap in ('iq', 'phase', 'adc'):
                        target=self.plram_cap
                    elif cr.tap == 'photon':
                        target = self.photon_cap
                    else:
                        target = self.stamp_cap
                    t = threading.Thread(target=target, name=f"CapThread: {cr.id}", args=(context, cr, self._ol))
                    tap_threads[cr.tap] = t
                    t.start()

        except Exception as e:
            getLogger(__name__).error(e)
        finally:
            requests.close()


    def _server_main(self):
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind(f"tcp://*:{self._port}")

            self._capcontext.socket(zmq.REP)
            socket.bind(f"tcp://*:{self._port}")
        except:

        while True:
            try:
                cmd = socket.recv_json()

            xmitThread = None
            capture = None
            last = None
            pending_captures = []
            run = True
            while run:
                if command_available(socket):
                    do(socket, ol)
                if capture and capture.complete and (last is None or last.sent()):
                    del last
                    last = capture
                    capture.send(socket)
                    capture = None
                if not capture and pending_captures:
                    try:
                        capture = pending_captures.pop(0)
                        capture.start()
                    except IndexError:
                        pass
                time.sleep(.01)

    async def run(self):
