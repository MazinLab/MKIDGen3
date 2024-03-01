from mkidgen3.rfsocmemory import determine_max_chunk
from logging import getLogger
from mkidgen3.server.feedline_config import FeedlineConfig, FeedlineConfigManager
from mkidgen3.server.captures import CaptureRequest
from mkidgen3.util import check_zmq_abort_pipe, AbortedException, format_bytes, format_time

import zmq
from mkidgen3.server.misc import zpipe
import threading
import time
import numpy as np
from mkidgen3.mkidpynq import unpack_photons

import queue
PHOTON_PACKED_DTYPE = np.uint64


class FeedlineHardware:
    def __init__(self):
        self.config_manager = FeedlineConfigManager()

    def config_compatible_with(self, config: FeedlineConfig):
        return self.config_manager.required().compatible_with(config)

    def derequire_config(self, id):
        """True iff the required settings changed as a result"""
        try:
            return self.config_manager.pop(id)
        except KeyError:
            return False

    def apply_config(self, id, config: FeedlineConfig):
        fl_setup = self.config_manager.add(id, config)

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
        except AssertionError as e:
            failmsg = str(e)
        except AttributeError as e:
            failmsg = f'Something is fundamentally wrong. Check your bitstream. ({str(e)})'

        try:
            cr.establish(context=context)
        except zmq.ZMQError as e:
            failmsg += f"Unable to establish capture {cr.id} due to {e}, dropping request."

        if failmsg:
            getLogger(__name__).error(failmsg)
            cr.fail(failmsg, raise_exception=False)
            try:
                pipe.close()
            except zmq.ZMQError:
                pass
            return

        hw_channels = tuple(sorted(cr.channels if cr.channels else range(2048)))

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
                                          f'requires {format_bytes(hw_size_bytes)} \n'
                                          f'Requested channel subset of hardware capture '
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

                if 'adc' in cr.tap:
                    shape = (csize, 2)
                    datatime = csize/4e9
                elif 'iq' in cr.tap:
                    shape = (csize, len(hw_channels), 2)
                    datatime = csize / 2e6
                else:
                    shape = (csize, len(hw_channels), 1)
                    datatime = csize / 1e6

                times.append(time.perf_counter())
                tic = time.perf_counter()
                data = np.random.randint(-32000, 32000, size=shape, dtype=np.int16)
                lefttime=datatime-(time.perf_counter()-tic)
                if lefttime<0:
                    getLogger(__name__).debug(f'numpy random data too extra time {-lefttime}')
                time.sleep(max(lefttime,0))
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
        try:
            pipe.close()
        except zmq.ZMQError:
            pass

    def photon_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context=None):
        """
        pipe: a zmq pair pipe to detect abort
        cr: the capture request
        """
        failmsg = ''
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
        # q = q_other = queue.Queue()  #an alternative

        def garbage_fountain(q: (zmq.Socket, queue.SimpleQueue), kill=None, copy_buffer=False, spawn=True,
                             discard_initial=3):
            if spawn:
                kill = threading.Event()
                _fountain_thread = threading.Thread(target=garbage_fountain, name='photon fountain',
                                                         args=(q,),
                                                         kwargs=dict(kill=kill, spawn=False, copy_buffer=copy_buffer,
                                                                     discard_initial=discard_initial))
                return _fountain_thread, kill

            assert isinstance(kill, threading.Event)

            log = getLogger(__name__)
            sender = q.put if hasattr(q, 'put') else lambda x: q.send(x, copy=False)

            while not kill.is_set():

                if discard_initial > 0:
                    log.debug(f'Discarding {discard_initial} before sending')
                    discard_initial -= 1
                    continue
                tic = time.perf_counter()
                x = np.random.randint(2**64-1, size=int(cr.nsamp*5000/1000*2048), dtype=PHOTON_PACKED_DTYPE)
                time.sleep(cr.nsamp/1000 - (time.perf_counter()-tic))
                sender(x)

            sender(b'')
            if isinstance(q, zmq.Socket):
                q.close()

        fountain, stop = garbage_fountain(q_other, spawn=True, copy_buffer=True)

        def photon_sender(q: zmq.Socket, cr, unpack=False):
            log = getLogger(__name__)
            from datetime import datetime
            toc = 0
            delts = []
            try:
                cr.establish(context=context)
                while True:
                    tic = time.perf_counter()
                    if toc:
                        delt = tic - toc
                        delts.append(delt)
                        x = tic.strftime('%M:%S.%f')
                        log.debug(f'Prep send @ {tic}, since last wait ended {delt:.06} s')

                    x = q.recv()
                    if x == b'':
                        cr.finish()
                        break
                    photons = np.frombuffer(x, dtype=PHOTON_PACKED_DTYPE)
                    toc = time.perf_counter()

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
            while not cr.completed:
                check_zmq_abort_pipe(pipe)
        except AbortedException as e:
            getLogger(__name__).error(f'Aborting photon capture {cr} due user request.')
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
        finally:
            stop.set()
            fountain.join()
            sender.join()
            pipe.close()
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
            if isinstance(q, zmq.Socket):
                q.close()

    def postage_cap(self, pipe: zmq.Socket, cr: CaptureRequest, context: zmq.Context = None):
        failmsg = ''
        try:
            assert cr.type == 'postage', 'Incorrect capture request type'
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

        maxtime = (8000 / 8 * 128 / 1e6) / 10
        try:
            tic = time.perf_counter()
            data = np.random.randint(-32000, 32000, size=(8000, 128, 2), dtype=np.int16)
            time.sleep(maxtime - (time.perf_counter() - tic))
            cr.send_data(data, copy=False, compress=True)
            cr.finish()
        except AbortedException as e:
            cr.abort(e, raise_exception=False)
        except Exception as e:
            getLogger(__name__).error(f'Terminating {cr} due to {e}')
            cr.fail(f'Failed due to {e}', raise_exception=False)
        finally:
            getLogger(__name__).debug(f'Deleting {cr} as all work is complete')
            del cr
            pipe.close()
