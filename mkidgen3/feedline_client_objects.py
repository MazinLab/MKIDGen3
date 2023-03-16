import threading
import time
import numpy as np
import zmq
import blosc2

from .funcs import SYSTEM_BANDWIDTH, compute_lo_steps
import os
from logging import getLogger
from hashlib import md5
from .feedline_objects import zpipe, FeedlineConfig, CaptureRequest
from typing import List

class CaptureSink(threading.Thread):
    def __init__(self, request, source, context=None, start=True):
        id = request.id.decode()
        super(CaptureSink, self).__init__(name=f'cap_id={id}')
        self._expected_buffer_shape = request.buffer_shape
        self.daemon = True
        self.cap_id = id
        self.data_source = source
        self.result = None
        self._ctx = context or zmq.Context.instance()
        self._pipe = zpipe(self._ctx)
        self._buf = None
        if start:
            self.start()

    def kill(self):
        self._pipe[0].send(b'')
        self._pipe[0].close()

    def _accumulate_data(self, d):
        if self._buf is None:
            self._buf = []
        self._buf.append(blosc2.decompress(d))

    def _finish_accumulation(self):
        self._buf = b''.join(self._buf)

    def _finalize_data(self):
        self.result = np.frombuffer(self._buf, dtype=np.int16)

    def run(self):
        try:
            with self._ctx.socket(zmq.SUB) as sock:
                sock.setsockopt(zmq.SUBSCRIBE, self.cap_id)
                sock.connect(self.data_source)

                poller = zmq.Poller()
                poller.register(self._pipe[1], flags=zmq.POLLIN)
                poller.register(sock, flags=zmq.POLLIN)

                getLogger(__name__).debug(f'Listening for data for {self.cap_id}')
                self._pipe[1].send(b'')
                while True:
                    avail = dict(poller.poll())
                    if self._pipe[1] in avail:
                        getLogger(__name__).debug(f'Received shutdown order, terminating data acq. of {self}')
                        break
                    elif sock not in avail:
                        time.sleep(.1)  # play nice
                        continue
                    id, data = sock.recv_multipart(copy=False)
                    if not data:
                        getLogger(__name__).debug(f'Received null, capture data stream over')
                        break
                    getLogger(__name__).debug(f'Received data snippet for {self}')
                    self._accumulate_data(data)
                self._finish_accumulation()
                self._finalize_data()
                getLogger(__name__).info(f'Capture data for {self.cap_id} processed into {self.result.shape} '
                                         f'{self.result.dtype}: {self.result}')
        except zmq.ZMQError as e:
            getLogger(__name__).warning(f'Shutting down {self} due to {e}')
        finally:
            self._pipe[1].close()

    def data(self):
        self.join()
        return self.result

    def ready(self):
        self._pipe[0].recv()  # TODO make this line block


class StreamCaptureSink(CaptureSink):
    def _finalize_data(self):
        # raw adc data is i0q0 i1q1 int16
        size = len(self._buf)/2  # n int16
        n = size//np.prod(self._expected_buffer_shape[1:])
        shape = (min(n, self._expected_buffer_shape[0]),) + self._expected_buffer_shape[1:]
        if n > self._expected_buffer_shape[0]:
            getLogger(__name__).warning(f'Received more data than expected for {self.cap_id}')
        elif n < self._expected_buffer_shape[0]:
            getLogger(__name__).warning(f'Finalizing incomplete capture data for {self.cap_id}')
        #TODO this would technically be a memory leak if we captured too much data
        self.result = np.frombuffer(self._buf, count=np.prod(shape, dtype=int), dtype=np.int16).reshape(shape).squeeze()


class ADCCaptureSink(StreamCaptureSink):
    pass


class IQCaptureSink(StreamCaptureSink):
    pass


class PhaseCaptureSink(StreamCaptureSink):
    pass


class PhotonCaptureSink(CaptureSink):
    def __init__(self, source, context: zmq.Context = None):
        pass

    def capture(self, hdf, xymap, feedline_source, fl_ids):
        t = threading.Thread(target=self._main, args=(hdf, xymap, feedline_source, fl_ids))
        t.start()

    @staticmethod
    def _main(hdf, xymap, feedline_source, fl_ids, term_source='inproc://PhotonCaptureSink.terminator.inproc'):
        """

        Args:
            xymap: [nfeedline, npixel, 2] array
            feedline_source: zmq.PUB socket with photonbufers published by feedline
            term_source: a zmq socket of undecided type for detecting shutdown requests

        Returns: None

        """

        fl_npix = 2048
        n_fl = 5
        MAX_NEW_PHOTONS = 5000
        DETECTOR_SHAPE = (128, 80)
        fl_id_to_index = np.arange(n_fl, dtype=int)

        context = zmq.Context.instance()
        term = context.socket(zmq.SUB)
        term.setsockopt(zmq.SUBSCRIBE, id)
        term.connect(term_source)

        data = context.socket(zmq.SUB)
        data.setsockopt(zmq.SUBSCRIBE, fl_ids)
        data.connect(feedline_source)

        poller = zmq.Poller()
        poller.register(term, flags=zmq.POLLIN)
        poller.register(data, flags=zmq.POLLIN)

        live_image = np.zeros(DETECTOR_SHAPE)
        live_image_socket = None
        live_image_by_fl = live_image.reshape(n_fl, fl_npix)
        photons_rabuf = np.recarray(MAX_NEW_PHOTONS,
                                    dtype=(('time', 'u32'), ('x', 'u32'), ('y', 'u32'),
                                           ('phase', 'u16')))

        while True:
            avail = poller.poll()
            if term in avail:
                break

            frame = data.recv_multipart(copy=False)
            fl_id = frame[0]
            time_offset = frame[1]
            d = blosc2.decompress(frame[1])
            frame_duration = None  # todo time coverage of data
            # buffer is nchan*nmax+1 32bit: 16bit time(base2) 16bit phase
            # make array of to [nnmax+1, nchan, 2] uint16
            # nmax will always be <<2^12 number valid will be at [0,:,0]
            # times need oring w offset
            # photon data is d[1:d[0,i,0], i, :]

            nnew = d[0, :, 0].sum()
            # if we wanted to save binary data then we could save this, the x,y list, and the time offset
            # mean pixel count rate in this packet is simply [0,:,0]/dt
            fl_ndx = fl_id_to_index[fl_id]
            live_image_by_fl[fl_ndx, :] += d[0, :, 0] / frame_duration

            # if live_image_ready
            live_image_socket.send_multipart([f'liveim', blosc2.compress(live_image)])

            cphot = np.cumsum(d[0, :, 0], dtype=int)
            for i in range(fl_npix):
                sl_out = slice(cphot[i], cphot[i] + d[0, i, 0])
                sl_in = slice(1, d[0, i, 0])
                photons_rabuf['time'][sl_out] = d[sl_in, :, 0]
                photons_rabuf['time'][sl_out] |= time_offset
                photons_rabuf['phase'][sl_out] = d[sl_in, :, 1]
                photons_rabuf['x'][sl_out] = xymap[fl_ndx, i, 0]
                photons_rabuf['y'][sl_out] = xymap[fl_ndx, i, 1]
            hdf.grow_by(photons_rabuf[:nnew])

        term.close()
        data.close()
        hdf.close()


class PostageCaptureSink(CaptureSink):
    def __init__(self, source, context: zmq.Context = None):
        pass


def CaptureSinkFactory(request, server, start=True) -> ((zmq.Socket, zmq.Socket), CaptureSink):
    if request.tap == 'adc':
        saver = ADCCaptureSink(request.id, server, start=start)
    elif request.tap == 'iq':
        saver = IQCaptureSink(request.id, server, start=start)
    elif request.tap == 'phase':
        saver = PhaseCaptureSink(request.id, server, start=start)
    elif request.tap == 'photon':
        saver = PhotonCaptureSink(request.id, server)
    elif request.tap == 'postage':
        saver = PostageCaptureSink(request.id, server)

    else:
        raise ValueError(f'Malformed CaptureRequest {request}')
    return saver


class StatusListner(threading.Thread):
    def __init__(self, id, source, initial_state='Created', start=True):
        super().__init__(name=f'StautsListner_{id}')
        self.daemon = True
        self.source = source
        self._pipe = zpipe(zmq.Context.instance())
        self.id = id
        self._status_messages = [initial_state]
        if start:
            self.start()

    def kill(self):
        self._pipe[0].send(b'')
        self._pipe[0].close()

    @staticmethod
    def is_final_status_message(update):
        """return True iff message is a final status update"""
        for r in CaptureRequest.FINAL_STATUSES:
            if update.startswith(r):
                return True
        return False

    def run(self):
        try:
            ctx = zmq.Context().instance()
            with ctx.socket(zmq.SUB) as sock:
                sock.linger = 0
                sock.setsockopt(zmq.SUBSCRIBE, self.id)
                sock.connect(self.source)

                poller = zmq.Poller()
                poller.register(self._pipe[1], flags=zmq.POLLIN)
                poller.register(sock, flags=zmq.POLLIN)
                getLogger(__name__).debug(f'Listening for status updates to {self.id}')
                self._pipe[1].send(b'')
                while True:
                    avail = dict(poller.poll())
                    if self._pipe[1] in avail:
                        getLogger(__name__).debug(f'Shutdown requested, terminating {self}')
                        break
                    elif sock not in avail:
                        time.sleep(.1)  # play nice
                        continue
                    id, update = sock.recv_multipart()
                    assert id == self.id
                    update = update.decode()
                    self._status_messages.append(update)
                    getLogger(__name__).debug(f'Status update for {self.id}: {update}')
                    if self.is_final_status_message(update):
                        break
        except zmq.ZMQError as e:
            getLogger(__name__).critical(f"{self} died due to {e}")
        finally:
            self._pipe[1].close()

    def latest(self):
        return self._status_messages[-1]

    def ready(self):
        self._pipe[0].recv()  # TODO make this line block


class CaptureJob:  # feedline client end
    def __init__(self, request: CaptureRequest, feedline_server: str, data_server: str, status_server: str,
                 submit=True):
        self.request = request
        self.feedline_server = feedline_server
        self._status_listner = StatusListner(request.id, status_server, initial_state='CREATED', start=False)
        self._datasaver = CaptureSinkFactory(request, data_server, start=False)
        if submit:
            self.submit()

    def status(self):
        """ Return the last known status of the request """
        return self._status_listner.latest()

    def cancel(self, kill_status_monitor=False, kill_data_sink=True):
        """

        Args:
            kill_status_monitor: Whether to terminate the status monitor
            kill_data_sink: Whether to terminate the data saver
        Returns: The parsed json submission result

        """
        self._kill_workers(kill_status_monitor=kill_status_monitor, kill_data_sink=kill_data_sink)
        ctx = zmq.Context().instance()
        with ctx.socket(zmq.REQ) as s:
            s.connect(self.feedline_server)
            s.send_pyobj(('abort', self.request.id))
            return s.recv_json()

    def _kill_workers(self, kill_status_monitor=True, kill_data_sink=True):
        if kill_status_monitor and self._status_listner.is_alive():
            try:
                self._datasaver.kill()
            except zmq.ZMQError as e:
                getLogger(__name__).warning(f'Caught {e} when telling data sink to terminate')
            if self._status_listner.is_alive():
                getLogger(__name__).warning(f'Status listener did not instantly terminate')
        if kill_data_sink and self._datasaver.is_alive():
            try:
                self._datasaver.kill()
            except zmq.ZMQError as e:
                getLogger(__name__).warning(f'Caught {e} when telling data sink to terminate')
            if self._datasaver.is_alive():
                getLogger(__name__).warning(f'Data sink did not instantly terminate')

    def data(self):
        return self._datasaver.data()

    def submit(self):
        self._status_listner.start()
        self._datasaver.start()
        self._status_listner.ready()
        self._datasaver.ready()
        try:
            self._submit()
        except Exception as e:
            self._status_listner.kill()
            self._datasaver.kill()
            raise e

    def _submit(self):
        ctx = zmq.Context().instance()
        with ctx.socket(zmq.REQ) as s:
            s.connect(self.feedline_server)
            s.send_pyobj(('capture', self.request))
            return s.recv_json()

    def __del__(self):
        self._kill_workers(kill_status_monitor=True, kill_data_sink=True)


class PowerSweepJob:
    def __init__(self, ntones=2048, points=512, min_attn=0, max_attn=30, attn_step=0.25, lo_center=0, fres=7.14e3,
                 use_cached=True):
        """
        Args:
            ntones (int): Number of tones in power sweep comb. Default is 2048.
            points (int): Number of I and Q samples to capture for each IF setting.
            min_attn (float): Lowest global attenuation value in dB. 0-30 dB allowed.
            max_attn (float): Highest global attenuation value in dB. 0-30 dB allowed.
            attn_step (float): Difference in dB between subsequent global attenuation settings.
                               0.25 dB is default and finest resolution.
            lo_center (float): Starting LO position in Hz. Default is XXX XX-XX allowed.
            fres (float): Difference in Hz between subsequent LO settings.
                               7.14e3 Hz is default and finest resolution we can produce with a 4.096 GSPS DAC
                               and 2**19 complex samples in the waveform look-up-table.

        Returns:
            PowerSweepRequest: Object which computes the appropriate hardware settings and produces the necessary
            CaptureRequests to collect power sweep data.
        """
        self.freqs = np.linspace(0, ntones - 1, ntones)
        self.points = points
        self.total_attens = np.arange(min_attn, max_attn + attn_step, attn_step)
        self._sweep_bw = SYSTEM_BANDWIDTH / ntones
        self.lo_centers = compute_lo_steps(center=lo_center, resolution=fres, bandwidth=self._sweep_bw)
        self.use_cached = use_cached

    def generate_jobs(self, submit=False) -> List[CaptureJob]:
        from .feedline_objects import DACConfig, IFConfig

        feedline_server = 'tcp://localhost:8888'
        capture_data_server = 'tcp://localhost:8889'
        status_server = 'tcp://localhost:8890'

        dacconfig = DACConfig('power_sweep_comb', n_uniform_tones=len(self.freqs))
        dacconfig_hash = hash(dacconfig)
        jobs = []
        for adc_atten, dac_atten in self.attens:
            for freq in self.lo_centers:
                ifconfig = IFConfig(lo=freq, adc_attn=adc_atten, dac_attn=dac_atten)
                fc = FeedlineConfig(dac_setup=dacconfig_hash if jobs else dacconfig, if_setup=ifconfig)
                cr = CaptureRequest(self.points, 'adc', feedline_config=fc, feedline_server=feedline_server)
                cj = CaptureJob(cr, feedline_server, capture_data_server, status_server, submit=False)
                jobs.append(cj)

        if submit:
            try:
                for j in jobs:
                    j.submit()
            except Exception as e:
                getLogger(__name__).debug('Cancelling all capture jobs use to a submission error.')
                for j in jobs:
                    j.cancel(kill_status_monitor=True, kill_data_sink=True)
                raise e

        return jobs
