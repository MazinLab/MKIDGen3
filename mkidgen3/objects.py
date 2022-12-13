import threading

import numpy
import numpy as np
import zmq

import mkidgen3 as g3
from scripts.zmq_server import ol

from . import power_sweep_freqs, N_CHANNELS, SYSTEM_BANDWIDTH
from .funcs import *
from .funcs import SYSTEM_BANDWIDTH, compute_lo_steps

import blosc

# _waveforms={}
# def WaveformFactory(*args, allow_caching=True, **kwargs):
#     global _waveforms
#     key = (args, tuple(kwargs.items()))
#     if allow_caching:
#         try:
#             return _waveforms[key]
#         except KeyError:
#             pass
#     wf = Waveform(*args,**kwargs)
#     if allow_caching:
#         _waveforms[key]=wf
#     return wf


class Waveform:
    def __init__(self, frequencies, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, seed=2, maximize_dynamic_range=True, compute=False):
        """
        Args:
            frequencies (float): list/array of frequencies in the comb
            n_samples (int): number of complex samples in waveform
            sample_rate (float): waveform sample rate in Hz
            amplitudes (float): list/array of amplitudes, one per frequency in (0,1]. If None, all ones is assumed.
            phases (float): list/array of phases, one per frequency in [0, 2*np.pi). If None, generates random phases using input seed.
            iq_ratios (float): list of ratios for IQ values used to help minimize image tones in band.
                       Allowed values between 0 and 1. If None, 50:50 ratio (all ones) is assumed.
                      TODO: what does this actually do and how does it work
            phase_offsets (float): list/array of phase offsets in [0, 2*np.pi)
            seed (int): random seed to seed phase randomization process

        Attributes:
            values (float): Computed waveform values. Amplitude is unscaled and is the product of additions of unit waveforms.
            quant_vals (int): Computed waveform values quantized to DAC digital format with optimum precision
            max_quant_error (float): maximum difference between quant_vals and values scaled to the DAC max output.
        """
        self.freqs = frequencies
        self.points = n_samples
        self.fs = sample_rate
        self.amps = amplitudes if amplitudes is not None else np.ones_like(frequencies)
        self.phases = phases if phases is not None else np.random.default_rng(seed=seed).uniform(0., 2.*np.pi, len(frequencies))
        self.iq_ratios = iq_ratios if iq_ratios is not None else np.ones_like(frequencies)
        self.phase_offsets = phase_offsets if phase_offsets is not None else np.zeros_like(frequencies)
        self._seed = seed
        self.quant_freqs = quantize_frequencies(self.freqs, rate=sample_rate, n_samples=n_samples)
        self._values = None
        self.quant_vals = None
        self.quant_error = None
        self.maximize_dynamic_range = maximize_dynamic_range
        if compute:
            self.values

    @property
    def values(self):
        if self._values is None:
            self._values=self._compute_waveform()
            if self.maximize_dynamic_range:
                self._waveform.optimize_random_phase(
                    max_quant_err=3 * predict_quantization_error(resolution=DAC_RESOLUTION),
                    max_attempts=10)
        return self._values

    def _compute_waveform(self):
        iq = np.zeros(self.points, dtype=np.complex64)
        # generate each signal
        t = 2 * np.pi * np.arange(self.points) / self.fs
        logging.getLogger(__name__).debug(f'Computing net waveform with {self.freqs.size} tones. For 2048 tones this takes about 7 min.')
        for i in range(self.freqs.size):
            exp = self.amps[i] * np.exp(1j * (t * self.quant_freqs[i] + self.phases[i]))
            scaled = np.sqrt(2) / np.sqrt(1 + self.iq_ratios[i] ** 2)
            c1 = self.iq_ratios[i] * scaled * np.exp(1j * np.deg2rad(self.phase_offsets)[i])
            iq.real += c1.real * exp.real + c1.imag * exp.imag
            iq.imag += scaled * exp.imag
        return iq

    def _optimize_random_phase(self, max_quant_err=3*predict_quantization_error(resolution=DAC_RESOLUTION), max_attempts=10):
        """
        inputs:
        - max_quant_error: float
            maximum allowable quantization error for real or imaginary samples.
            see predict_quantization_error() for how to estimate this value.
        - max_attempts: int
            Max number of times to recompute the waveform and attempt to get a quantization error below the specified max
            before giving up.

        returns: floating point complex waveform with optimized random phases
        """
        if max_quant_err is None:
            max_quant_err = 3*predict_quantization_error(resolution=DAC_RESOLUTION)

        self.quant_vals, self.quant_error = quantize_to_int(self._values, resolution=DAC_RESOLUTION, signed=True,
                                                            word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                            return_error=True)
        cnt=0
        while self.quant_error>max_quant_err:
            logging.getLogger(__name__).warning("Max quantization error exceeded. The freq comb's relative phases may have added up sub-optimally."
                                        "Calculating with new random phases")
            self._seed+=1
            self.phases = np.random.default_rng(seed=self._seed).uniform(0., 2. * np.pi, len(self.freqs))
            self._values = self._compute_waveform()
            self.quant_vals, self.quant_error = quantize_to_int(self._values, resolution=DAC_RESOLUTION, signed=True,
                                           word_length=ADC_DAC_INTERFACE_WORD_LENGTH, return_error=True)
            cnt+=1
            if cnt>max_attempts:
                raise Exception("Process reach maximum attempts: Could not find solution below max quantization error.")
        return


class DACOutputSpec:
    def __init__(self, ntones, name: str, n_uniform_tones=None, waveform_spec: [np.array, dict, Waveform]=None,
                 qmc_settings=None):
        self.spec_type = name
        freqs = power_sweep_freqs(ntones, bandwidth=SYSTEM_BANDWIDTH)
        wf_spec = dict(n_samples = 2 ** 19, sample_rate = 4.096e9, amplitudes = None, phases = None,
                       iq_ratios = None, phase_offsets = None, seed = 2)
        if isinstance(waveform_spec, (np.array, list)):
            wf_spec['freqs']=np.asarray(waveform_spec)

        if isinstance(waveform_spec,(dict, np.array, list)):
            wf_spec.update(waveform_spec)
            self._waveform = Waveform(**wf_spec)
        elif isinstance(waveform_spec, Waveform):
            self._waveform = waveform_spec
        else:
            raise ValueError('doing it wrong')


        self.qmc_settings = qmc_settings

    def __hash__(self):
        return hash(f'{self.waveform.quant_vals}{self.qmc_settings}')

    @property
    def waveform(self):
        return self._waveform.values


class IFSetup:
    def __init__(self, lo, adc_attn, dac_attn):
        self.lo = lo
        self.adc_attn = adc_attn
        self.dac_attn = dac_attn

    def __hash__(self):
        return hash(f'{self.lo}{self.adc_attn}{self.dac_attn}')


class TriggerSettings:
    def __init__(self):
        self.holdoffs = None
        self.thresholds = None


class DDCConfig:
    def __init__(self, tones, centers, offsets):
        self.tones=tones
        self.centers=centers
        self.offsets=offsets

    def __hash__(self):
        return hash(f'{self.tones}{self.centers}{self.offsets}')



class FLSetup:
    def __init__(self, if_setup=None, dac_setup=None, pp_setup=None, ddc=None,
                 adc_setup=None, channels=None, filters=None, thresholds=None):
        self.if_setup = if_setup
        self.dac_setup = dac_setup
        self.pp_setup = pp_setup
        self.adc_setup = adc_setup
        self.filters = filters
        self.thresholds = thresholds
        self.channels = channels
        self.ddc = ddc

    def compatible_with(self, other) -> bool:
        dac = self.dac_setup is None or self.dac_setup.compatible_with(other.dac_setup)
        ifb = self.if_setup is None or self.if_setup.compatible_with(other.if_setup)
        adc = self.adc_setup is None or self.adc_setup.compatible_with(other.adc_setup)
        pp = self.pp_setup is None or self.pp_setup.compatible_with(other.pp_setup)
        thresh = self.thresholds is None or self.thresholds.compatible_with(other.thresholds)
        chan = self.channels is None or self.channels.compatible_with(other.channels)
        filt = self.filters is None or self.filters.compatible_with(other.filters)
        ddc = self.ddc is None or self.ddc.compatible_with(other.ddc)
        return dac and ifb and adc and pp and filt and chan and thresh and ddc


class PowerSweepPipeCfg(FLSetup):
    def __init__(self, dac: DACOutputSpec):
        dac = DACOutputSpec('regular_comb')
        super().__init__(dac)
        self.channels = np.arange(0, 4096, 2, dtype=int)



class FLPhotonBuffer:
    """An nxm+1 sparse array full of photon events"""
    WATERMARK = 4500
    def __init__(self, _buf=None):
        self._buf=_buf

    @property
    def full(self):
        return (self._buf[0,:]>self.WATERMARK).any()

class CapDest:
    def __init__(self, data_dest:str, status_dest:str = ''):
        self._dest = data_dest
        self._socket = None
        self._status = None
        self._status_dest = status_dest

    def establish(self, context:zmq.Context):
        if self._status_dest:
            self._status = context.socket(zmq.REQ)
            self._status.connect(self._dest)
        if self._dest.startswith('file'):
            raise NotImplementedError
            f = os.path.open(self._dest,'ab')
            f.close()
        else:
            self._socket = context.socket(zmq.PUB)
            self._socket.connect(self._dest)

    def update_status(self, status, context):
        self._socket = context.socket(zmq.PUB)
        self._socket.connect(self._dest)

    def __del__(self):
        try:
            self._socket.close()
        except AttributeError:
            pass
        try:
            self._status.close()
        except AttributeError:
            pass

class CaptureAbortedException(Exception):
    pass

class ADCCaptureSink:
    pass

class PhotonCaptureSink:
    def __init__(self, source, context:zmq.Context=None):
        pass

    def terminate(self, context:zmq.Context=None):
        context = context or zmq.Context.instance()
        _terminate = context.socket(zmq.PUB)
        _terminate.connect('inproc://PhotonCaptureSink.terminator.inproc')
        _terminate.send(b'')
        _terminate.close()

    def capture(self):
        t = threading.Thread(target =self._main, args=(hdf, xymap, feedline_source, fl_ids))
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
        DETECTOR_SHAPE = (128,80)
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

        live_image=np.zeros(DETECTOR_SHAPE)
        live_image_by_fl = live_image.reshape(n_fl, fl_npix)
        photons_rabuf = np.recarray(MAX_NEW_PHOTONS,
                                       dtype=(('time', 'u32'), ('x','u32'), ('y','u32'),
                                              ('phase','u16')))

        while True:
            avail = poller.poll()
            if term in avail:
                break

            frame = data.recv_multipart(copy=False)
            fl_id = frame[0]
            time_offset = frame[1]
            d = blosc.decompress(frame[1])
            #buffer is nchan*nmax+1 32bit: 16bit time(base2) 16bit phase
            #make array of to [nnmax+1, nchan, 2] uint16
            #nmax will always be <<2^12 number valid will be at [0,:,0]
            #times need oring w offset
            #photon data is d[1:d[0,i,0], i, :]

            nnew = d[0,:,0].sum()
            #if we wanted to save binary data then we could save this, the x,y list, and the time offset
            #mean pixel count rate in this packet is simply [0,:,0]/dt
            fl_ndx = fl_id_to_index[fl_id]
            live_image_by_fl[fl_ndx, :] += d[0,:,0]

            # if live_image_ready:
            #     live_image_socket.send_multipart([f'liveim', blosc.compress(live_image)])

            cphot = np.cumsum(d[0,:,0], dtype=int)
            for i in range(fl_npix):
                sl_out = slice(cphot[i], cphot[i] + d[0,i,0])
                sl_in = slice(1, d[0,i,0])
                photons_rabuf['time'][sl_out] = d[sl_in,:, 0]
                photons_rabuf['time'][sl_out]|=time_offset
                photons_rabuf['phase'][sl_out] = d[sl_in, :, 1]
                photons_rabuf['x'][sl_out] = xymap[fl_ndx, i, 0]
                photons_rabuf['y'][sl_out] = xymap[fl_ndx, i, 1]
            hdf.grow_by(photons_rabuf[:nnew])

        term.close()
        data.close()
        hdf.close()



class CaptureRequest:
    def __init__(self, n, if_setup: IFSetup, pipe_setup, dac_setup: DACOutputSpec, channel_spec, ):
        self._buffer=None
        self._thread=None
        self.points = n
        self.tap=None
        self.points=n
        self.if_setup=if_setup
        self.pipe_setup=pipe_setup
        self.dac_setup=dac_setup
        self.channel_spec=channel_spec
        self._id = hash(repr(self))

        self._data_dest = ''
        self._status_dest = ''
        self._abort_source = ''
        self._status_socket = None
        self._data_socket = None
        self._abort_socket = None

        self._status = 'created'
        self._status_messages = []


    @property
    def id(self):
        return self._id

    def abort(self, context:zmq.Context=None):
        if self._abort_socket is None:
            self._abort_socket = context.socket(zmq.REQ)
            self._abort_socket.setsockopt(zmq.SUBSCRIBE, self._id)
            self._abort_socket.connect(self._abort_source)
        else:
            self._abort_socket = zmq.Socket
            if self._abort_socket.type==zmq.REP:
                raise RuntimeError('Abort may only be called by the requester.')
        return self._abort_socket.send(b'')

    def aborted(self, context:zmq.Context=None):
        if self._abort_socket is None:
            self._abort_socket = context.socket(zmq.REP)
            self._abort_socket.connect(self._abort_source)
        return self._abort_socket.poll(1)!=0

    def establish(self, context:zmq.Context=None):
        self._status_socket = context.socket(zmq.REQ)
        self._status_socket.connect(self._status_dest)
        self._data_socket = context.socket(zmq.PUB)
        self._data_socket.connect(self._data_dest)
        self._abort_socket = context.socket(zmq.REP)
        self._abort_socket.connect(self._abort_source)
        self._established = True
        self.set_status('established')

    def fail(self, message, context:zmq.Context=None):
        self.set_status('failed', message, context=context)

    def finish(self):
        self.set_status('complete')

    def add_data(self, data, status=''):
        if not self._established:
            raise RuntimeError('Establish must be called before add_data')
        self._data_dest.send_multipart([self.id, blosc.compress(data)])
        self.set_status('capturing', message=status)

    def set_status(self, status, message='', context:zmq.Context=None):
        self._status = status
        if message:
            self._status_messages.append(message)
        """get appropriate context and send current sztatus message after connecting soocket. if no then simply log"""
        if not self._established:
            _status_dest = context.socket(zmq.REQ)
            _status_dest.connect(self._status_dest)
        else:
            _status_dest=self._status_dest
        _status_dest.send_multipart([status, message])

    @property
    def size(self):
        return self.points*2048

    @property
    def settings(self)->FLSetup:
        return FLSetup(if_setup=self.if_setup, pipe_setup=self.pipe_setup, dac_setup=self.dac_setup,
                       channel_spec=self.channel_spec)



class PowerSweepRequest:
    def __init__(self, ntones=2048, points=512, min_attn=0, max_attn=30, attn_step=0.25, lo_center=0, fres=7.14e3, use_cached=True):
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
        self.freqs=
        self.points = points
        self.total_attens=np.arange(min_attn,max_attn+attn_step,attn_step)
        self._sweep_bw=SYSTEM_BANDWIDTH/ntones
        self.lo_centers = compute_lo_steps(center=lo_center, resolution=fres, bandwidth=self._sweep_bw)
        self.use_cached = use_cached

    def capture_requests(self):
        dacsetup=DACOutputSpec('power_sweep_comb', n_uniform_tones=self.ntones)
        return [CaptureRequest(self.samples, dac_setup=dacsetup,
                               if_setup=IFSetup(lo=freq, adc_attn=adc_atten,dac_attn=dac_atten))
                for (adc_atten,dac_atten) in self.attens for freq in self.lo_centers]
