import threading

import numpy

import mkidgen3 as g3
from scripts.zmq_server import ol

from . import power_sweep_freqs, N_CHANNELS, SYSTEM_BANDWIDTH
from .funcs import *
from .funcs import SYSTEM_BANDWIDTH, compute_lo_steps


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


class PhotonPipeCfg:
    def __init__(self, dac: DACOutputSpec, channel_spec, filter_coeffs, ddc_config, trigger_settings: TriggerSettings):
        self.dac = dac
        self.adc = None
        self.ddc = ddc_config
        self.channels = None
        self.filters = None
        self.thresholds = None


class PowerSweepPipeCfg(PhotonPipeCfg):
    def __init__(self, dac: DACOutputSpec):
        dac = DACOutputSpec('regular_comb')
        super().__init__(dac)
        self.channels = np.arange(0, 4096, 2, dtype=int)

class FLSetup:
    pass

    def compatible_with(self, other) -> bool:
        return False


class PhotonBuffer:
    """An nxm+1 sparse array full of photon events"""
    WATERMARK = 4500
    def __init__(self, _buf=None):
        self._buf=_buf

    @property
    def full(self):
        return (self._buf[0,:]>self.WATERMARK).any()

class CaptureRequest:
    def __init__(self, n, if_setup: IFSetup, pipe_setup, dac_setup: DACOutputSpec, channel_spec, ):
        self.ol =ol
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

    @property
    def id(self):
        return self._id

    @property
    def size(self):
        return self.points*2048

    @property
    def settings(self)->FLSetup:
        return FLSetup(if_setup=self.if_setup, pipe_setup=self.pipe_setup, dac_setup=self.dac_setup,
                       channel_spec=self.channel_spec)

    @property
    def complete(self):
        return self.ol.capture.axis2mm.complete

    def start(self):
        g3.apply_setup(ifsetup=self.if_setup, pipe=self.pipe_setup, dac=self.dac_setup)
        self._buffer = self.ol.capture.capture_adc(2 ** 19, complex=False, sleep=False, use_interrupt=False)

    def send(self, socket):
        def send_data(socket, capture_data):
            socket.send_pyobj(capture_data)
        self._thread = threading.Thread(target=send_data, args=(socket, self._buffer.copy()), daemon=True,
                                        name=f'CapXmit: {self}')
        self._thread.start()

    def sent(self):
        return self._thread is not None and not self._thread.is_alive()

    def __del__(self):
        if self._buffer is not None:
            self._buffer.free()


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
