import threading
import time

import numpy as np
import zmq
import blosc2

# from . import power_sweep_freqs, N_CHANNELS, SYSTEM_BANDWIDTH
from mkidgen3.funcs import *
from mkidgen3.funcs import SYSTEM_BANDWIDTH, compute_lo_steps
import logging
import binascii
import os
from logging import getLogger
from hashlib import md5
from collections import defaultdict


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


def hasher(v, pass_none=False):
    if v is None and pass_none:
        return None

    try:
        if v is None:
            v = '___python_None'
        return md5(str(v).encode()).hexdigest()
    except TypeError:
        return md5(v.tobytes()).hexdigest()


class Waveform:
    def __init__(self, frequencies=None, n_samples=2 ** 19, sample_rate=4.096e9, amplitudes=None, phases=None,
                 iq_ratios=None,
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
        self.phases = phases if phases is not None else np.random.default_rng(seed=seed).uniform(0., 2. * np.pi,
                                                                                                 len(frequencies))
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
            self._values = self._compute_waveform()
            if self.maximize_dynamic_range:
                self._waveform.optimize_random_phase(
                    max_quant_err=3 * predict_quantization_error(resolution=DAC_RESOLUTION),
                    max_attempts=10)
        return self._values

    def _compute_waveform(self):
        iq = np.zeros(self.points, dtype=np.complex64)
        # generate each signal
        t = 2 * np.pi * np.arange(self.points) / self.fs
        logging.getLogger(__name__).debug(
            f'Computing net waveform with {self.freqs.size} tones. For 2048 tones this takes about 7 min.')
        for i in range(self.freqs.size):
            exp = self.amps[i] * np.exp(1j * (t * self.quant_freqs[i] + self.phases[i]))
            scaled = np.sqrt(2) / np.sqrt(1 + self.iq_ratios[i] ** 2)
            c1 = self.iq_ratios[i] * scaled * np.exp(1j * np.deg2rad(self.phase_offsets)[i])
            iq.real += c1.real * exp.real + c1.imag * exp.imag
            iq.imag += scaled * exp.imag
        return iq

    def _optimize_random_phase(self, max_quant_err=3 * predict_quantization_error(resolution=DAC_RESOLUTION),
                               max_attempts=10):
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
            max_quant_err = 3 * predict_quantization_error(resolution=DAC_RESOLUTION)

        self.quant_vals, self.quant_error = quantize_to_int(self._values, resolution=DAC_RESOLUTION, signed=True,
                                                            word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                            return_error=True)
        cnt = 0
        while self.quant_error > max_quant_err:
            logging.getLogger(__name__).warning(
                "Max quantization error exceeded. The freq comb's relative phases may have added up sub-optimally."
                "Calculating with new random phases")
            self._seed += 1
            self.phases = np.random.default_rng(seed=self._seed).uniform(0., 2. * np.pi, len(self.freqs))
            self._values = self._compute_waveform()
            self.quant_vals, self.quant_error = quantize_to_int(self._values, resolution=DAC_RESOLUTION, signed=True,
                                                                word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                                return_error=True)
            cnt += 1
            if cnt > max_attempts:
                raise Exception("Process reach maximum attempts: Could not find solution below max quantization error.")
        return


class FLConfigMixin:
    _settings = tuple()

    def __eq__(self, other):
        """ Feedline configs are equivalent if all of their settings are equivalent"""
        if not isinstance(other, type(self)):
            return False
        if self.hashed or other.hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if a is None or b is None:
                continue
            if a != b:
                return False
        return True

    @property
    def hashed(self):
        return self._hashed is not None

    @property
    def _hash_data(self):
        hash_data = ((k, hasher(getattr(self, k), pass_none=True)) for k in self._settings)
        return tuple(sorted(hash_data, key=lambda x: x[0]))

    def __hash__(self):
        if self.hashed:
            return self._hashed
        return int(hasher(self._hash_data), 16)

    def __str__(self):
        name = self.__class__.__name__
        if self._hashed:
            return f"{name}: {hash(self)} (hashed)"
        else:
            return (f"{name}: {hash(self)}\n"
                    f"  {self.settings_dict()}")

    @property
    def hashed_form(self):
        return type(self)(_hashed=hash(self))

    def settings_dict(self, omit_none=True, hashed=False, unhasher_cache=None):
        """Will not include settings that are None"""
        if self._hashed and not hashed:
            try:
                x = unhasher_cache[self._hashed]
            except (KeyError, TypeError):
                raise ValueError('Hashed configs do not support unhashed settings '
                                 'retrieval without an entry the unhasher_cache')
        else:
            x = self
        if hashed:
            d = {'_hashed': hash(self)}
        else:
            d = {k: getattr(x, k) for k in self._settings if not (getattr(x, k) is None and omit_none)}
        return d


class FLMetaConfigMixin:
    def __eq__(self, other):
        """ Feedline configs are equivalent if all of their settings are equivalent"""
        for k, v in self.__dict__.items():
            other_v = getattr(other, k)
            assert isinstance(v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            assert isinstance(other_v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            # if either is None we match
            if v is None or other_v is None:
                continue

            # Compute hash if necessary
            v_hash = hash(v) if isinstance(v, FLMetaConfigMixin) else v
            o_hash = hash(other_v) if isinstance(other_v, FLMetaConfigMixin) else other_v
            if v_hash != o_hash:
                return False
        return True

    def __hash__(self):
        def hasher(v):
            if v is None:
                v = '___python_None'
            return md5(str(v).encode()).hexdigest()

        return int(hasher(tuple(sorted(((k, hasher(v)) for k, v in self.__dict__.items()), key=lambda x: x[0]))), 16)

    def __iter__(self):
        for v in vars(self):
            if v.startswith('_'):
                continue
            yield v, getattr(self, v)

    @property
    def hashed_form(self):
        d = {k: v if v is None else v.settings_dict(hashed=True) for k, v in self}
        return type(self)(**d)

    def settings_dict(self, hashed=False):
        return {k: v if v is None else v.settings_dict(hashed=hashed) for k, v in self}

    def iter(self, hashed=True, unhashed=True):
        """an iterator of config_key: value pairs"""
        for k, v in self:
            if isinstance(v, FLMetaConfigMixin):
                for a, b in v.iter(hashed=hashed, unhashed=unhashed):
                    yield f'{k}.{a}', b
            else:
                if hashed and v.hashed:
                    yield k, v
                if unhashed and not v.hashed:
                    yield k, v


class ADCconfig(FLConfigMixin):
    _settings = tuple()

    def __init__(self, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return


class DACConfig(FLConfigMixin):
    _settings = ('n_uniform_tones', 'qmc_settings')

    def __init__(self, n_uniform_tones=None, waveform_spec: [np.ndarray, dict, Waveform] = None,
                 qmc_settings=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.n_uniform_tones = n_uniform_tones
        self.waveform_spec=waveform_spec
        self._waveform = None
        # self.spec_type =

        wf_spec = dict(n_samples=2 ** 19, sample_rate=4.096e9, amplitudes=None, phases=None,
                       iq_ratios=None, phase_offsets=None, seed=2)
        if n_uniform_tones is not None:
            wf_spec['frequencies'] = power_sweep_freqs(n_uniform_tones, bandwidth=SYSTEM_BANDWIDTH)

        if isinstance(waveform_spec, (np.ndarray, list)):
            wf_spec['frequencies'] = np.asarray(waveform_spec)
        elif isinstance(waveform_spec, dict):
            wf_spec.update(waveform_spec)
        elif isinstance(waveform_spec, Waveform):
            self._waveform = waveform_spec

        if not isinstance(waveform_spec,Waveform):
            self._waveform = Waveform(**wf_spec)

        if self._waveform is None:
            raise ValueError('doing it wrong')

        self.qmc_settings = qmc_settings


    @property
    def quant_vals(self):
        return self._waveform.quant_vals

    @property
    def waveform(self):
        return self._waveform.values


class IFConfig(FLConfigMixin):
    _settings = ('lo', 'adc_attn', 'dac_attn')

    def __init__(self, lo=None, adc_attn=None, dac_attn=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.lo = lo
        self.adc_attn = adc_attn
        self.dac_attn = dac_attn


class TriggerConfig(FLConfigMixin):
    _settings = ('holdoffs', 'thresholds')

    def __init__(self, holdoffs: np.ndarray=None, thresholds: np.ndarray=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.holdoffs = holdoffs
        self.thresholds = thresholds


class ChannelConfig(FLConfigMixin):
    _settings = ('frequencies',)

    def __init__(self, frequencies=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.frequencies = frequencies


class DDCConfig(FLConfigMixin):
    _settings = ('tones', 'loop_center', 'phase_offset', 'center_relative', 'quantize')

    def __init__(self, tones=None, loop_center=None, phase_offset=None, center_relative=False,
                 quantize=True, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.tones = tones
        self.loop_center = loop_center
        self.phase_offset = phase_offset
        self.center_relative = center_relative
        self.quantize = quantize


class FilterConfig(FLConfigMixin):
    _settings = ('coefficients',)

    def __init__(self, coefficients=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.coefficients = coefficients


class PhotonPipeConfig(FLMetaConfigMixin):
    def __init__(self, chan_config: ChannelConfig = None, ddc_config: DDCConfig = None,
                 filter_config: FilterConfig = None, trig_config: TriggerConfig = None):
        self.chan_config = ChannelConfig(**chan_config) if isinstance(chan_config, dict) else chan_config
        self.ddc_config = DDCConfig(**ddc_config) if isinstance(ddc_config, dict) else ddc_config
        self.trig_config = TriggerConfig(**trig_config) if isinstance(trig_config, dict) else trig_config
        self.filter_config = FilterConfig(**filter_config) if isinstance(filter_config, dict) else filter_config

    def __str__(self):
        return (f"PhotonPipe {hash(self)}:\n"
                f"  Chan: {self.chan_config}\n"
                f"  DDC: {self.ddc_config}\n"
                f"  Filt: {self.filter_config}\n"
                f"  Trig: {self.trig_config}")


class FeedlineConfig(FLMetaConfigMixin):
    """All attributes must be _FLConfigMixin"""

    def __init__(self, if_config: IFConfig = None, dac_config: DACConfig = None, pp_config: PhotonPipeConfig = None,
                 adc_config: ADCconfig = None):
        self.if_config = IFConfig(**if_config) if isinstance(if_config, dict) else if_config
        self.dac_config = DACConfig(**dac_config) if isinstance(dac_config, dict) else dac_config
        self.pp_config = PhotonPipeConfig(**pp_config) if isinstance(pp_config, dict) else pp_config
        self.adc_config = ADCconfig(**adc_config) if isinstance(adc_config, dict) else adc_config
        #TODO to support captures of less than all groups we need to add a self.capture_setup which has group
        # settings for iq and phase

    def __str__(self):
        pp = str(self.pp_config).replace('\n  ', '\n    ')
        return (f"FeedlineConfig {hash(self)}:\n"
                f"  IF: {self.if_config}\n"
                f"  DAC: {self.dac_config}\n"
                f"  ADC: {self.adc_config}\n"
                f"  PP: {pp}")

    def compatible_with(self, other):
        """ Compatibility of hashed configs is more restrictive than unhashed as the comparison can not
        handle the compatibility of 'None' """
        return self == other

    # def iter(self, hashed=True, unhashed=True):
    #     """an iterator of config_key: value pairs"""
    #     for k, v in self:
    #         if isinstance(v, FLMetaConfigMixin):
    #             for a, b in v.iter(hashed=hashed, unhashed=unhashed):
    #                 yield f'{k}.{a}', b
    #         else:
    #             if hashed and v.hashed:
    #                 yield k, v
    #             if unhashed and not v.hashed:
    #                 yield k, v


class FeedlineConfigManager:
    def __init__(self):
        self._config = {}
        self._settings = {}
        self._cache = {}

    def learn(self, config: FeedlineConfig):
        """Commit configuration info to memory for later use, hashed configurations are not learnable"""
        self._cache.update({hash(v): v for k, v in config.iter(hashed=False)})# if hash(v) not in self._cache})

    def unlearned_hashes(self, config: FeedlineConfig):
        """
        Return a set of any hashed config hashes in the config that have not been learned.

        The presence of unlearned hashes will prevent a config from being added to the manager.
        """
        return set(hash(v) for k, v in config.iter(unhashed=False) if hash(v) not in self._cache)

    def effective(self) -> FeedlineConfig:
        """
        Return a feedline configuration resulting from settings in the set,
        the config will not contain any hashed settings.
        """
        setting_dict = defaultdict(lambda: defaultdict(dict))

        for config in self._config.values():
            for k, v in config:
                if isinstance(v, FLMetaConfigMixin):
                    for k2, v2 in v:
                        # if k2 not in setting_dict[k]:
                        #     setting_dict[k][k2] = {}
                        setting_dict[k][k2].update(v2.settings_dict(unhasher_cache=self._cache))
                else:
                    setting_dict[k].update(v.settings_dict(unhasher_cache=self._cache))
        return FeedlineConfig(**setting_dict)

    def pop(self, id) -> bool:
        """
        Remove settings from the set

        Args:
            id: settings id to remove, raises KeyError if the key hasn't been added to the manager

        Returns: True iff the effective settings changed as a result of the pop
        """
        if id not in self._config:
            return False
        x = self.effective()
        self._config.pop(id)
        return self.effective() != x

    def add(self, id, config: FeedlineConfig) -> FeedlineConfig:
        """
        Add a feedline config to the managed set of settings with an id for later removal. Configs with hashed
        settings can not be added unless their hashes have been previously learned. All unhashed settings are learned
        upon addition.

        If the added config has settings that are not compatible with existing settings it the set then XXX happen

        Args:
            id: an id for the settings (for later removal from the set)
            config: a feedline config at any level of specificity

        Returns: A feedline config with the settings that need updating populated and the rest None

        Raises: ValueError if a config contains unlearned hashes.

        """
        self.learn(config)
        if self.unlearned_hashes(config):
            raise ValueError('Config contains unlearned hashes')

        old = self.effective()
        self._config[id] = config
        new = self.effective()
        for k,v in new:
            if getattr(old, k) is None:
                continue
            if isinstance(v, FLMetaConfigMixin):
                for k2, v2 in v:
                    if getattr(getattr(old, k), k2) == v:
                        setattr(setattr(new, k), k2, None)
            else:
                if getattr(old, k) == v:
                    setattr(new, k, None)
        return new


class FeedlineStatus:
    def __init__(self):
        self.status = 'feedline status'


class DACStatus:
    def __init__(self, waveform: Waveform):
        self.waveform = waveform
        self._output_on = False


class DDCStatus:
    def __init__(self, tone_increments, phase_offsets, centers):
        self.tone_increments = tone_increments
        self.phase_offsets = phase_offsets
        self.centers = centers


class PowerSweepPipeCfg(FeedlineConfig):
    def __init__(self):
        super().__init__(dac_setup=DACConfig('regular_comb'),
                         channels=np.arange(0, 4096, 2, dtype=int))


class FLPhotonBuffer:
    """An nxm+1 sparse array full of photon events"""
    WATERMARK = 4500

    def __init__(self, _buf=None):
        self._buf = _buf

    @property
    def full(self):
        return (self._buf[0, :] > self.WATERMARK).any()


class CaptureAbortedException(Exception):
    pass


class CaptureRequest:
    STATUS_ENDPOINT = 'inproc://cap_stat.xsub'
    DATA_ENDPOINT = 'inproc://cap_data.xsub'
    FINAL_STATUSES = ('finished', 'aborted', 'failed')

    def __init__(self, n, tap, feedline_config: FeedlineConfig, feedline_server):
        self.nsamp = n
        self._last_status = None
        self.tap = tap  # maybe add some error handling here
        self.feedline_config = feedline_config
        self._feedline_server = feedline_server
        self._status_socket = None
        self._data_socket = None
        self._established = False

    def __hash__(self):
        return int(md5(str((hash(self.feedline_config), self.tap,
                            self.nsamp, self._feedline_server)).encode()).hexdigest(), 16)

    def __del__(self):
        self.destablish()

    def __str__(self):
        return f'CapReq {str(hash(self))}'

    @property
    def type(self):
        return 'engineering' if self.tap in ('adc', 'iq', 'phase') else self.tap

    @property
    def id(self):
        return str(hash(self)).encode()

    def establish(self, context: zmq.Context = None):
        context = context or zmq.Context.instance()
        self._status_socket = context.socket(zmq.PUB)
        self._status_socket.connect(self.STATUS_ENDPOINT)
        self._data_socket = context.socket(zmq.PUB)
        self._data_socket.connect(self.DATA_ENDPOINT)
        self._send_status('established')

    def destablish(self):
        try:
            self._status_socket.close()  # TODO do we need to wait to make sure any previous sends get sent
            self._status_socket = None
        except AttributeError:
            pass
        try:
            self._data_socket.close()
            self._data_socket = None
        except AttributeError:
            pass

    def fail(self, message):
        self._data_socket.send_multipart([self.id, b''])
        self._send_status('failed', message)
        self.destablish()

    def finish(self):
        self._data_socket.send_multipart([self.id, b''])
        self._send_status('finished')
        self.destablish()

    def abort(self, message):
        self._data_socket.send_multipart([self.id, b''])
        self._send_status('aborted', message)
        self.destablish()

    def add_data(self, data, status=''):
        if not self._data_socket or not self._status_socket:
            raise RuntimeError('Establish must be called before add_data')
        # TODO ensure we are being smart about pointers and buffer acces vs copys
        self._data_socket.send_multipart([self.id, blosc2.compress(data)])
        self._send_status('capturing', status)

    def _send_status(self, status, message=''):
        if not self._status_socket:
            raise RuntimeError('No status_socket connection available')
        update = f'{status}:{message}'
        getLogger(__name__).debug(f'Published status update {self.id}: "{update}"')
        self._last_status = (status, message)
        self._status_socket.send_multipart([self.id, update.encode()])

    def set_status(self, status, message='', context: zmq.Context = None):
        """
        get appropriate context and send current status message after connecting the status socket

        context is ignored if a status destination connection is extant
        """
        if not self._status_socket:
            context = context or zmq.Context().instance()
            self._status_socket = context.socket(zmq.PUB)
            self._status_socket.connect(self.STATUS_ENDPOINT)
        self._send_status(status, message)

    @property
    def size_bytes(self):
        return self.nsamp * self.nchan * self.dwid

    @property
    def nchan(self):
        return 2048 if self.tap in ('iq', 'phase') else 1

    @property
    def ngroup(self):
        return 256 if self.tap in ('iq', 'phase') else 1

    @property
    def dwid(self):
        """Data size of sample in bytes"""
        return 4 if self.tap in ('adc', 'iq') else 2

    @property
    def buffer_shape(self):
        return self.nsamp, self.nchan, self.dwid//2


class PowerSweepRequest:
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

    def capture_requests(self):
        dacsetup = DACOutputSpec('power_sweep_comb', n_uniform_tones=self.ntones)
        return [CaptureRequest(self.samples, dac_setup=dacsetup,
                               if_setup=IFSetup(lo=freq, adc_attn=adc_atten, dac_attn=dac_atten))
                for (adc_atten, dac_atten) in self.attens for freq in self.lo_centers]
