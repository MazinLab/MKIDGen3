import pickle
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
    @property
    def output_waveform(self):
        """Subclasses shall implement are return """
        return self._values

    @property
    def sample_rate(self):
        """Subclasses shall implement are return """
        return self._sample_rate

    @property
    def fpgen(self):
        return self._fpgen


class TabulatedWaveform(Waveform):
    def __init__(self, tabulated_values=None, sample_rate=DAC_SAMPLE_RATE):
        self._values = tabulated_values
        self._fpgen = None
        self._sample_rate=sample_rate


class FreqlistWaveform(Waveform):
    def __init__(self, frequencies=None, n_samples=2 ** 19, sample_rate=4.096e9, amplitudes=None, phases=None,
                 iq_ratios=None, phase_offsets=None, seed=2, maximize_dynamic_range=True, compute=False):
        """
        Args:
            frequencies: list/array of frequencies in the comb
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
        self.freqs = np.asarray(frequencies)
        self.n_samples = n_samples
        self._sample_rate = sample_rate
        self.amps = amplitudes if amplitudes is not None else np.ones_like(frequencies)

        if phases is None:
            self.phases = np.random.default_rng(seed=seed).uniform(0., 2. * np.pi, size=self.freqs.size)
        else:
            self.phases = np.asarray(phases)
        self.iq_ratios = np.asarray(iq_ratios) if iq_ratios is not None else np.ones_like(frequencies)
        self.phase_offsets = np.asarray(phase_offsets) if phase_offsets is not None else np.zeros_like(frequencies)
        self.quant_freqs = quantize_frequencies(self.freqs, rate=sample_rate, n_samples=n_samples)
        self._seed = seed
        self.__values = None
        self.quant_vals = None
        self.quant_error = None
        self.maximize_dynamic_range = maximize_dynamic_range
        self._fpgen = None if self.maximize_dynamic_range else 'simple'
        if compute:
            self.output_waveform

    @property
    def _values(self):
        if self.__values is None:
            self.__values = self._compute_waveform()
            if self.maximize_dynamic_range:
                self._optimize_random_phase(max_quant_err=3 * predict_quantization_error(resolution=DAC_RESOLUTION),
                                            max_attempts=10)
        return self.__values if self.maximize_dynamic_range else self.quant_vals

    def _compute_waveform(self):
        iq = np.zeros(self.n_samples, dtype=np.complex64)
        # generate each signal
        t = 2 * np.pi * np.arange(iq.size) / self._sample_rate
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
            self.__values = self._compute_waveform()
            self.quant_vals, self.quant_error = quantize_to_int(self._values, resolution=DAC_RESOLUTION, signed=True,
                                                                word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                                return_error=True)
            cnt += 1
            if cnt > max_attempts:
                raise Exception("Process reach maximum attempts: Could not find solution below max quantization error.")
        return


def WaveformFactory(n_uniform_tones=None, output_waveform=None, frequencies=None,
             n_samples=2 ** 19, sample_rate=4.096e9, amplitudes=None, phases=None,
             iq_ratios=None, phase_offsets=None, seed=2, maximize_dynamic_range=True, compute=False):
    if output_waveform is not None:
        return TabulatedWaveform(tabulated_values=output_waveform, sample_rate=sample_rate)

    if n_uniform_tones is not None:
        if n_uniform_tones not in (512, 1024, 2048):
            raise ValueError('Requested number of power sweep tones not supported. Allowed values are 512, 1024, 2048.')
        frequencies = uniform_freqs(n_uniform_tones, bandwidth=SYSTEM_BANDWIDTH)[::N_CHANNELS // n_uniform_tones]
    frequencies = np.asarray(frequencies)
    return FreqlistWaveform(frequencies=frequencies, n_samples=n_samples, sample_rate=sample_rate,
                            amplitudes=amplitudes, phases=phases, iq_ratios=iq_ratios, phase_offsets=phase_offsets,
                            seed=seed, maximize_dynamic_range=maximize_dynamic_range, compute=compute)


class FLConfigMixin:
    _settings = tuple()

    def __ge__(self, other):
        """ Returns true if self is at least as specified as other (i.e. other has = or more Nones"""
        if other is None:
            return True
        if not isinstance(other, type(self)):
            raise ValueError(f'Invalid type: {type(self)}')
        if self.hashed or other.hashed:
            raise ValueError('Unable to compare hashed configs.')
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if b is not None and a is None:
                return False
        return True

    def __eq__(self, other):
        """ Feedline configs are equivalent if all of their settings are equivalent"""
        if not isinstance(other, type(self)):
            return False
        if self.hashed or other.hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if not self._hashdata_vals_equal(a,b):
                return False
        return True

    def __le__(self, other):
        if other is None:
            return False
        return other.__ge__(self)

    def __lt__(self, other):
        """ Returns true if self other is more specified than self (i.e. self has more Nones) """
        # if other is None:
        #     return False
        return not self.__ge__(other) #and not self == other

    def __gt__(self, other):
        if other is None:
            return True
        return self.__ge__(other) and not self == other

    @staticmethod
    def _hashdata_vals_equal(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and not np.allclose(a, b):
            return False
        elif a != b:
            return False
        return True

    def compatible_with(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.hashed or other.hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if a is None or b is None:
                continue
            if not self._hashdata_vals_equal(a,b):
                return False
        return True

    def deltafy(self, other):
        """ Return a new config with only the _settings that went from None to a value, all else None """
        d = {k: getattr(other, k) for k in self._settings if getattr(self, k) is None and getattr(other, k) is not None}
        return type(self)(**d)

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
    def compatible_with(self, other):
        """ Feedline configs are equivalent if all of their settings are equivalent"""
        if other is None:
            return True
        for k, v in self:
            other_v = getattr(other, k)
            assert isinstance(v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            assert isinstance(other_v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            if v is None or other_v is None:
                continue
            if not v.compatible_with(other_v):
                return False
        return True

    def __eq__(self, other):
        """ Feedline configs are equivalent if all of their settings are equivalent"""
        if other is None:
            return False

        for k, v in self:
            other_v = getattr(other, k)
            assert isinstance(v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            assert isinstance(other_v, (FLMetaConfigMixin, FLConfigMixin, type(None), int))
            # Compute hash if necessary
            v_hash = hash(v) if isinstance(v, FLMetaConfigMixin) else v
            o_hash = hash(other_v) if isinstance(other_v, FLMetaConfigMixin) else other_v
            if not v_hash == o_hash:
                return False
        return True

    def __gt__(self, other):
        if other is None:
            return True
        return self.__ge__(self) and not self==other

    def __lt__(self, other):
        return not self.__ge__(other)

    def __ge__(self, other):
        """Self is more specified or equal to other """
        if other is None:
            return True
        for k, v in self.iter():
            ov = getattr(other, k)
            if v is None and ov is not None:
                return False
            if v is None and ov is None:
                continue
            if not v >= ov:
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
                if v is None:
                    yield k, v
                elif hashed and v.hashed:
                    yield k, v
                elif unhashed and not v.hashed:
                    yield k, v


class ADCconfig(FLConfigMixin):
    _settings = tuple()

    def __init__(self, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return




class DACConfig(FLConfigMixin):
    _settings = ('output_waveform', 'qmc_settings', 'fpgen')

    def __init__(self, output_waveform=None, fpgen=None, qmc_settings=None, _hashed=None, **waveform_spec):
        self._hashed = _hashed
        if self._hashed:
            return

        # TODO [optional] make hash use waveform_spec instead of output_waveform so that deferred computation isn't
        # triggered by a request for the hash
        waveform_spec['output_waveform'] = output_waveform
        waveform_spec['n_samples'] = 2 ** 19
        waveform_spec['sample_rate'] = 4.096e9
        self._waveform = WaveformFactory(**waveform_spec)
        self.fpgen = self._waveform.fpgen
        self.qmc_settings = qmc_settings

    @property
    def output_waveform(self):
        """This is a property so that compute=False is respected"""
        return self._waveform.output_waveform



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

    def __init__(self, holdoffs: np.ndarray = None, thresholds: np.ndarray = None, _hashed=None):
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

    def __init__(self, tones=None, loop_center=None, phase_offset=None, center_relative=None,
                 quantize=None, _hashed=None):
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
        # TODO to support captures of less than all groups we need to add a self.capture_setup which has group
        # settings for iq and phase

    def __str__(self):
        pp = str(self.pp_config).replace('\n  ', '\n    ')
        return (f"FeedlineConfig {hash(self)}:\n"
                f"  IF: {self.if_config}\n"
                f"  DAC: {self.dac_config}\n"
                f"  ADC: {self.adc_config}\n"
                f"  PP: {pp}")


class FeedlineConfigManager:
    def __init__(self):
        self._config = {}
        self._cache = {}

    def learn(self, config: FeedlineConfig):
        """Commit configuration info to memory for later use, hashed configurations are not learnable"""
        self._cache.update({hash(v): v for k, v in config.iter(hashed=False)})  # if hash(v) not in self._cache})

    def unlearned_hashes(self, config: FeedlineConfig):
        """
        Return a set of any hashed config hashes in the config that have not been learned.

        The presence of unlearned hashes will prevent a config from being added to the manager.
        """
        return set(hash(v) for k, v in config.iter(unhashed=False) if hash(v) not in self._cache)

    def required(self) -> FeedlineConfig:
        """
        Return a feedline configuration resulting from settings in the set, the config will not contain any hashed
        settings.
        """
        setting_dict = defaultdict(lambda: defaultdict(dict))

        for config in self._config.values():
            for k, v in config:
                if isinstance(v, FLMetaConfigMixin):
                    for k2, v2 in v:
                        setting_dict[k][k2].update(v2.settings_dict(unhasher_cache=self._cache))
                else:
                    setting_dict[k].update(v.settings_dict(unhasher_cache=self._cache))
        return FeedlineConfig(**setting_dict)

    def pop(self, id) -> bool:
        """
        Remove a config from the manager

        Args:
            id: settings id to remove, does nothing if it is unknown

        Returns: True iff the required settings after the pop are less restrictive
        """
        if id not in self._config:
            return False
        x = self.required()
        self._config.pop(id)
        return self.required() < x

    def add(self, id, config: FeedlineConfig) -> FeedlineConfig:
        """
        Add a feedline config to the managed set of settings with an id for later removal. Configs with hashed
        settings can not be added unless their hashes have been previously learned. All unhashed settings are learned
        upon addition.

        If the config to be added has settings that are not compatible with existing settings it an error will be
        raised.

        Args:
            id: an id for the settings (for later removal from the set)
            config: a feedline config at any level of specificity

        Returns: A feedline config with the settings that need updating populated and the rest None

        Raises: ValueError if a config contains unlearned hashes.

        """
        self.learn(config)
        if self.unlearned_hashes(config):
            raise ValueError('Config contains unlearned hashes')

        old = self.required()
        if not old.compatible_with(config):
            raise ValueError('Proposed settings not compatible with required settings')
        self._config[id] = config
        new = self.required()

        # NB >= is read as "at more or equally restrictive" on the needed FPGA settings
        # so IFConfig(dac_atten=3) and IFConfig(dac_atten=4) are equally restrictive but not equal, not compatible
        # hence the need for the compatible_with check above. IFConfig(dac_atten=3) is more restrictive than
        # IFConfig(dac_atten=None) and thus no settings changes would be needed.

        for k, v in new:
            ov = getattr(old, k)
            if ov is None:
                continue
            if isinstance(v, FLMetaConfigMixin):
                for k2, v2 in v:
                    ov2 = getattr(ov, k2)
                    if ov2 >= v2:
                        setattr(v, k2, None)
                    else:
                        setattr(v, k2, ov2.deltafy(v2))
            elif ov >= v:
                setattr(new, k, None)
            else:
                setattr(new, k, ov.deltafy(v))
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

    def __init__(self, n, tap, feedline_config: FeedlineConfig, feedline_server=None):
        self.nsamp = n  # n is treated as the buffer time in ms for photons, and has limits enforced by the driver
        self._last_status = None
        self.tap = tap  # maybe add some error handling here
        self.feedline_config = feedline_config
        self._feedline_server = feedline_server
        self._status_socket = None
        self._data_socket = None
        self._established = False

    @property
    def feedline_server(self):
        return self._feedline_server

    @feedline_server.setter
    def feedline_server(self, server):
        if self._feedline_server is not None:
            raise RuntimeError('Once set the server may not be changed.')
        self._feedline_server = server

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

    def fail(self, message, raise_exception=False):
        if self.completed:
            return
        try:
            self._data_socket.send_multipart([self.id, b''])
            self._send_status('failed', message)
            self.destablish()
        except zmq.ZMQError as ez:
            getLogger(__name__).warning(f'Failed to send fail message {self} due to {ez}')
            if raise_exception:
                raise ez

    def finish(self, raise_exception=True):
        if self.completed:
            return
        try:
            self._data_socket.send_multipart([self.id, b''])
            self._send_status('finished')
            self.destablish()
        except zmq.ZMQError as ez:
            getLogger(__name__).warning(f'Failed to send finished message {self} due to {ez}')
            if raise_exception:
                raise ez

    def abort(self, message, raise_exception=False):
        if self.completed:
            return
        try:
            self._data_socket.send_multipart([self.id, b''])
            self._send_status('aborted', message)
            self.destablish()
        except zmq.ZMQError as ez:
            getLogger(__name__).warning(f'Failed to send abort message {self} due to {ez}')
            if raise_exception:
                raise ez

    def add_data(self, data, status='', copy=False):
        if not self._data_socket or not self._status_socket:
            raise RuntimeError('Establish must be called before add_data')
        # TODO ensure we are being smart about pointers and buffer access vs copys

        #TODO how do we ship the various data formats?
        if not isinstance(data, np.ndarray):
            data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if copy:
            data = np.array(data)

        self._data_socket.send_multipart([self.id, blosc2.compress(data)], copy=False)
        self._send_status('capturing', status)

    def _send_status(self, status, message=''):
        if not self._status_socket:
            raise RuntimeError('No status_socket connection available')
        update = f'{status}:{message}'
        getLogger(__name__).debug(f'Published status update {self.id}: "{update}"')
        self._last_status = (status, message)
        self._status_socket.send_multipart([self.id, update.encode()])

    @property
    def completed(self):
        return self._last_status is not None and self._last_status[0] in ('finished', 'aborted', 'failed')

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
        return self.nsamp, self.nchan, self.dwid // 2


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
