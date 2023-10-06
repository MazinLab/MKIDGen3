import numpy as np

from hashlib import md5
from collections import defaultdict
import copy
from .waveform import WaveformFactory


def _hasher(v, pass_none=False):
    """
    Hash a value

    Args:
        v: the value to be hashed, the md5 hexdigest of the bytestring of v is used.
        pass_none: If true None through without hashing, otherwise the string '___python_None' is used in its place.

    Returns: The hash
    """
    if v is None and pass_none:
        return None

    try:
        if v is None:
            v = '___python_None'
        return md5(str(v).encode()).hexdigest()
    except TypeError:
        return md5(v.tobytes()).hexdigest()


class _FLConfigMixin:
    _settings = tuple()

    def __ge__(self, other):
        """ Returns true if self is at least as specified as other (i.e. other has = or more Nones)"""
        if other is None:
            return True
        if not isinstance(other, type(self)):
            raise ValueError(f'Invalid type: {type(self)}')
        if self.is_hashed or other.is_hashed:
            raise ValueError('Unable to compare hashed configs.')
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if b is not None and a is None:
                return False
        return True

    def __eq__(self, other):
        """
        Configs are equivalent if all of their settings are the equ.

        If either is hashed then their hashes must match exactly. Otherwise the data

        """
        if not isinstance(other, type(self)):
            return False
        if self.is_hashed or other.is_hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if not self._hashdata_vals_equal(a, b):
                return False
        return True

    def __le__(self, other):
        if other is None:
            return False
        return other.__ge__(self)

    def __lt__(self, other):
        """ Returns true if other is more specified than self (i.e. self has more Nones) """
        # if other is None:
        #     return False
        return not self.__ge__(other)  # and not self == other

    def __gt__(self, other):
        if other is None:
            return True
        return self.__ge__(other) and not self == other

    @staticmethod
    def _hashdata_vals_equal(a, b) -> bool:
        """Compare settings values for equality"""
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and not np.allclose(a, b):
            return False
        elif a != b:
            return False
        return True

    def merge_with(self, other):
        """Adopt the specified values of the other"""
        assert isinstance(other, type(self)), 'other must be of the same type'
        assert not self.is_hashed and not other.is_hashed, 'Hashed FLConfigs can not be merged'
        for k in self._settings:
            v = getattr(other, k)
            if v is not None:
                setattr(self, k, v)

    def compatible_with(self, other) -> bool:
        """
        Determine if two FLConfig's settings are compatible with one another.

        The present implementation is maximally conservative: all settings much match, None matches anything
        (see _hashdata_vals_equal for details of equality)
        Args:
            other: Another FLConfig

        Returns: True if the two configs' settings can coexist in hardware
        """
        if not isinstance(other, type(self)):
            return False
        if self.is_hashed or other.is_hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if a is None or b is None:
                continue
            if not self._hashdata_vals_equal(a, b):
                return False
        return True

    def deltafy(self, other):
        """
        Generate a new config with only the settings that specified by other but not self, all other settings are None.

        No comparison of setting values is made.
        """
        d = {k: getattr(other, k) for k in self._settings if getattr(self, k) is None and getattr(other, k) is not None}
        return type(self)(**d)

    @property
    def is_hashed(self) -> bool:
        return self._hashed is not None

    @property
    def _hash_data(self) -> tuple:
        """Return a tuple of (setting_key, hashed_data|None) pairs sorted on the setting key"""
        hash_data = ((k, _hasher(getattr(self, k), pass_none=True)) for k in self._settings)
        return tuple(sorted(hash_data, key=lambda x: x[0]))

    def __hash__(self):
        """
        Return a hash of the config, the same settings will always return the same hash.

        A hashed config has the same hash as an unhashed config.

        A setting set to None is different than one set to a value.
        """
        if self.is_hashed:
            return self._hashed
        return int(_hasher(self._hash_data), 16)

    def __str__(self):
        """Nicely format one's self"""
        name = self.__class__.__name__
        if self._hashed:
            return f"{name}: {hash(self)} (hashed)"
        else:
            return (f"{name}: {hash(self)}\n"
                    f"  {self.settings_dict()}")

    @property
    def hashed_form(self):
        """
        Return a new config of the same type in hashed form

        In hashed form the settings values are used to compute a hash and it is stored instead of the settings values.
        """
        return type(self)(_hashed=hash(self))

    def settings_dict(self, omit_none=True, _hashed=False, unhasher_cache=None) -> dict:
        """
        Build a dict of setting_keys:values for use in creating a clone of the config or
        passing to a drivers configure() method

        Args:
            omit_none: Exclude settings with no set value from the dictionary
            _hashed: for internal use building a hashed FLConfig, if True omit_none and unhasher_cache are ignored
            unhasher_cache: May be set to a dictionary to fetch actual values in liu of the hash.
                Required if calling and self.is_hashed is true. hash(self) is expected to be in the unhasher_cache.

        Returns: A dictionary of setting_key:value pairs, keys that are none may or may not be present.

        Raises: AssertionError if called on a hashed config without and unhasher_cache containing the hash.
        Will not be raised if _hashed is set.
        """
        if self._hashed and not _hashed:
            assert isinstance(unhasher_cache, dict), ('An unhasher_cache is required for to build a settings '
                                                      'dictionary from a hashed FLConfig')
            assert self._hashed in unhasher_cache, ('A hash entry in the unhasher_cache is required for to build a '
                                                    'settings dictionary from a hashed FLConfig')
            x = unhasher_cache[self._hashed]
        else:
            x = self
        if _hashed:
            d = {'_hashed': hash(self)}
        else:
            d = {k: getattr(x, k) for k in self._settings if not (getattr(x, k) is None and omit_none)}
        return d


class _FLMetaconfigMixin:
    """
    Like FLConfig the FLMetaconfig is a collection of settings for the PL and FL hardware (i.e. inclusive of
    clocking, and IF boards).

    All non-internal attributes are assumed to be instances of FLMetaConfigMixin or FLConfigMixin
    """

    def compatible_with(self, other):
        """
        Determine if two FLConfig's settings are compatible with one another.

        The present implementation is maximally conservative: all settings much match, None matches anything
        Args:
            other: Another FLConfig

        Returns: True if the two configs' settings can coexist in hardware
        """
        if other is None:
            return True
        for k, v in self:
            other_v = getattr(other, k)
            assert isinstance(v, (_FLMetaconfigMixin, _FLConfigMixin, type(None), int))
            assert isinstance(other_v, (_FLMetaconfigMixin, _FLConfigMixin, type(None), int))
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
            assert isinstance(v, (_FLMetaconfigMixin, _FLConfigMixin, type(None), int))
            assert isinstance(other_v, (_FLMetaconfigMixin, _FLConfigMixin, type(None), int))
            # Compute hash if necessary
            v_hash = hash(v) if isinstance(v, _FLMetaconfigMixin) else v
            o_hash = hash(other_v) if isinstance(other_v, _FLMetaconfigMixin) else other_v
            if not v_hash == o_hash:
                return False
        return True

    def __gt__(self, other):
        if other is None:
            return True
        return self.__ge__(self) and not self == other

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
                v = '___python_None'  # we want None to hash to the same value always
            return md5(str(v).encode()).hexdigest()

        return int(hasher(tuple(sorted(((k, hasher(v)) for k, v in self.__dict__.items()), key=lambda x: x[0]))), 16)

    def __iter__(self):
        for v in sorted(vars(self)):
            if v.startswith('_'):
                continue
            yield v, getattr(self, v)

    def merge_with(self, other):
        """Adopt the specified values of the other"""
        assert isinstance(other, type(self)), 'other must be of the same type'
        for (_,v1), (_,v2) in zip(self, other):
            v1.merge_with(v2)

    @property
    def hashed_form(self):
        """
        Return a new config of the same type in hashed form

        In hashed form setting's that are FLConfigs are converted to their hashed form. This is done recursively
        through any keys that are FLMetaconfigs.
        """
        d = {k: v if v is None else v.settings_dict(_hashed=True) for k, v in self}
        return type(self)(**d)

    def settings_dict(self, _hashed=False, unhasher_cache=None, omit_none=True):
        """
        Build a dict of setting_keys:values for use in creating a clone of the config or
        passing to a drivers configure() method. The values of any settings that are FLMetaconfigs will
        themselves be dictionaries.

        Args:
            omit_none: Exclude settings with no set value from the dictionary
            _hashed: for internal use building a hashed FLConfig, if True omit_none and unhasher_cache are ignored
            unhasher_cache: May be set to a dictionary to

        Returns: A dictionary of setting_key:value pairs, values that are None may or may not be present. Values that
        are FLConfigs will be dictionaries
        """
        skip_none = omit_none
        if _hashed:
            skip_none = False
        return {k: v if v is None else v.settings_dict(omit_none=omit_none, _hashed=_hashed,
                                                       unhasher_cache=unhasher_cache)
                for k, v in self if not (v is None and skip_none)}

    def iter(self, hashed=True, unhashed=True):  # -> (str,FLConfigMixin|None):
        """
        a iterator of config_key: value pairs including any nested meta configs

        nested meta config's config keys are yielded as key.child_key, note that nothing prevents a child
        from also having meta config so.this.is.a.possible.key

        for example 'pp_config.trig_config', TrigerConfig|None
        """
        for k, v in self:
            if isinstance(v, _FLMetaconfigMixin):
                for a, b in v.iter(hashed=hashed, unhashed=unhashed):
                    yield f'{k}.{a}', b
            else:
                if v is None:
                    yield k, v
                elif hashed and v.is_hashed:
                    yield k, v
                elif unhashed and not v.is_hashed:
                    yield k, v


class ADCConfig(_FLConfigMixin):
    _settings = tuple()

    def __init__(self, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return


class IFConfig(_FLConfigMixin):
    _settings = ('lo', 'adc_attn', 'dac_attn')

    def __init__(self, lo=None, adc_attn=None, dac_attn=None, _hashed=None):
        """
        Args:
            lo: lo position in MHz
            adc_attn: output attenuation in dB. Max is 40.
            dac_attn: input attenuation in dB. Max is 40.
            _hashed:
        """
        self._hashed = _hashed
        if self._hashed:
            return

        self.lo = lo
        self.adc_attn = adc_attn
        self.dac_attn = dac_attn


class TriggerConfig(_FLConfigMixin):
    _settings = ('holdoffs', 'thresholds')

    def __init__(self, holdoffs: np.ndarray = None, thresholds: np.ndarray = None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.holdoffs = holdoffs
        self.thresholds = thresholds


class ChannelConfig(_FLConfigMixin):
    _settings = ('frequencies',)

    def __init__(self, frequencies=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.frequencies = frequencies


class DDCConfig(_FLConfigMixin):
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


class FilterConfig(_FLConfigMixin):
    _settings = ('coefficients',)

    def __init__(self, coefficients=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return

        self.coefficients = coefficients


class DACConfig(_FLConfigMixin):
    _settings = ('output_waveform', 'qmc_settings', 'fpgen')

    def __init__(self, output_waveform=None, qmc_settings=None, _hashed=None, **waveform_spec):
        self._hashed = _hashed
        if self._hashed:
            return

        # TODO [optional] make hash use waveform_spec instead of output_waveform so that deferred computation isn't
        #  triggered by a request for the hash
        waveform_spec['output_waveform'] = output_waveform
        waveform_spec['n_samples'] = 2 ** 19
        waveform_spec['sample_rate'] = 4.096e9
        self._waveform = WaveformFactory(**waveform_spec)
        self.fpgen = self._waveform.fpgen if self._waveform is not None else None
        self.qmc_settings = qmc_settings

    @property
    def output_waveform(self):
        """This is a property so that compute=False is respected"""
        return self._waveform.output_waveform if self._waveform is not None else None

    @property
    def default_channel_config(self)->ChannelConfig:
        """A convenience method to get a ChannelConfig using all the waveform's frequencies"""
        return ChannelConfig(frequencies=self._waveform.freqs)


class PhotonPipeConfig(_FLMetaconfigMixin):
    def __init__(self, chan_config: (dict, ChannelConfig) = None, ddc_config: (dict, DDCConfig) = None,
                 filter_config: (dict, FilterConfig) = None, trig_config: (dict, TriggerConfig) = None):
        self.chan_config = ChannelConfig(**chan_config) if isinstance(chan_config, dict) else chan_config
        self.ddc_config = DDCConfig(**ddc_config) if isinstance(ddc_config, dict) else ddc_config
        self.trig_config = TriggerConfig(**trig_config) if isinstance(trig_config, dict) else trig_config
        self.filter_config = FilterConfig(**filter_config) if isinstance(filter_config, dict) else filter_config

    @staticmethod
    def empty_config():
        return PhotonPipeConfig(chan_config=ChannelConfig(), ddc_config=DDCConfig(), filter_config=FilterConfig(),
                                trig_config=TriggerConfig())

    def __str__(self):
        return (f"PhotonPipe {hash(self)}:\n"
                f"  Chan: {self.chan_config}\n"
                f"  DDC: {self.ddc_config}\n"
                f"  Filt: {self.filter_config}\n"
                f"  Trig: {self.trig_config}")


class FeedlineConfig(_FLMetaconfigMixin):
    @staticmethod
    def empty_config():
        return FeedlineConfig(if_config=IFConfig(), dac_config=DACConfig(),
                              pp_config=PhotonPipeConfig.empty_config(), adc_config=ADCConfig())

    def __init__(self, if_config: (dict, IFConfig) = None, dac_config: (DACConfig, dict) = None,
                 pp_config: (dict, PhotonPipeConfig) = None, adc_config: (dict, ADCConfig) = None):
        self.if_config = IFConfig(**if_config) if isinstance(if_config, dict) else if_config
        self.dac_config = DACConfig(**dac_config) if isinstance(dac_config, dict) else dac_config
        self.pp_config = PhotonPipeConfig(**pp_config) if isinstance(pp_config, dict) else pp_config
        self.adc_config = ADCConfig(**adc_config) if isinstance(adc_config, dict) else adc_config
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
        """Commit configuration info to memory for later use, hashed configurations do not add to knowledge"""
        self._cache.update({hash(v): v for _, v in config.iter(hashed=False)})  # if hash(v) not in self._cache})

    def unlearned_hashes(self, config: FeedlineConfig) -> set:
        """
        Return a set of any hashed config hashes in the config that have not been learned.

        The presence of unlearned hashes will prevent a config from being added to the manager.
        """
        return set(hash(v) for k, v in config.iter(unhashed=False) if hash(v) not in self._cache)

    def required(self) -> FeedlineConfig:
        """
        Return the feedline configuration built from all FeedlineConfigs in the manager. The config will be
        free of hashed settings but may not specify a requirement for many settings (i.e. they may be None)

        """
        # This was the original approach and should be correct
        # settings = defaultdict(lambda: defaultdict(dict))
        # for config in self._config.values():
        #     for k, v in config:
        #         if isinstance(v, _FLMetaconfigMixin):
        #             for k2, v2 in v:
        #                 assert not isinstance(v2, _FLMetaconfigMixin)
        #                 if v2 is not None:
        #                     settings[k][k2].update(v2.settings_dict(unhasher_cache=self._cache,
        #                                                             _hashed=False, omit_none=True))
        #         elif v is not None:
        #             settings[k].update(v.settings_dict(unhasher_cache=self._cache, _hashed=False, omit_none=True))
        # return FeedlineConfig(**settings)

        #This is a new, much cleaner approach
        cfg = FeedlineConfig.empty_config()
        for config in self._config.values():
            # This adopts values specified in config (i.e. Nones are skipped). While it later configs set values are
            # overwritten
            cfg.merge_with(config)  # there is a potential concurrency issue here if the configs are being mutated

        return copy.deepcopy(cfg)  # ensure the result doesn't share data!

        #Yet another approach
        # settings = defaultdict(lambda: dict)  # We need to build up a nested set of dicts to be used as kwargs
        # for config in self._config.values():  # iterate through all the FeedlineConfigs in the pot
        #     specified_settings = config.settings_dict(unhasher_cache=self._cache, _hashed=False, omit_none=True)
        #     # this is a dict (of dicts maybe of dicts... )
        #     # we need to merge them all with update all the way down,
        #     # the caveat is that if a FLConfig has a setting that is itself a dict
        #     # it should be treated as a value and simple recursion does not allow us to know which it is
        #     ....


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
            if isinstance(v, _FLMetaconfigMixin):
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
