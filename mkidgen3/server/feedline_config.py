import logging

import numpy as np
from hashlib import md5
import copy
from mkidgen3.util import convert_freq_to_ddc_bins
from logging import getLogger





def _hasher(v, pass_none=False):
    """
    Hash a value

    Args:
        v: the value to be hashed, the md5 hexdigest of the bytestring of v is used. .tobytes() will be used if
        present otherwise str().encode() is used. hash() is called on _FLConfigMixin or _FLMetaonfigMixin objects.
        pass_none: If true, None is passed through without hashing, otherwise the string '___python_None' is
        used in its place.

    Returns: The hash
    """
    if v is None:
        return None if pass_none else md5(str('___python_None').encode()).hexdigest()
    elif isinstance(v, (_FLConfigMixin, _FLMetaconfigMixin)):
        return hash(v)
    else:
        try:
            v = v.tobytes()
        except AttributeError:
            v = str(v).encode()
        return md5(v).hexdigest()


class _FLConfigMixin:
    _settings = tuple()

    def __eq__(self, other):
        """
        Configs are equivalent if all of their settings are the equ.

        If either is hashed then their hashes must match exactly. Otherwise the data

        """
        if other is None:
            return self.empty()
        if not isinstance(other, type(self)):
            return False
        if self.is_hashed or other.is_hashed:
            return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if not self._hashdata_vals_equal(a, b):
                return False
        return True

    def __ge__(self, other):
        """ Returns true if self is at least as specified as other (i.e. other has = or more Nones)"""
        if other is None:
            return True
        if not isinstance(other, type(self)):
            raise TypeError(f'Invalid type: {type(other)}')
        if self.is_hashed or other.is_hashed:
            raise ValueError('Unable to compare hashed configs.')
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if b is not None and a is None:
                return False
        return True

    def __le__(self, other):
        if other is None:
            return self.empty()
        if not isinstance(other, type(self)):
            raise TypeError('Incorrect type for config comparison')
        return other.__ge__(self)

    def __lt__(self, other):
        """ Returns true if other is more specified than self (i.e. self has more Nones) """
        return not self.__ge__(other)

    def __gt__(self, other):
        return self.__ge__(other) and not self == other

    def __bool__(self):
        """ True iff not empty i.e. specifies config settings """
        return not self.empty()

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

    @property
    def _hash_data(self) -> tuple:
        if self.is_hashed:
            raise RuntimeError('Not supported for hashed FLConfig')
        """Return a tuple of (setting_key, hashed_data|None) pairs sorted on the setting key"""
        hash_data = ((k, _hasher(getattr(self, k), pass_none=True)) for k in self._settings)
        return tuple(sorted(hash_data, key=lambda x: x[0]))

    def merge_with(self, other):
        """Adopt the specified values of the other"""
        if other is None:
            return
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
        if other is None:
            return True
        if not isinstance(other, type(self)):
            return False
        if other.is_hashed:
            if self.empty():
                return True
            else:
                return hash(self) == hash(other)
        for (_, a), (_, b) in zip(self._hash_data, other._hash_data):
            if a is None or b is None:
                continue
            if not self._hashdata_vals_equal(a, b):
                return False
        return True

    def deltafy(self, other):
        """
        Generate a new config with only the settings that are specified by other but not self, or have changed in other
        from self, all other settings are None.

        """
        d = {}
        if other is not None:
            for k in self._settings:
                sv = getattr(self, k)
                ov = getattr(other, k)
                if ov is not None:
                    if sv is None:
                        d[k] = ov
                        continue
                    if isinstance(ov, np.ndarray):
                        try:
                            if not (sv==ov).all():
                                d[k] = ov
                        except ValueError:
                            getLogger(__name__).warning(f'Array data comparison issue in {type(self)}.{k}.'
                                                        f'Check config definitions.')
                            d[k] = ov
                    elif sv != ov:
                        d[k] = ov
        return type(self)(**d)

    def empty(self):
        """
        Return true if all the settings are unspecified (e.g. None)
        """
        if self.is_hashed:
            return hash(self) == hash(type(self)())
        return not self.settings_dict(omit_none=True)

    @property
    def is_hashed(self) -> bool:
        return self._hashed is not None

    @property
    def hashed_form(self):
        """
        Return a new config of the same type in hashed form

        In hashed form the settings values are used to compute a hash, and it is stored instead of the settings values.
        """
        return type(self)(_hashed=hash(self))

    def unhashed_form(self, hash_cache: dict):
        if not self.is_hashed:
            return self
        d = self.settings_dict(unhasher_cache=hash_cache, _hashed=False, omit_none=True)
        return type(self)(**d)

    def settings_dict(self, omit_none=True, _hashed=False, unhasher_cache=None) -> dict:
        """
        Build a dict of setting_keys:values for use in creating a clone of the config or
        passing to a drivers configure() method

        TODO how does a hanshed None setting (is this possible?) interact with omit_none

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
            d = {k: copy.deepcopy(getattr(x, k)) for k in self._settings if not (getattr(x, k) is None and omit_none)}
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
        if not isinstance(other, type(self)):  #NB a hased config is still a config, not an int for a metaconfig
            return False
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
        """
        Feedline configs are == if all of their settings are the same.
        NB: None is compatible with anything but not considered the same however None is == a config with all None settings
        """
        for k, v in self:
            other_v = None if other is None else getattr(other, k)
            assert isinstance(v, (_FLConfigMixin, type(None)))
            assert isinstance(other_v, (_FLConfigMixin, type(None)))
            if v is None and other_v is not None:
                if not other_v.empty():
                    return False
            elif not v == other_v:
                return False
        return True

    def __ge__(self, other):
        """Self is more specified or equal to other """

        for k, v in self:
            ov = getattr(other, k)
            if v is None and ov is not None and not ov.empty():
                return False
            if not v >= ov:
                return False
        return True

        # for k, v in self.iter():
        #     ov = getattr(other, k)
        #     if v is None and ov is not None:
        #         return False
        #     if v is None and ov is None:
        #         continue
        #     if not v >= ov:
        #         return False
        # return True

    def __le__(self, other):
        if other is None:
            return self.empty()
        if not isinstance(other, type(self)):
            raise TypeError('Incorrect type for config comparison')
        return other.__ge__(self)

    def __lt__(self, other):
        return not self.__ge__(other)

    def __gt__(self, other):
        return self.__ge__(other) and not self == other

    def __hash__(self):
        return int(_hasher(tuple(sorted(((k, _hasher(v)) for k, v in self.__dict__.items()), key=lambda x: x[0]))), 16)

    def __bool__(self):
        """ True iff not empty i.e. specifies config settings """
        return not self.empty()

    def __iter__(self):
        """
        An iterator of config_key: value pairs
        """
        for v in sorted(vars(self)):
            if v.startswith('_'):
                continue
            yield v, getattr(self, v)

    def empty(self):
        for _, v in self:
            if v is None or v.empty():
                continue
            return False
        return True

    def merge_with(self, other):
        """Adopt the specified values of the other"""
        if other is None:
            return
        assert isinstance(other, type(self)), 'other must be of the same type'
        for (_, v1), (_, v2) in zip(self, other):
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

    @property
    def is_hashed(self):
        for _, v in self:
            if v is not None and v.is_hashed:
                return True
        return False

    def unhashed_form(self, hash_cache: dict):
        if not self.is_hashed:
            return self
        d = self.settings_dict(unhasher_cache=hash_cache, _hashed=False, omit_none=True)
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
        An iterator of config_key: value pairs including any nested meta configs

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


class BitstreamConfig(_FLConfigMixin):
    _settings = ('bitstream', 'ignore_version')

    def __init__(self, bitstream: str | None = None, ignore_version: bool | None = None, _hashed=None):
        """
        Args:
            bitstream: bitstream file name (full path)
            ignore_version: suppress errors related to differences in IP release verison and what driver expects
            _hashed:
        """
        self._hashed = _hashed
        if self._hashed:
            return
        self.bitstream = bitstream
        self.ignore_version = ignore_version


class RFDCClockingConfig(_FLConfigMixin):
    _settings = ('programming_key', 'clock_source')

    def __init__(self, programming_key: str | None = None, clock_source: str | None = None, _hashed=None):
        """
        Args:
            programming_key: RFDC LMX / LMK programming files
            clock_source: internal or external 10 MHz source #TODO: this should probably be made a bool
            _hashed:
        """
        self._hashed = _hashed
        if self._hashed:
            return
        self.programming_key = programming_key
        self.clock_source = clock_source


class RFDCConfig(_FLConfigMixin):
    _settings = ('dac_mts', 'adc_mts', 'adc_gains', 'dac_gains')

    def __init__(self, dac_mts: bool | None = None, adc_mts: bool | None = None, adc_gains: (float, float) = None,
                 dac_gains: (bool, bool) = None, _hashed=None):
        """
        Args:
            dac_mts: enable dac MTS
            adc_mts:  enable adc MTS (also enables DAC MTS)
            adc_gains: ADC0, ADC1. Allowed values 0-2.0
            dac_gains: DAC0, DAC1. Allowed values 0-2.0
            _hashed:
        """
        self._hashed = _hashed
        if self._hashed:
            return
        self.dac_mts = dac_mts
        self.adc_mts = adc_mts
        self.adc_gains = adc_gains
        self.dac_gains = dac_gains


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

    def __init__(self, holdoffs: np.ndarray | list = None, thresholds: np.ndarray | list = None, _hashed=None):
        """ See drivers.PhotonTrigger for notes on what these values should be"""
        self._hashed = _hashed
        if self._hashed:
            return

        self.holdoffs = np.asarray(holdoffs) if holdoffs is not None else None
        self.thresholds = np.asarray(thresholds) if thresholds is not None else None


class ChannelConfig(_FLConfigMixin):
    _settings = ('bins',)

    def __init__(self, freqs=None, bins=None, _hashed=None):
        self._hashed = _hashed
        if self._hashed:
            return
        if freqs is not None:
            bins = convert_freq_to_ddc_bins(freqs)
        self.bins = bins


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


class WaveformConfig(_FLConfigMixin):
    _settings = ('waveform',)

    def __init__(self, _hashed=None, waveform=None):
        """

        Args:
            qmc_settings:
            _hashed:
            waveform_values:
            waveform_spec:
            if waveform values and waveform_spec are both passed, they should be consistent
            waveform values will take precedence.
        """
        self._hashed = _hashed
        if self._hashed:
            return
        # TODO [optional] make hash use waveform_spec instead of output_waveform so that deferred computation isn't
        #  triggered by a request for the hash
        self.waveform = waveform

    @property
    def default_channel_config(self) -> ChannelConfig:
        """A convenience method to get a ChannelConfig using all the waveform's frequencies.
        Default channel config is not available for tabulated waveforms."""
        freqs = self.waveform.freqs
        bins = convert_freq_to_ddc_bins(freqs)
        return ChannelConfig(bins=bins, freqs=freqs)

    @property
    def default_ddc_config(self) -> DDCConfig:
        """A convenience method to get a DDCConfig using all the waveform's frequencies.
        Default channel config is not available for tabulated waveforms."""
        ddc_tones = np.zeros(2048)
        ddc_tones[:self.waveform.freqs.size] = self.waveform.freqs
        centers = np.zeros(2048, dtype=np.complex64)
        phase_offsets = np.zeros(2048)
        return DDCConfig(tones=ddc_tones, phase_offset=phase_offsets, loop_center=centers)


class FeedlineConfig(_FLMetaconfigMixin):
    @staticmethod
    def empty_config():
        return FeedlineConfig(bitstream=BitstreamConfig(), rfdc_clk=RFDCClockingConfig(),
                              rfdc=RFDCConfig(), if_board=IFConfig(),
                              waveform=WaveformConfig(), chan=ChannelConfig(),
                              ddc=DDCConfig(), filter=FilterConfig(),
                              trig=TriggerConfig())

    def __init__(self, bitstream: (BitstreamConfig, dict) = None,
                 rfdc_clk: (RFDCClockingConfig, dict) = None,
                 rfdc: (RFDCConfig, dict) = None,
                 if_board: (IFConfig, dict) = None,
                 waveform: (WaveformConfig, dict) = None,
                 chan: (ChannelConfig, dict) = None,
                 ddc: (DDCConfig, dict) = None,
                 filter: (FilterConfig, dict) = None,
                 trig: (TriggerConfig, dict) = None):
        self.bitstream = BitstreamConfig(**bitstream) if isinstance(bitstream, dict) else bitstream
        self.rfdc_clk = RFDCClockingConfig(**rfdc_clk) if isinstance(rfdc_clk, dict) else rfdc_clk
        self.rfdc = RFDCConfig(**rfdc) if isinstance(rfdc, dict) else rfdc
        self.if_board = IFConfig(**if_board) if isinstance(if_board, dict) else if_board
        self.waveform = WaveformConfig(**waveform) if isinstance(waveform, dict) else waveform
        self.chan = ChannelConfig(**chan) if isinstance(chan, dict) else chan
        self.ddc = DDCConfig(**ddc) if isinstance(ddc, dict) else ddc
        self.filter = FilterConfig(**filter) if isinstance(filter, dict) else filter
        self.trig = TriggerConfig(**trig) if isinstance(trig, dict) else trig

        assert isinstance(self.bitstream, (type(None), BitstreamConfig))
        assert isinstance(self.rfdc_clk, (type(None), RFDCClockingConfig))
        assert isinstance(self.rfdc, (type(None), RFDCConfig))
        assert isinstance(self.if_board, (type(None), IFConfig))
        assert isinstance(self.waveform, (type(None), WaveformConfig))
        assert isinstance(self.chan, (type(None), ChannelConfig))
        assert isinstance(self.ddc, (type(None), DDCConfig))
        assert isinstance(self.filter, (type(None), FilterConfig))
        assert isinstance(self.trig, (type(None), TriggerConfig))
        # TODO to support captures of less than all groups we need to add a self.capture_setup which has group
        # settings for iq and phase

    def __str__(self):
        def nonestr(x, name):
            return f'{name}: None' if x is None else f'{x}'.replace('\n','\n  ')

        return (f"FeedlineConfig {hash(self)}:\n"
                f"  {nonestr(self.bitstream, 'BitstreamConfig')}\n"
                f"  {nonestr(self.rfdc_clk, 'RFDCClockingConfig')}\n"
                f"  {nonestr(self.rfdc, 'RFDCConfig')}\n"
                f"  {nonestr(self.if_board, 'IFConfig')}\n"
                f"  {nonestr(self.waveform, 'WaveformConfig')}\n"
                f"  {nonestr(self.chan, 'ChannelConfig')}\n"
                f"  {nonestr(self.ddc, 'DDCConfig')}\n"
                f"  {nonestr(self.filter, 'FilterConfig')}\n"
                f"  {nonestr(self.trig, 'TriggerConfig')}")


class FeedlineConfigManager:
    def __init__(self):
        self._config = {}
        self._cache = {}
        self._current = FeedlineConfig.empty_config()

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
        # This is a new, much cleaner approach
        cfg = FeedlineConfig.empty_config()
        for config in self._config.values():
            # This adopts values specified in config (i.e. Nones are skipped). While it later configs set values are
            # overwritten
            # there is a potential concurrency issue here if the configs are being mutated, but we don't do that!!!
            cfg.merge_with(config)
        return copy.deepcopy(cfg)  # ensure the result doesn't share data!

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
        if id in self._config and self._config[id] != config:
            raise ValueError(f'ID {id} already used by a previously added config')

        self.learn(config)
        if self.unlearned_hashes(config):
            raise ValueError('Config contains unlearned hashes')

        required = self.required()

        if not required.compatible_with(config):
            raise ValueError('Proposed settings not compatible with required settings')

        self._config[id] = config.unhashed_form(self._cache)
        new = self.required()

        # NB >= is read as "at more or equally restrictive" on the needed FPGA settings
        # so IFConfig(dac_atten=3) and IFConfig(dac_atten=4) are equally restrictive but not equal, not compatible
        # hence the need for the compatible_with check above. IFConfig(dac_atten=3) is more restrictive than
        # IFConfig(dac_atten=None) and thus no settings changes would be needed.

        for k, v in new:
            cv = getattr(self._current, k)
            if v is None or cv is None:
                continue  # our work is done or we must set for sure, either way continue

            setattr(new, k, cv.deltafy(v))  # replace with what's actually changed

        old = self._current
        self._current.merge_with(new)
        return new
