import numpy as np
#from .funcs import *

reg_waveformspec = DACOutputSpec('regular',
                                 freq = power_sweep_freqs(n_channels=N_CHANNELS, bandwidth=BANDWIDTH),
                                 amplitudes=np.ones(2048))

reg_waveformspec = DACOutputSpec('regular_normalized_comb', n_tines=200)
reg_waveform = compute_waveform(reg_waveformspec)

threetone_dacspec_unity = DACOutputSpec('3tone', [1,2,3], [1,1,1], None, None)

threetone_unity_waveform = threetone_dacspec_unity.waveform()
threetone_unity_waveform = compute_waveform(threetone_dacspec_unity)


class DACOutputSpec:
    def __init__(self, name, tones=None, amplitudes=None, phases=None, offsets=None,
                 qmc_settings=None):
        self.spec_type = name
        self.tones = tones
        self.amplitudes = amplitudes
        self.phases = phases
        self.offsets = offsets
        self.qmc_settings = qmc_settings


    def __hash__(self):
        return hash(f'{self.tones}{self.amplitudes}{self.phases}{self.offsets}{self.qmc_settings}')

    @property
    def waveform(self):
        return optimize_random_phase(self.tones, n_samples=2 ** 19, sample_rate=4.096e9,
                                                  amplitudes=None, phases=None, iq_ratios=None,
                                                  phase_offsets=None, seed=2,
                                                  max_quant_err=predict_quantization_error(),
                                                  max_attempts=10, return_quantized=True)




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