from mkidgen3.funcs import *
import logging

from mkidgen3.system_parameters import ADC_DAC_INTERFACE_WORD_LENGTH, DAC_RESOLUTION, DAC_SAMPLE_RATE, SYSTEM_BANDWIDTH


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
        self._sample_rate = sample_rate

    def __str__(self):
        return f'TabulatedWaveform with fpgen {self._fpgen}'

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

    def __repr__(self):
        return f'<{str(self)}>'

    def __str__(self):
        preview_dict = {'freqs':self.freqs, 'amps':self.amps, 'phases': self.phases,
                        'iq_ratios': self.iq_ratios, 'phase_offsets': self.phase_offsets,
                        'quant_error': self.quant_error}
        for key, value  in preview_dict.items():
            if value is None or (value.size < 3):
                preview_dict[key] = value
            else:
                preview_dict[key] = value[:3]

        return f'FreqlistWaveform: {preview_dict}'

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
        frequencies = uniform_freqs(n_uniform_tones, bandwidth=SYSTEM_BANDWIDTH)
    if frequencies is None:
        return None
    frequencies = np.asarray(frequencies)
    return FreqlistWaveform(frequencies=frequencies, n_samples=n_samples, sample_rate=sample_rate,
                            amplitudes=amplitudes, phases=phases, iq_ratios=iq_ratios, phase_offsets=phase_offsets,
                            seed=seed, maximize_dynamic_range=maximize_dynamic_range, compute=compute)
