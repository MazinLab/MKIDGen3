from mkidgen3.funcs import *
import logging
import numpy as np
import numpy.typing as nt
import platform
import matplotlib.pyplot as plt

from mkidgen3.system_parameters import (ADC_DAC_INTERFACE_WORD_LENGTH, DAC_RESOLUTION, DAC_SAMPLE_RATE, SYSTEM_BANDWIDTH,
                                        DAC_FREQ_RES, DAC_FREQ_MAX, DAC_FREQ_MIN)


def _same(a, b):
    """quick test for array vs None"""
    if not isinstance(a, type(b)):
        return False
    if a is None and b is None:
        return True
    try:
        return np.all(a == b)
    except:
        pass
    try:
        return a == b
    except:
        return False


class Waveform:
    @property
    def output_waveform(self):
        """Subclasses shall implement _values """
        return self._values

    @property
    def sample_rate(self):
        """Subclasses shall implement _sample_rate """
        return self._sample_rate


class TabulatedWaveform(Waveform):
    """
    Use this class if you want to pass existing tabulated values directly to the DAC LUT without any scaling or
    computation
    """

    def __init__(self, tabulated_values=None, sample_rate=DAC_SAMPLE_RATE):
        self._values = tabulated_values
        self._sample_rate = sample_rate

    def __str__(self):
        return f'TabulatedWaveform with sample rate {self._sample_rate}'

    def __ne__(self, other):
        return not (self._sample_rate == other.sample_rate and
                    _same(self._values, other._values))


class SimpleFreqlistWaveform(Waveform):
    def __init__(
            self,
            frequencies,
            amplitudes=None,
            phases=None,
            n_samples=(1 << 19),
            sample_rate=4.096e9,
            allow_sat=False,
    ):
        self.freqs = quantize_frequencies(
            np.asarray(frequencies), rate=sample_rate, n_samples=n_samples
        )
        if amplitudes is None:
            self.amps = np.ones_like(self.freqs) / self.freqs.size
        else:
            self.amps = np.asarray(amplitudes)
        if phases is None:
            self.phases = np.random.random(self.freqs.size)*2*np.pi
        else:
            self.phases = np.asarray(phases)
        self.n_samples = n_samples
        self._sample_rate = sample_rate
        self.allow_sat = allow_sat

    @property
    def quant_freqs(self):
        return self.freqs

    @property
    def _values(self):
        fft_freqs = np.fft.fftfreq(self.n_samples, 1 / self.sample_rate)
        fft = np.zeros_like(fft_freqs, dtype=np.complex128)
        for i, f in enumerate(self.freqs):
            idx = np.argmin(np.abs(fft_freqs - f))
            fft[idx] += self.amps[i] * np.exp(1.j * self.phases[i]) * self.n_samples
        data = np.fft.ifft(fft)

        if self.allow_sat:
            data.real = np.clip(data.real, -1, 1)
            data.imag = np.clip(data.imag, -1, 1)
        else:
            if np.max(np.abs(data.real)) > 1 or np.max(np.abs(data.imag)) > 1:
                raise RuntimeError(
                    "Data exceeded DAC output range i: ({:f}, {:f}), q: ({:f}, {:f})".format(
                        np.min(data.real),
                        np.max(data.real),
                        np.min(data.imag),
                        np.max(data.imag),
                    )
                )
        return data * ((1 << 15) - 6)


class FreqlistWaveform(Waveform):
    def __init__(self, frequencies: Iterable[int | float] = None, n_samples: int = DAC_LUT_SIZE,
                 sample_rate: int | float = DAC_SAMPLE_RATE, amplitudes: Iterable[int | float] = None,
                 phases: Iterable[int | float] = None, iq_ratios: Iterable[int | float] = None,
                 phase_offsets: Iterable[int | float] = None, seed: int = 2, dac_dynamic_range: float = 1.0,
                 optimize_phase: bool = True, compute: bool = False):
        """
        Args:
            frequencies: frequencies in the waveform [Hz]
            n_samples: number of complex samples in the waveform
            sample_rate: waveform sample rate [Hz] (should be the DAC sample rate)
            amplitudes: amplitudes of each tone in the waveform. If None, all ones is assumed
            phases: phases of each tone in the waveform in [0, 2*pi). If None, random phases are generated using seed
            iq_ratios: ratios for IQ values used to help minimize image tones in band. Allowed values between 0 and 1
                       If None, 50:50 ratio (all ones) is assumed
            phase_offsets: phase offsets in [0, 2*np.pi)
            seed: random seed to seed phase randomization process
            dac_dynamic_range: how much of dac dynamic range to use. Allow values [0.0,1.0]. Default is 1.0 (all)
            optimize_phase: check quantization error and re-generate waveform with new random phases if too large
            compute: compute waveform
        """
        self.freqs = np.asarray(frequencies)
        assert (DAC_FREQ_MIN <= self.freqs).all() and (self.freqs <= DAC_FREQ_MAX).all(), (f"freqs must be in "
                                                                                      f"[{DAC_FREQ_MIN}, "
                                                                                      f"{DAC_FREQ_MAX}]")
        self.n_samples = n_samples
        self.amps = amplitudes if amplitudes is not None else np.ones_like(frequencies)
        self._sample_rate = sample_rate
        self._optimize_phase = optimize_phase

        if phases is None:
            self.phases = np.random.default_rng(seed=seed).uniform(0, 2 * np.pi, size=self.freqs.size)
        else:
            self.phases = np.asarray(phases)
        assert (0 <= self.phases).all and (self.phases < 2*np.pi).all, "phases must be between 0 and 2 pi"
        self.dac_dynamic_range = dac_dynamic_range

        self.iq_ratios = np.asarray(iq_ratios) if iq_ratios is not None else np.ones_like(frequencies)
        self.phase_offsets = np.asarray(phase_offsets) if phase_offsets is not None else np.zeros_like(frequencies)
        assert (0 <= self.phase_offsets).all and (self.phase_offsets < 2*np.pi).all, ("phase offsets must be between 0 "
                                                                                      "and 2 pi")
        self.quant_freqs = quantize_frequencies(self.freqs, rate=sample_rate, n_samples=n_samples)

        self._seed = seed
        self.__values = None  # cache

        self.quant_vals = None
        self.quant_error = None

        if compute:
            self.output_waveform

    def __ne__(self, other):
        if self.quant_vals is not None and other.n_samples is not None:
            return not _same(self.quant_vals, other.quant_vals)

        return not (self.dac_dynamic_range == other.dac_dynamic_range and
                    _same(self.iq_ratios, other.iq_ratios) and
                    self.n_samples == other.n_samples and
                    _same(self.phases, other.phases) and
                    _same(self.quant_freqs, other.quant_freqs) and
                    self._sample_rate == other._sample_rate and
                    _same(self.amps, other.amps) and
                    _same(self.phase_offsets, other.phase_offsets))

    def __repr__(self):
        return f'<{str(self)}>'

    def __str__(self):
        preview_dict = {'freqs': self.freqs, 'amps': self.amps, 'phases': self.phases,
                        'iq_ratios': self.iq_ratios, 'phase_offsets': self.phase_offsets,
                        'quant_error': self.quant_error}
        for key, value in preview_dict.items():
            if value is None or (value.size < 3):
                preview_dict[key] = value
            else:
                preview_dict[key] = value[:3]

        return f'FreqlistWaveform: {preview_dict}'

    @property
    def _values(self) -> nt.NDArray[np.complex128]:
        """
        Return or calculate waveform values
        Returns: Complex values where the real and imag part have been quantized to ints in accordance with specified
                 dac dynamic range

        """
        if self.quant_vals is None:
            self.__values = self._compute_waveform()

            self.quant_vals, self.quant_error = quantize_to_int(self.__values, resolution=DAC_RESOLUTION, signed=True,
                                                                word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                                dyn_range=self.dac_dynamic_range, return_error=True)
            if self._optimize_phase:
                self._optimize_random_phase(max_quant_err=1 * predict_quantization_error(resolution=DAC_RESOLUTION),
                                            max_attempts=3)
        return self.quant_vals

    def _compute_waveform(self, phases: Iterable | None = None) -> nt.NDArray[np.complex64]:
        """
        Compute the raw waveform with no scaling or casting.
        Args:
            phases: new phases to compute waveform with (useful for re-generating random phases)

        Returns: Raw waveform values.

        """
        iq = np.zeros(self.n_samples, dtype=np.complex64)
        # generate each signal
        t = 2 * np.pi * np.arange(iq.size) / self._sample_rate

        phases = self.phases if phases is None else phases

        if self.phase_offsets.any() or (self.iq_ratios != 1).any() or self.phases.ndim > 1:
            logging.getLogger(__name__).debug(
                f'Computing net waveform with {self.freqs.size} tones in a loop to apply IQ ratios and phase offsets.\n'
                f'For 2048 tones this takes about 7 min.')
            for i in range(self.freqs.size):
                exp = self.amps[i] * np.exp(1j * (t * self.quant_freqs[i] + phases[i]))
                scaled = np.sqrt(2) / np.sqrt(1 + self.iq_ratios[i] ** 2)
                c1 = self.iq_ratios[i] * scaled * np.exp(1j * np.deg2rad(self.phase_offsets)[i])
                iq.real += c1.real * exp.real + c1.imag * exp.imag
                iq.imag += scaled * exp.imag

        else:
            logging.getLogger(__name__).debug(
                f'Computing net waveform with {self.freqs.size} tones using IFFT.')
            possible_tones = np.linspace(-DAC_SAMPLE_RATE/2, (DAC_SAMPLE_RATE/2)-DAC_FREQ_RES, DAC_LUT_SIZE)
            tone_idxs = np.concatenate([np.where(possible_tones == freq) for freq in self.quant_freqs]).flatten()
            fft = np.zeros(2**19, dtype=np.complex64)
            for tone_number, tone_idx in enumerate(tone_idxs):
                fft[tone_idx] = self.amps[tone_number]*np.exp(1j*self.phases[tone_number])
            iq = np.fft.ifft(np.fft.fftshift(fft))

        return iq

    def _optimize_random_phase(self,
                               max_quant_err: float | int = 1 * predict_quantization_error(resolution=DAC_RESOLUTION),
                               max_attempts: int = 3) -> None:
        """
        Regenerate random phases, waveform values, and quantized values if quantization error is too high.
        Args:
            max_quant_err: maximum absolute allowable quantization error defined as abs(expected value - achieved value)
            max_attempts: maximum numer of times to regenerate random phases, waveform values, and quantized values

        Returns: None

        Waveform random phases, waveform values, quantized values, and quantization error are only updated if a
        solution is found.
        """

        if max_quant_err is None:
            max_quant_err = 3 * predict_quantization_error(resolution=DAC_RESOLUTION)

        if self.quant_error < max_quant_err:  # already optimal
            return

        quant_error = self.quant_error
        cnt = 0
        while quant_error > max_quant_err:
            logging.getLogger(__name__).warning(
                f"Quantization error {quant_error} exceeded max quantization error {max_quant_err}. The freq comb's "
                f"relative phases may have added up sub-optimally. Calculating with new random phases")
            self._seed += 1
            self.phases = np.random.default_rng(seed=self._seed).uniform(0., 2. * np.pi, len(self.freqs))
            values = self._compute_waveform(phases=self.phases)
            quant_vals, quant_error = quantize_to_int(values, resolution=DAC_RESOLUTION, signed=True,
                                                      word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                                      return_error=True)
            cnt += 1
            if cnt > max_attempts:
                raise Exception("Process reach maximum attempts: Could not find solution below max quantization error.")

        self.__values = values
        self.quant_vals, self.quant_error = quant_vals, quant_error


def WaveformFactory(n_uniform_tones=None, output_waveform=None, frequencies=None,
                    n_samples=DAC_LUT_SIZE, sample_rate=DAC_SAMPLE_RATE, amplitudes=None, phases=None,
                    iq_ratios=None, phase_offsets=None, seed=2, dac_dynamic_range=1.0, compute=False):
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
                            seed=seed, dac_dynamic_range=dac_dynamic_range, compute=compute)


if __name__ == '__main__':
    """
    Test code for debugging waveform generation.
    """
    from mkidgen3.util import pseudo_random_tones
    from mkidgen3.server.feedline_config import *
    import matplotlib.pyplot as plt

    tone = 400e6
    exclude = np.array([tone - 4e6, tone - 3e6, tone - 2e6, tone - 1e6, tone, tone + 1e6, tone + 2e6, tone + 3e6, tone + 4e6])
    random_tones = pseudo_random_tones(n=2048, buffer=300e3, spread=True, exclude=exclude)
    wvfm_tones = np.append(np.array([tone]), random_tones)
    wvfm_cfg = WaveformConfig(
        waveform=WaveformFactory(frequencies=wvfm_tones, seed=5, dac_dynamic_range=1 / 150, compute=True))
    wvfm_fft = np.abs(np.fft.fftshift(np.fft.fft(wvfm_cfg.waveform.output_waveform)))
    freqs = np.linspace(-2.048e9, 2.048e9 - 7.8125e3, 2 ** 19)
    plt.plot(freqs, wvfm_fft)
    plt.xlim(395e6, 405e6)
    plt.show()
    print('hi')
