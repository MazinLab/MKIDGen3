from mkidgen3.funcs import quantize_frequencies
import numpy as np
import numpy.typing as nt

from mkidgen3.system_parameters import (ADC_DAC_INTERFACE_WORD_LENGTH, DAC_RESOLUTION, DAC_SAMPLE_RATE, SYSTEM_BANDWIDTH,
                                        DAC_FREQ_RES, DAC_FREQ_MAX, DAC_FREQ_MIN)

class SimpleWaveform(Waveform):
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
