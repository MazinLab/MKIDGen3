from mkidgen3.server.waveform import WaveformFactory, FreqlistWaveform
import numpy as np
from mkidgen3.system_parameters import DAC_SAMPLE_RATE, DAC_FREQ_RES, DAC_LUT_SIZE
import scipy.signal as sig
import pytest

test_wvfms = [WaveformFactory(frequencies=[200e6, 400e6 + 7812.5], amplitudes=[2, 3], compute=True),
              WaveformFactory(n_uniform_tones=2048, compute=True),
              WaveformFactory(n_uniform_tones=2048, dac_dynamic_range=0.8, compute=True)]


@pytest.mark.parametrize("wvfm", test_wvfms)
def test_wvfm_tones(wvfm: FreqlistWaveform) -> None:
    """
    Inspect generated waveform via FFT and ensure the tones are at the right frequencies.
    """
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(wvfm.quant_vals)))
    tones, _ = sig.find_peaks(fft_abs, height=np.mean(fft_abs))
    possible_tones = np.linspace(-DAC_SAMPLE_RATE / 2, (DAC_SAMPLE_RATE / 2) - DAC_FREQ_RES, DAC_LUT_SIZE)

    assert (possible_tones[tones] == wvfm.quant_freqs).all, "Tones are not in the correct place"


@pytest.mark.parametrize("wvfm", test_wvfms)
def test_wvfm_phases(wvfm: FreqlistWaveform) -> None:
    """
    Inspect generated waveform via FFT and ensure the phases are correct.
    """
    fft = np.fft.fftshift(np.fft.fft(wvfm.quant_vals))
    tones, _ = sig.find_peaks(np.abs(fft), height=np.mean(np.abs(fft)))
    phases = np.angle(fft[tones])
    assert (np.isclose(wvfm.phases, phases)).all, "Phases are wrong"
