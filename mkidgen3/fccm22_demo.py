import matplotlib.pyplot as plt
from mkidgen3.dsp import opfb_bin_spectrum, opfb_bin_frequencies
import numpy as np


def fccm_tone_bins_spectra(data, bins=(2347, 2348, 2349, 2350), fft_shift=True, left_snip=1, ol=True):
    if fft_shift:
        data = np.fft.fftshift(data, axes=1)
    bin_freqs = opfb_bin_frequencies(bins, data.shape[0], left_snip=left_snip)
    spectra = opfb_bin_spectrum(data, bins)
    sl = slice(data.shape[0] // 3, -data.shape[0] // 3) if not ol else slice(0, -1)
    return bin_freqs[sl], spectra[sl]


def plot_fccm_demo_bins(opfb_shifted, bins=(2347, 2348, 2349, 2350)):
    a = fccm_tone_bins_spectra(opfb_shifted, bins)
    plt.figure(figsize=(16, 6))
    plt.title('OPFB Data')
    plt.plot(a[0][:, 0] * 1e-6, a[1][:, 0], color=(0.8, 0.2, 0.3, 0.9), linewidth=2)
    plt.plot(a[0][:, 1] * 1e-6, a[1][:, 1], color=(0.1, 0.7, 0.8, 0.8), linewidth=8)
    plt.plot(a[0][:, 2] * 1e-6, a[1][:, 2], color=(0.5, 0.1, 0.5, 0.8), linewidth=2)
    plt.plot(a[0][:, 3] * 1e-6, a[1][:, 3], color=(0.4, 0.8, 0.4, 0.8), linewidth=2)
    plt.xlabel("Frequency (MHz)", position=(0.5, 0.5))
    plt.ylabel("power (dB)", position=(1, 0.5))
    plt.xlim(298, 303)
    plt.grid()
    plt.legend(['Bin 2347', 'Bin 2348', 'Bin 2349', 'Bin 2350'])
