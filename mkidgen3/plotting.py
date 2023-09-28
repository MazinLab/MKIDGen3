import matplotlib.pyplot as plt
from mkidgen3.opfb import opfb_bin_spectrum, opfb_bin_frequencies
import numpy as np


# NB requires a newer version than we have by default in pynq2.7 venv
# def adc_test_plot(adc_data, timerange, fft_range, fft_zoom,  fs=4.096e9, figsize=(13, 10), **mosaic_kw):
#     gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1.4])
#     fig, axd = plt.subplot_mosaic([['adc', 'adc'],
#                                    ['fft', 'fftzoom']],
#                                   gridspec_kw=gs_kw, figsize=figsize,
#                                   constrained_layout=True, **mosaic_kw)
#
#
#     adc_timeseries(adc_data, timerange, fs=fs, ax=axd['adc'])
#
#     fft_start, fft_stop = fft_range
#     fft_sl = slice(*fft_range)
#     fft_freqs = np.linspace(-fs/2, fs/2, fft_stop - fft_start)
#     y_fft = np.abs(np.fft.fftshift(np.fft.fft(adc_data[fft_sl])))
#
#     plot_fft(fft_freqs[::2], y_fft[::2] - max(y_fft), ax=axd['fft'])
#     plot_fft(fft_freqs[::2], y_fft[::2] - max(y_fft), ax=axd['fftzoom'], xlim=fft_zoom)
#
#     fig.suptitle('ADC Data')


def adc_test_plot(adc_data, timerange, fft_range, fft_zoom, db=True, fs=4.096e9, figsize=(16, 8), **mosaic_kw):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.8])
    adcax = fig.add_subplot(spec[0, :])
    fftax = fig.add_subplot(spec[1, 0])
    fftzoomax = fig.add_subplot(spec[1, 1])

    adc_timeseries(adc_data, timerange, fs=fs, ax=adcax)

    fft_start, fft_stop = fft_range
    fft_sl = slice(*fft_range)
    fft_freqs = np.linspace(-fs / 2, fs / 2, fft_stop - fft_start)
    y_fft = np.abs(np.fft.fftshift(np.fft.fft(adc_data[fft_sl])))

    if db:
        y_fft = 20 * np.log10(y_fft)

    plot_fft(fft_freqs[::2], y_fft[::2] - max(y_fft), ax=fftax)
    plot_fft(fft_freqs[::2], y_fft[::2] - max(y_fft), ax=fftzoomax, xlim=fft_zoom)

    fig.suptitle('ADC Data')


def adc_timeseries(data, timerange=(None, None), fs=4.096e9, ax=None, **kwargs):
    """
    data: np array of captured adc data
    start: first sample # in plot
    stop: last sample # in plot
    fs = ADC sample rate [Hz]

    Returns:
        Plot of ADC timeseries.
    """

    if ax is None:
        plt.figure(figsize=(10, 5))
    else:
        plt.sca(ax)

    n = data.shape[0]  # total samples
    tvec = np.linspace(0, n / fs, n) * 1e9  # time vector [nano seconds]  TODO only generate used samples
    sl = slice(*timerange)  # plt slice
    plt.plot(tvec[sl], data.real[sl], color='#34D576', linewidth=6)
    plt.plot(tvec[sl], data.real[sl], "o", color='#346B76', linewidth=8)
    plt.grid(True)
    plt.xlabel("time (ns)", position=(0.5, 1))
    plt.ylabel("signal (V)", position=(0, 0.5))

    plt.gca().set_xlim(None if timerange[0] is None else tvec[timerange[0]],
                       None if timerange[1] is None else tvec[timerange[1]])
    plt.title('Time Series')


def plot_fft(f, y, db=True, xlim=(-2.048e9, 2.048e9), ylim=None, ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.plot(f, y, color='#346B76', linewidth=3)
    plt.grid(True)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Frequency (Hz)", position=(0.5, 0.5))
    plt.ylabel("power (linear)", position=(1, 0.5))
    if db:
        plt.ylabel("power (dB)", position=(1, 0.5))
    plt.title('FFT')


def plot_adc_fft(data, fs=4.096e9, db=True, fft_points=2 ** 14, xlim=None, ylim=None, ax=None):
    fft_freqs = np.linspace(-2.048e9, 2.048e9, fft_points)
    fft_data = np.abs(np.fft.fftshift(np.fft.fft(data[:fft_points])))
    if db:
        fft_data = 20 * np.log10(fft_data)
    fft_data = fft_data - max(fft_data)
    if ax is not None:
        plt.sca(ax)
    plt.plot(fft_freqs, fft_data, color='#346B76', linewidth=3)
    plt.grid(True)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Frequency (Hz)", position=(0.5, 0.5))
    plt.ylabel("power (linear)", position=(1, 0.5))
    if db:
        plt.ylabel("power (dB)", position=(1, 0.5))
    plt.title('Spectrum')


def plot_opfb_bins(data, bins, fine_fft_shift=True, fft_shift=True, left_snip=0, ol=True):
    """
    Inputs:
    - data: Raw data out of OPFB. Should be in the form N x 4096 where N is the number of samples from a single bin.
    - bins: list of OPFB bins 0 to 4095 (note bin 0 is far left in the +/- 2 GHz spectrum).
    - fine_fft_shift: boolean. To apply an fft shift to the fine fft spectrum of each bin or not. (You should.)
    - fft_shift: boolean. To apply an fft shift to the entire OPFB output spectrum or not. (You should.)
    - ol: To plot the bins as overlapping or discard the overlap region."""

    if fft_shift:
        data = np.fft.fftshift(data, axes=1)
    bin_freqs = opfb_bin_frequencies(bins, data.shape[0])
    spectra = opfb_bin_spectrum(data, bins)

    plt.figure(figsize=(16, 6))
    sl = slice(data.shape[0] // 3, -data.shape[0] // 3) if not ol else slice(0, -1)
    plt.plot(bin_freqs[sl] * 1e-6, spectra[sl])
    plt.xlabel("Frequency (MHz)", position=(0.5, 0.5))
    plt.ylabel("power (dB)", position=(1, 0.5))
    plt.xlim(-2000, 2000)
    return bin_freqs[sl], spectra[sl]


def find_opfb_tones(data):
    """
    Plot the absolute value of the average of the real signal in each OPFB bin. Also report the max peak.
    This quickly gives the location of the peak tone in the OPFB
    Inputs:
    - data: Raw data out of OPFB. Should be in the form N x 4096 where N is the number of samples from a single bin.
    """
    plt.plot(np.abs(data).mean(0), linewidth=3)
    plt.xlabel("OPFB Bin (Raw SSR FFT Output Order)")
    plt.ylabel("|Averge Value of Real Signal|")
    print(f"peak in bin: {np.argmax(abs(data.real.mean(0)))}")


def plot_waveforms(x, sample_rate=2e6, sw_phase=False, cordic=False, ax=None, label=None, xlabel='t (ms)', **pltargs):
    if ax is not None:
        plt.sca(ax)
    t = np.arange(x.size) / sample_rate * 1e3
    if sw_phase:
        plt.plot(t, np.angle(x) / np.pi, **pltargs)
        plt.ylim(-1.1, 1.1)
    if cordic:
        plt.plot(t, x, **pltargs)
    else:
        plt.plot(t, x.real, **pltargs)
        plt.plot(t, x.imag, **pltargs)
    if label:
        plt.ylabel(label)
    if xlabel:
        plt.xlabel(xlabel)


def plot_res_chan(riq, channel, **kwargs):
    reschan_fft = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(riq[:, channel]))))
    reschan_norm = reschan_fft - max(reschan_fft)
    f_ax = np.linspace(-1.024, 1.024, riq.shape[0])
    plt.plot(f_ax, reschan_norm, **kwargs)
