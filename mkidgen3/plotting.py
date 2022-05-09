import matplotlib.pyplot as plt
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


def adc_test_plot(adc_data, timerange, fft_range, fft_zoom,  db=True, fs=4.096e9, figsize=(13, 8), **mosaic_kw):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.8])
    adcax = fig.add_subplot(spec[0, :])
    fftax = fig.add_subplot(spec[1, 0])
    fftzoomax = fig.add_subplot(spec[1, 1])

    adc_timeseries(adc_data, timerange, fs=fs, ax=adcax)

    fft_start, fft_stop = fft_range
    fft_sl = slice(*fft_range)
    fft_freqs = np.linspace(-fs/2, fs/2, fft_stop - fft_start)
    y_fft = np.abs(np.fft.fftshift(np.fft.fft(adc_data[fft_sl])))


    if db:
        y_fft = 20*np.log10(y_fft)

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
    tvec = np.linspace(0, n/fs, n)*1e9  # time vector [nano seconds]  TODO only generate used samples
    sl = slice(*timerange)  # plt slice
    plt.plot(tvec[sl], data.real[sl])
    plt.plot(tvec[sl], data.real[sl], "o")
    plt.grid(True)
    plt.xlabel("time (ns)", position=(0.5, 1))
    plt.ylabel("signal (V)", position=(0, 0.5))

    plt.gca().set_xlim(None if timerange[0] is None else tvec[timerange[0]],
                       None if timerange[1] is None else tvec[timerange[1]])
    plt.title('Time Series')


def plot_fft(f, y, db=True, xlim=(-2.048e9, 2.048e9), ylim=None, ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.plot(f, y)
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
