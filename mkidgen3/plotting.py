import matplotlib.pyplot as plt
from mkidgen3.opfb import opfb_bin_spectrum, opfb_bin_frequencies
from mkidgen3.util import rx_power
import numpy.typing as nt
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

def plot_single_res_sweep(lo_sweep_freqs, tones, iq_vals, lo_res_freq = None, iq_val_res = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.rcParams.update({'font.size': 12})

    ax1.plot(lo_sweep_freqs + tones[0], 20 * np.log10(np.abs(iq_vals)), linestyle=" ", marker=".", markersize=10)
    if lo_res_freq is not None:
        ax1.plot(lo_res_freq * 1e6 + tones[0], 20 * np.log10(np.abs(iq_val_res)), linestyle=" ", marker=".", markersize=10,
             label='bias point')
    ax1.set_ylabel('|S21| [dB]')
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_title('Transmission')
    ax1.legend(loc='center left')

    ax2.plot(iq_vals.real, iq_vals.imag, 'o')
    if lo_res_freq is not None:
        ax2.plot(iq_val_res.real, iq_val_res.imag, 'o', label='bias point')
    ax2.set_xlabel('Real(S21)')
    ax2.set_aspect('equal')
    ax2.set_ylabel('Imag(S21)')
    ax2.set_title('IQ Loop')
    ax2.legend(loc='center left')
    plt.tight_layout()

def plot_phase(phase: nt.NDArray[np.int16], xlim: tuple) -> None:
    """

    Args:
        phase: phase data

    Returns:

    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    t = 1e3 * np.arange(phase.size) / 2e6
    ax.plot(t, phase)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Phase (radians)')
    ax.set_xlim(xlim);
    ax.set_ylim(-np.pi, np.pi);


def plot_ddc(ddciq: nt.NDArray[np.complex128], phase: nt.NDArray[np.int16], chan_plt: list[int]) -> None:
    """
    Plot post-DDC IQ and phase for given channels
    Args:
        ddciq: complex data from ddciq tap
        phase: phase data
        chan_plt: channels to plot

    Returns:

    """
    fig, axes = plt.subplots(4, len(chan_plt), figsize=(15, 10))
    for j, (i, ax) in enumerate(zip(chan_plt, axes.T)):
        plot_waveforms(ddciq[:100, i], 2e6, mode='iq', ax=ax[0], label='DDC Out' if not j else '')
        plot_waveforms(ddciq[:50, i], 1e6, mode='sw_phase', ax=ax[1], label='Phase/pi (software)' if not j else '')
        plot_waveforms(phase[:50, i] / 2 ** 15, 1e6, mode='phase', ax=ax[2], label='Phase/pi (FPGA)' if not j else '')
        plt.ylim([-1, 1])
        err = np.angle(ddciq[:50, i]) - np.pi * phase[:50, i] / 2 ** 15
        plot_waveforms(err, 1e6, mode='phase', ax=ax[3], label='Phase error (radians)' if not j else '')
    plt.suptitle('Gen3 DSP Pipeline (OPFB Out, DDC, Software phase, FPGA (cordic) phase)');


def plot_comp_sat(
        ol,
        ifboard,
        output_atten=0,
        input_range=(60, 50, 1),
        **kwargs,
):
    input_attens = np.arange(
        input_range[1],
        input_range[0],
        1 if len(input_range) == 2 else input_range[2],
    )[::-1]
    dbms_i, dbms_q = [], []
    adcs_i, adcs_q = [], []
    for input_atten in input_attens:
        ifboard.set_attens(output_attens=output_atten, input_attens=input_atten)
        _, dbm, adc = rx_power(ol.capture.capture_adc(2 ** 19, complex=False))
        dbms_i.append(dbm[0])
        adcs_i.append(adc[0])
        dbms_q.append(dbm[1])
        adcs_q.append(adc[1])
    ifboard.set_attens(output_attens=output_atten, input_attens=input_attens[0])
    dbms_i, dbms_q = np.array(dbms_i), np.array(dbms_q)
    adcs_i, adcs_q = np.array(adcs_i), np.array(adcs_q)

    fig, (ax1, ax3, ax2) = plt.subplots(3, sharex=True, **kwargs)
    ax1.plot(input_attens, dbms_i, label="I")
    ax1.plot(input_attens, dbms_q, label="Q")
    ax1.plot(
        input_attens,
        np.min(dbms_i) - (input_attens - np.max(input_attens)),
        linestyle='--',
        color='gray',
        label="Ideal",
    )
    ax1.set_ylabel("RX Power @ ZU48DR Port (dBm)")

    ax3.plot(input_attens, dbms_i - (np.min(dbms_i) - (input_attens - np.max(input_attens))), label="I")
    ax3.plot(input_attens, dbms_q - (np.min(dbms_i) - (input_attens - np.max(input_attens))), label="Q")
    ax3.plot(
        input_attens,
        np.zeros_like(input_attens),
        linestyle='--',
        color='gray',
        label="Ideal",
    )
    ax3.set_ylabel("Residual (dBm)")

    pg = lambda db: 10 ** (db / 10)

    ax2.plot(input_attens, adcs_q, label='I')
    ax2.plot(input_attens, adcs_i, label='Q')
    ax2.plot(
        input_attens,
        adcs_i[np.argmax(input_attens)] * np.sqrt(pg(np.max(input_attens) - input_attens)),
        label='expected gain')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(1, label="Saturation Point", color="black", zorder=-1, linestyle='-.')
    ax2.set_ylabel("ADC Max Magnitude [0, 1)")
    ax2.set_xlabel("Input Attenuation (dB)")

    ax1.legend()
    ax2.legend()
    ax2.invert_xaxis()
    ax2.set_xlim(np.max(input_attens), np.min(input_attens))
    plt.tight_layout()
    plt.show()

    return dbms_i, dbms_q, adcs_i, adcs_q


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

    plot_fft(fft_freqs[::2]*1e-6, y_fft[::2] - max(y_fft), ax=fftax)
    plot_fft(fft_freqs[::2]*1e-6, y_fft[::2] - max(y_fft), ax=fftzoomax, xlim=fft_zoom)

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
    tvec = np.linspace(0, n / fs, n) * 1e9  # time vector [nano seconds]
    sl = slice(*timerange)  # plt slice
    plt.plot(tvec[sl], data.real[sl], color='#34D576', linewidth=6)
    plt.plot(tvec[sl], data.real[sl], "o", color='#346B76', linewidth=8)
    plt.grid(True)
    plt.xlabel("time (ns)", position=(0.5, 1))
    plt.ylabel("signal (V)", position=(0, 0.5))

    plt.gca().set_xlim(None if timerange[0] is None else tvec[timerange[0]],
                       None if timerange[1] is None else tvec[timerange[1]])
    plt.title('Time Series')


def plot_fft(f, y, db=True, xlim=(-2048, 2048), ylim=None, ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.plot(f, y, color='#346B76', linewidth=3)
    plt.grid(True)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Frequency (MHz)", position=(0.5, 0.5))
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


def plot_waveforms(x, sample_rate=2e6, mode=None, ax=None, label=None, xlabel='t (ms)', **pltargs):
    """
    Plots IQ and Phase data from OPFB and DDC Bins.
    Args:
        x: input data as phase or iq values
        sample_rate: samplpe rate in Hz
        mode:
            - "sw_phase": computes and plots phase from IQ values
            - "phase": plots phase
            - "iq": Plots I and Q values as two overlaid timetraces
        ax: plot ax
        label: plot label
        xlabel: xaxis label
        **pltargs: plotargs

    Returns:

    """
    if ax is not None:
        plt.sca(ax)
    t = np.arange(x.size) / sample_rate * 1e3
    if mode == 'sw_phase':
        plt.plot(t, np.angle(x) / np.pi, **pltargs)
        plt.ylim(-1.1, 1.1)
    elif mode == 'phase':
        plt.plot(t, x, **pltargs)
        plt.ylim(-1.1, 1.1)
    elif mode == 'iq':
        plt.plot(t, x.real, **pltargs)
        plt.plot(t, x.imag, **pltargs)
    else:
        raise NotImplementedError(f'mode {mode} is unkown. Valid modes are: "sw_phase", "phase", or "iq"')
    if label:
        plt.ylabel(label)
    if xlabel:
        plt.xlabel(xlabel)


def plot_res_chan(riq, channel, **kwargs):
    reschan_fft = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(riq[:, channel]))))
    reschan_norm = reschan_fft - max(reschan_fft)
    f_ax = np.linspace(-1.024, 1.024, riq.shape[0])
    plt.plot(f_ax, reschan_norm, **kwargs)
