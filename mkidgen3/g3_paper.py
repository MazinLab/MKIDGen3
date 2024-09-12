import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy.optimize as spo
from mkidgen3.server.feedline_config import *
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.opfb import opfb_bin_number


def fit_psd_floor(freqs, spect, cutoff=30e3):
    def flatoverf(freq, a, b):
        return a + b / freq

    fitcutoff = np.argmin(np.abs(freqs - cutoff))  # remove rolloff
    popt, pcov = spo.curve_fit(flatoverf, freqs[1:fitcutoff], spect[1:fitcutoff], sigma=spect[1:fitcutoff])
    return popt[0]


def get_multi_tone_phase_psd(ro_tone, static_tones, dac_dynamic_range, ol, psd_rolloff_cutoff, program_matched_filt=False, plot=True):
    # Run DAC
    if (np.abs(ro_tone - static_tones) < 1.0e6).any():
        ro_tone = static_tones[np.abs(ro_tone - static_tones) < 1.0e6][0]
        wvfm_tones = static_tones
    else:
        wvfm_tones = np.append(ro_tone, static_tones)
    wvfm_cfg = WaveformConfig(waveform=WaveformFactory(frequencies=wvfm_tones, seed=6, dac_dynamic_range=dac_dynamic_range, compute=True))
    ol.dac_table.configure(**wvfm_cfg.settings_dict())

    # Bin2Res + DDC
    bins = np.zeros(2048, dtype=int)
    bins[0] = opfb_bin_number(ro_tone, ssr_raw_order=True)

    chan = ChannelConfig(bins=bins)
    ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())

    ddc_tones = np.zeros(2048)
    ddc_tones[0] = ro_tone

    ddc = DDCConfig(tones=ddc_tones)
    ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())


    if program_matched_filt:
        # Matched Filter
        filtercfg=FilterConfig(coefficients=f'unity{2048}')
        ol.photon_pipe.phasematch.configure(**filtercfg.settings_dict())

    # Capture Phase
    x = ol.capture.capture_phase(2 ** 19, [0, 1], tap_location='filtphase')
    phase = np.array(x)
    x.freebuffer()
    phase0 = np.pi * phase[:, 0] / (2 ** 15 - 1)

    f, psd = welch(phase0, fs=1e6, nperseg=1e6 / 1e3)

    floor = fit_psd_floor(f, psd, psd_rolloff_cutoff)

    if plot:
        fig, ax = plt.subplots()
        ax.semilogx(f, 10 * np.log10(psd))
        ax.axhline(10 * np.log10(floor),  color='r', linestyle='--')
        ax.set_xlabel(f'Frequency [Hz] ({1e3 * 1e-3:g} kHz resolution)')
        ax.set_ylabel('dBc/Hz')
        ax.grid()
        ax.set_title('Power Spectral Density')

    return f, psd, floor, ro_tone


def get_single_tone_phase_psd(tone, dac_dynamic_range, ol, psd_rolloff_cutoff, program_matched_filt=False, plot=True):
    # Run DAC
    wvfm_cfg = WaveformConfig(waveform=WaveformFactory(frequencies=tone, seed=6, dac_dynamic_range=dac_dynamic_range, compute=True))
    ol.dac_table.configure(**wvfm_cfg.settings_dict())

    # Bin2Res + DDC
    chan = wvfm_cfg.default_channel_config
    ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
    ddc = wvfm_cfg.default_ddc_config
    ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

    if program_matched_filt:
        # Matched Filter
        filtercfg=FilterConfig(coefficients=f'unity{2048}')
        ol.photon_pipe.phasematch.configure(**filtercfg.settings_dict())

    # Capture Phase
    x = ol.capture.capture_phase(2 ** 19, [0, 1], tap_location='filtphase')
    phase = np.array(x)
    x.freebuffer()
    phase0 = np.pi * phase[:, 0] / (2 ** 15 - 1)

    f, psd = welch(phase0, fs=1e6, nperseg=1e6 / 1e3)

    floor = fit_psd_floor(f, psd, psd_rolloff_cutoff)

    if plot:
        fig, ax = plt.subplots()
        ax.semilogx(f, 10 * np.log10(psd))
        ax.axhline(10 * np.log10(floor),  color='r', linestyle='--')
        ax.set_xlabel(f'Frequency [Hz] ({1e3 * 1e-3:g} kHz resolution)')
        ax.set_ylabel('dBc/Hz')
        ax.grid()
        ax.set_title('Power Spectral Density')

    return f, psd, floor


def calculate_psd(data, fs=1e6, fres=1e3, fit_level=True, plot=True):
    plt.figure()
    f, psd = welch(data, fs=fs, nperseg=fs/fres)

    if fit_level:
        pass

    if plot:
        fig, ax = plt.subplots()
        ax.semilogx(f, 10 * np.log10(psd))
        ax.set_xlabel(f'Frequency [Hz] ({fres * 1e-3:g} kHz resolution)')
        ax.set_ylabel('dBc/Hz')
        ax.grid()
        ax.set_title('Power Spectral Density')

    return f, psd


