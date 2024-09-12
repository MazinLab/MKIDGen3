import numpy as np
import scipy.special
from logging import getLogger
import logging
from mkidgen3.gen2 import SweepFile, parse_lo
from mkidgen3.opfb import quantize_frequencies

DAC_REPLAY_SAMPLES=262144


def generate_dac_comb(frequencies, n_samples, sample_rate, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, return_merged=True):
    """
    Generate a list of complex signals with amplitudes and phases specified and frequencies quantized

    All specified inputs must be of the same shape

    INPUTS:
        frequencies - resonator frequencies
        n_samples - Number of time samples
        sample_rate - Used to quantize the frequencies
        amplitudes - Tone amplitudes (0 -1). If None, use 1.
        phases - Phases to use. If None, use random phase
        return_merged - set to false to return timeseries for each frequency independently. Uses frequencies.size times
        more memory

    OUTPUTS (in dict):
        iq - complex64 array of IQ(t) values. If unmerged will have an additional axis corresponding to frequency
        frequencies - quantized frequencies
        phases - phases for each frequency
    """
    if amplitudes is None:
        amplitudes = np.ones_like(frequencies)
    if phases is None:
        phases = np.random.uniform(0., 2. * np.pi, len(frequencies))
    if iq_ratios is None:
        iq_ratios = np.ones_like(frequencies)
    if phase_offsets is None:
        phase_offsets = np.zeros_like(frequencies)

    # Quantize the frequencies to their closest digital value
    quantized_freqs = quantize_frequencies(frequencies, rate=sample_rate, n_samples=n_samples)
    phase_offsets_radians = np.deg2rad(phase_offsets)

    if return_merged:
        iq = np.zeros(n_samples, dtype=np.complex64)
    else:
        iq = np.zeros((frequencies.size, n_samples), dtype=np.complex64)

    # generate each signal
    t = 2 * np.pi * np.arange(n_samples)/sample_rate
    for i in range(frequencies.size):
        phi = t * quantized_freqs[i]
        exp = amplitudes[i] * np.exp(1j * (phi + phases[i]))
        iScale = np.sqrt(2) * iq_ratios[i] / np.sqrt(1. + iq_ratios[i] ** 2)
        qScale = np.sqrt(2) / np.sqrt(1 + iq_ratios[i] ** 2)
        if return_merged:
            iq.real += iScale * (np.cos(phase_offsets_radians[i]) * np.real(exp) +
                                 np.sin(phase_offsets_radians[i]) * np.imag(exp))
            iq.imag += qScale * np.imag(exp)
        else:
            iq[i].real = iScale * (np.cos(phase_offsets_radians[i]) * np.real(exp) +
                                   np.sin(phase_offsets_radians[i]) * np.imag(exp))
            iq[i].imag = qScale * np.imag(exp)

    return {'iq': iq, 'frequencies': quantized_freqs, 'phases': phases}


def daccomb(frequencies, attenuations, phases=None, iq_ratios=None, phase_offsets=None, spike_percentile_limit=.9,
             globalDacAtten=None, lo=None, return_full=True, max_chan=2048, sample_rate=4.096e9, n_iq_bits=32,
             n_samples=2**19):
    """
    Creates DAC frequency comb by adding many complex frequencies together with specified amplitudes and phases.

    The attenuations holds the absolute attenuation for each resonator signal coming out of the DAC.
    Zero attenuation means that the tone amplitude is set to the full dynamic range of the DAC and the
    DAC attenuator(s) are set to 0. Thus, all values in attenuations must be larger than globalDacAtten.
    If you decrease the globalDacAtten, the amplitude in the DAC LUT decreases so that the total
    attenuation of the signal is the same.

    Note: The freqList need not be unique. If there are repeated values in the freqList then
    they are completely ignored when making the comb along with their corresponding attenuation, phase, etc...

    INPUTS:
        frequencies - list of all resonator frequencies.
        attenuations - list of absolute attenuation values (dB) for each resonator.
        phases - list of phases for each complex signal. If None, generates random phases.
        iq_ratios -
        phase_offsets -
        spike_percentile_limit - loop generateTones() function with random phases to avoid spikes greater than the
        specified percentile in the output comb. Set to >=1 to disable.

    OUTPUTS:
        dictionary with keywords
        I - I(t) values for frequency comb [signed 32-bit integers]
        Q - Q(t)
        quantizedFreqList - list of frequencies after digitial quantiziation
        dacAtten - The global dac hardware attenuation in dB that should be set

    """

    spike_percentile_limit=max(spike_percentile_limit, .01)

    if len(frequencies) != len(attenuations):
        raise ValueError("Need exactly one attenuation value for each resonant frequency!")

    if phases is not None and len(frequencies) != len(phases):
        raise ValueError("Need exactly one phase value for each resonant frequency!")

    if iq_ratios is not None and len(frequencies) != len(iq_ratios):
        raise ValueError("Need exactly one iqRatio value for each resonant frequency!")

    if phase_offsets is not None and len(frequencies) != len(phase_offsets):
        raise ValueError("Need exactly one iqPhaseOffs value for each resonant frequency!")

    if len(frequencies) > max_chan:
        getLogger(__name__).warning(f"Clipping the last {frequencies.size-max_chan}. MAX_CHAN={max_chan}.")
        frequencies = frequencies[:max_chan]
        attenuations = attenuations[:max_chan]
        if phase_offsets is not None:
            phase_offsets=phase_offsets[:max_chan]
        if iq_ratios is not None:
            iq_ratios=iq_ratios[:max_chan]
        if phases is not None:
            phases=phases[:max_chan]

    getLogger(__name__).debug('Generating DAC comb...')

    autoDacAtten = globalDacAtten is None
    if autoDacAtten:
        globalDacAtten = np.amin(attenuations)

    # Calculate relative amplitudes for DAC LUT
    nBitsPerSampleComponent = n_iq_bits / 2
    maxAmp = int(np.round(2 ** (nBitsPerSampleComponent - 1) - 1))  # 1 bit for sign
    amplitudes = maxAmp * 10 ** (-(attenuations - globalDacAtten) / 20)

    # Calculate resonator frequencies for DAC
    LOFreq = parse_lo(lo, frequencies=frequencies, sample_rate=sample_rate)

    dacFreqList = frequencies - LOFreq
    dacFreqList[dacFreqList < 0] += sample_rate  # For +/- freq

    # Make sure dac tones are unique
    dacFreqList, args, args_inv = np.unique(dacFreqList, return_index=True, return_inverse=True)

    rstate = np.random.get_state()
    from numpy.random import MT19937, RandomState, SeedSequence
    np.random.set_state(RandomState(MT19937(SeedSequence(123456789))).get_state())

    # Generate and add up individual tone time series.
    toneDict = generate_dac_comb(dacFreqList, n_samples, sample_rate, return_merged=True,
                                 amplitudes=amplitudes[args], phases=None if phases is None else phases[args],
                                 iq_ratios=None if iq_ratios is None else iq_ratios[args],
                                 phase_offsets=None if phase_offsets is None else phase_offsets[args])

    # This part takes the longest

    iq = toneDict['iq']

    # check that we are utilizing the dynamic range of the DAC correctly
    sig_i = iq.real.std()
    sig_q = iq.imag.std()

    # 10% of the time there should be a point this many sigmas higher than average
    expectedmax_sig = scipy.special.erfinv((iq.size + spike_percentile_limit - 1)/ iq.size) * np.sqrt(2)
    if spike_percentile_limit < 1 and sig_i > 0 and sig_q > 0:
        while max(np.abs(iq.real).max() / sig_i, np.abs(iq.imag).max() / sig_q) >= expectedmax_sig:
            getLogger(__name__).warning("The freq comb's relative phases may have added up sub-optimally. "
                                        "Calculating with new random phases")
            toneDict = generate_dac_comb(dacFreqList, n_samples, sample_rate, amplitudes=amplitudes[args], phases=None,
                                         iq_ratios=None if iq_ratios is None else iq_ratios[args],
                                         phase_offsets=None if phase_offsets is None else phase_offsets[args],
                                         return_merged=True)
            iq = toneDict['iq']

    np.random.set_state(rstate)

    dacQuantizedFreqList = (toneDict['frequencies'])[args_inv]
    dacPhaseList = (toneDict['phases'])[args_inv]

    if autoDacAtten:
        highestVal = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
        dBexcess = 20 * np.log10(highestVal / maxAmp)
        dBexcess = np.ceil(4 * dBexcess) / 4  # rounded up to nearest 1/4 dB
        globalDacAtten -= dBexcess
        # reduce to fit into DAC dynamic range and quantize to integer

        if globalDacAtten > 31.75 * 2:
            dB_reduce = globalDacAtten - 31.75 * 2
            getLogger(__name__).warning(f"Unable to fully utilize DAC dynamic range by {dB_reduce} dB")
            globalDacAtten -= dB_reduce
            dBexcess += dB_reduce
        elif np.max((np.abs(iq.real).max(), np.abs(iq.imag).max())) > maxAmp:
            dBexcess += 0.25  # Since there's some rounding there's a small chance we need to decrease by another atten step

        iq /= 10 ** (dBexcess / 20)

    np.round(iq, out=iq)

    highestVal = max(np.abs(iq.real).max(), np.abs(iq.imag).max())

    msg = ('\tGlobal DAC atten: {} dB'.format(globalDacAtten) +
           '\tUsing {} percent of DAC dynamic range\n'.format(highestVal / maxAmp * 100) +
           '\thighest: {} out of {}\n'.format(highestVal, maxAmp) +
           '\tsigma_I: {}  sigma_Q:{}\n'.format(np.std(iq.real), np.std(iq.imag)) +
           '\tLargest val_I: {} sigma. '.format(np.abs(iq.real).max() / np.std(iq.real)) +
           'val_Q: {} sigma.\n'.format(np.abs(iq.imag).max() / np.std(iq.imag)) +
           '\tExpected val: {} sigmas\n'.format(expectedmax_sig))
    getLogger(__name__).debug(msg)

    if globalDacAtten < 0:
        raise ValueError("Desired resonator powers are unacheivable. "
                         f"Increase resonator attens by {-1 * globalDacAtten} dB")

    if return_full:
        return {'frequencies': dacQuantizedFreqList, 'attenuation': globalDacAtten,
                'comb': iq, 'phases': dacPhaseList}
    else:
        return iq


def meccomb(mec_freqfile, lo):
    freqfile = SweepFile(mec_freqfile)
    combdata = daccomb(frequencies=freqfile.freq, attenuations=freqfile.atten, phases=freqfile.phases,
                       iq_ratios=freqfile.iqRatios, globalDacAtten=None, lo=lo)
    return combdata
