import numpy as np
import scipy.special
from logging import getLogger
import logging
from .gen2 import SweepFile, parse_lo

nDacSamplesPerCycle = 8
nLutRowsToUse = 2 ** 15
dacSampleRate = 2.048e9
nBitsPerSamplePair = 32
nChannels = 1024


def generateTones(frequencies, nSamples, sampleRate, amplitudes=None, phases=None, iq_ratios=None,
                  phase_offsets=None, return_merged=True):
    """
    Generate a list of complex signals with amplitudes and phases specified and frequencies quantized

    All specified inputs must be of the same shape

    INPUTS:
        freqList - list of resonator frequencies
        nSamples - Number of time samples
        sampleRate - Used to quantize the frequencies
        amplitudeList - list of amplitudes. If None, use 1.
        phaseList - list of phases. If None, use random phase
        return_merged - if set to fault use frequencies.size times more memory and return an unmerged frequency comb

    OUTPUTS:
        dictionary with keywords
        I - each element is a list of I(t) values for specific freq if not return_merged else the summed I(t)
        Q - Same as I but for Q(t)
        frequencies - list of frequencies after digital quantization
        phases - list of phases for each frequency
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
    freq_res = sampleRate / nSamples
    quantized_freqs = np.round(frequencies / freq_res) * freq_res
    phase_offsets_radians = np.deg2rad(phase_offsets)

    if return_merged:
        ivals = np.zeros(nSamples)
        qvals = np.zeros(nSamples)
    else:
        ivals = np.zeros((frequencies.size, nSamples))
        qvals = np.zeros((frequencies.size, nSamples))

    # generate each signal
    t = 2 * np.pi * np.arange(nSamples)/sampleRate
    for i in range(frequencies.size):
        phi = t * quantized_freqs[i]
        exp = amplitudes[i] * np.exp(1j * (phi + phases[i]))
        iScale = np.sqrt(2) * iq_ratios[i] / np.sqrt(1. + iq_ratios[i] ** 2)
        qScale = np.sqrt(2) / np.sqrt(1 + iq_ratios[i] ** 2)
        if return_merged:
            ivals += iScale * (np.cos(phase_offsets_radians[i]) * np.real(exp) +
                               np.sin(phase_offsets_radians[i]) * np.imag(exp))
            ivals += qScale * np.imag(exp)
        else:
            ivals[i] = iScale * (np.cos(phase_offsets_radians[i]) * np.real(exp) +
                                 np.sin(phase_offsets_radians[i]) * np.imag(exp))
            ivals[i] = qScale * np.imag(exp)

    return {'I': ivals, 'Q': qvals, 'frequencies': quantized_freqs, 'phases': phases}


def generate(frequencies, attenuations, phases=None, iq_ratios=None, phase_offsets=None, spike_percentile_limit=.9,
             globalDacAtten=None, lo=None, return_full=True, MAX_CHAN=2048):
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

    if frequencies > MAX_CHAN:
        getLogger(__name__).warning(f"Clipping the last {frequencies.size-MAX_CHAN}. MAX_CHAN={MAX_CHAN}.")
        frequencies = frequencies[:MAX_CHAN]
        attenuations = attenuations[:MAX_CHAN]
        if phase_offsets is not None:
            phase_offsets=phase_offsets[:MAX_CHAN]
        if iq_ratios is not None:
            iq_ratios=iq_ratios[:MAX_CHAN]
        if phases is not None:
            phases=phases[:MAX_CHAN]

    getLogger(__name__).debug('Generating DAC comb...')

    autoDacAtten = globalDacAtten is None
    if autoDacAtten:
        globalDacAtten = np.amin(attenuations)

    # Calculate relative amplitudes for DAC LUT
    nBitsPerSampleComponent = nBitsPerSamplePair / 2
    maxAmp = int(np.round(2 ** (nBitsPerSampleComponent - 1) - 1))  # 1 bit for sign
    amplitudes = maxAmp * 10 ** (-(attenuations - globalDacAtten) / 20)

    # Calculate nSamples and sampleRate
    nSamples = nDacSamplesPerCycle * nLutRowsToUse
    sampleRate = dacSampleRate

    # Calculate resonator frequencies for DAC
    LOFreq = parse_lo(lo, freqList=frequencies)

    dacFreqList = frequencies - LOFreq
    dacFreqList[dacFreqList < 0.] += dacSampleRate  # For +/- freq

    # Make sure dac tones are unique
    dacFreqList, args, args_inv = np.unique(dacFreqList, return_index=True, return_inverse=True)

    rstate = np.random.get_state()
    np.random.seed(0)

    # Generate and add up individual tone time series.
    toneDict = generateTones(frequencies= dacFreqList, nSamples=nSamples, sampleRate=sampleRate,
                             amplitudes=amplitudes[args],
                             phases=None if phases is None else phases[args],
                             iq_ratios=None if iq_ratios is None else iq_ratios[args],
                             phase_offsets=None if phase_offsets is None else phase_offsets[args],
                             return_merged=True)

    # This part takes the longest
    iValues = toneDict['I']
    qValues = toneDict['Q']

    # check that we are utilizing the dynamic range of the DAC correctly
    sig_i = iValues.std()
    sig_q = qValues.std()

    # 10% of the time there should be a point this many sigmas higher than average
    expectedHighestVal_sig = scipy.special.erfinv((len(iValues) + spike_percentile_limit -1)/ len(iValues)) * np.sqrt(2)
    if spike_percentile_limit<1 and sig_i > 0 and sig_q > 0:
        while max(np.abs(iValues).max() / sig_i, np.abs(qValues).max() / sig_q) >= expectedHighestVal_sig:
            getLogger(__name__).warning("The freq comb's relative phases may have added up sub-optimally. "
                                        "Calculating with new random phases")
            toneDict = generateTones(frequencies=dacFreqList, nSamples=nSamples, sampleRate=sampleRate,
                                     amplitudes=amplitudes[args], phases=None,
                                     iq_ratios=None if iq_ratios is None else iq_ratios[args],
                                     phase_offsets=None if phase_offsets is None else phase_offsets[args],
                                     return_merged=True)
            iValues = toneDict['I']
            qValues = toneDict['Q']

    np.random.set_state(rstate)

    dacQuantizedFreqList = (toneDict['frequencies'])[args_inv]
    dacPhaseList = (toneDict['phases'])[args_inv]

    if autoDacAtten:
        highestVal = max(np.abs(iValues).max(), np.abs(qValues).max())
        dBexcess = 20 * np.log10(highestVal / maxAmp)
        dBexcess = np.ceil(4 * dBexcess) / 4  # rounded up to nearest 1/4 dB
        # reduce to fit into DAC dynamic range and quantize to integer
        iValues_new = np.round(iValues / 10 ** (dBexcess / 20)).astype(np.int)
        qValues_new = np.round(qValues / 10 ** (dBexcess / 20)).astype(np.int)
        if np.max((np.abs(iValues).max(), np.abs(qValues).max())) > maxAmp:
            dBexcess += 0.25  # Since there's some rounding there's a small chance we need to decrease by another atten step
            iValues_new = np.round(iValues / 10 ** (dBexcess / 20)).astype(np.int)
            qValues_new = np.round(qValues / 10 ** (dBexcess / 20)).astype(np.int)

        globalDacAtten -= dBexcess
        if globalDacAtten > 31.75 * 2:
            dB_reduce = globalDacAtten - 31.75 * 2
            getLogger(__name__).warning(f"Unable to fully utilize DAC dynamic range by {dB_reduce} dB")
            globalDacAtten -= dB_reduce
            dBexcess += dB_reduce
            iValues_new = np.round(iValues / 10 ** (dBexcess / 20)).astype(np.int)
            qValues_new = np.round(qValues / 10 ** (dBexcess / 20)).astype(np.int)

        iValues = iValues_new
        qValues = qValues_new

    else:
        iValues = np.round(iValues).astype(np.int)
        qValues = np.round(qValues).astype(np.int)

    highestVal = max(np.abs(iValues).max(), np.abs(qValues).max())

    dacFreqComb = iValues + 1j * qValues

    msg = ('\tGlobal DAC atten: {} dB'.format(globalDacAtten) +
           '\tUsing {} percent of DAC dynamic range\n'.format(highestVal / maxAmp * 100) +
           '\thighest: {} out of {}\n'.format(highestVal, maxAmp) +
           '\tsigma_I: {}  sigma_Q:{}\n'.format(np.std(iValues), np.std(qValues)) +
           '\tLargest val_I: {} sigma. '.format(np.abs(iValues).max() / np.std(iValues)) +
           'val_Q: {} sigma.\n'.format(np.abs(qValues).max() / np.std(qValues)) +
           '\tExpected val: {} sigmas\n'.format(expectedHighestVal_sig))
    getLogger(__name__).debug(msg)

    if globalDacAtten < 0:
        raise ValueError("Desired resonator powers are unacheivable. "
                         f"Increase resonator attens by {-1 * globalDacAtten} dB")

    if return_full:
        return {'i': iValues, 'q': qValues, 'frequencies': dacQuantizedFreqList, 'attenuation': globalDacAtten,
                 'comb': dacFreqComb, 'phases': dacPhaseList}
    else:
        return dacFreqComb


def generate_from_MEC(mec_freqfile, lo):
    freqfile = SweepFile(mec_freqfile)
    combdata = generate(frequencies=freqfile.freq, attenuations=freqfile.atten, phases=freqfile.phases,
                        iq_ratios=freqfile.iqRatios, globalDacAtten=None, lo=lo)
    return combdata


if __name__ == '__main__':

    mec_freqfile='/Users/one/Desktop/untitled folder/psfreqs_FL8a_clip.txt'
    lo=4428029278.278099

    freqfile = SweepFile(mec_freqfile)
    logging.basicConfig()
    combdata = generate(frequencies=freqfile.freq, attenuations=freqfile.atten, phases=freqfile.phases,
                        iq_ratios=freqfile.iqRatios, globalDacAtten=None, lo=lo)
    np.savez('mec_fl8a_dec19_62dB.npz', combdata['comb'])