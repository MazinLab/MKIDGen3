import numpy as np
from fpbinary import FpBinary
import logging

DAC_MAX_OUTPUT_DBM =  1 # [dBm] see Xilinx DS926
DAC_MAX_INT = 8191 # see Xilinx docs
ADC_DAC_INTERFACE_WORD_LENGTH = 16 # bits see Xilinx docs
DAC_RESOLUTION = 14 # bits
DAC_LUT_SIZE = 2**19 # values
DAC_SAMPLE_RATE = 4.096e9 # GSPS
N_OPFB_CHANNELS = 4096 # Number of DDC channels
BANDWIDTH = 4.096e9 # Hz Full readout bandwidth
OS = 2 # OPFB Overlap factor
IF_ATTN_STEP = 0.25 #dB IF attenuator step size TODO: is this combined??

def db2lin(values, mode='voltage'):
    """ Convert a value or values in dB to linear units.
    inputs:
    - values: float
        values in dB to be converted to linear units
    -mode: str
        'power': returns dB value in same power units as reference power
        'voltage': returns dB value proportional to RMS voltage
    Example: if 1 dBm is input (that is 1 dB referenced to 1 milliWatt), 'power'
    will return 1.26 mW.
    """
    values = np.asarray(values)
    if mode == 'power':
        return 10**(values/10)
    if mode == 'voltage':
        return 10**(values/20)

def find_relative_amplitudes(if_attenuations):
    """
    Computes the relative amplitudes of the MKID frequencies based on the if_attenuation they looked best at.
    This is used as a helper function to generate the readout waveform with appropriately scaled powers.
    inputs:
    - mkid_frequencies: list of floats
        a list of mkid resonant frequencies
    - if_attenutation: list of floats
        a list of total if_attenuations, one corresponding to each provided mkid frequency in dB
        VALUES ARE ASSUMED TO BE POSITIVE!!
    returns:
    - relative_amplitudes: list of amplitudes between 0 and 1, one for each MKID in linear units.
    """
    return db2lin(-if_attenuations)/db2lin(-if_attenuations).max()
def ensure_array_or_scalar(x):
    if x is None:
        return x
    return x if np.isscalar(x) else np.asarray(x)

def quantize_frequencies(freqs, rate=4.096e9, n_samples=DAC_LUT_SIZE):
    """
    Quantizes frequencies to nearest available value give the sample rate and number of samples.
    inputs:
    - freqs: array or list of floats
        frequencies to be quantized
    - rate: float
        samples per second
    - n_samples: int
        number of samplesp
    returns: scalar or array
        quantized frequencies
    """
    freqs = ensure_array_or_scalar(freqs)
    freq_res = rate / n_samples
    return np.round(freqs / freq_res) * freq_res

def predict_quantization_error(resolution=DAC_RESOLUTION, signed=True):
    """
    Predict max quantization error when quantizing to an integer with resolution bits of precision.
    Assumes values to be quantized have been scaled to maximize dynamic range i.e max value is 2**(resolution-signed) - 1
    inputs:
    - resolution: int
    number of integer bits with which to quantize
    - signed: bool
    whether or not the quantized values are signed

    Note: It's assumed the quantization step size is small relative to the variation in the signal being quantized.
    """
    max_val = 2**(resolution-signed)-1
    min_val = -2**(resolution-signed)
    return (max_val-min_val)/2**resolution

def quantize_to_int(x, resolution=DAC_RESOLUTION, signed=True, word_length=ADC_DAC_INTERFACE_WORD_LENGTH, return_error=True):
    """"""
    if np.iscomplex(x).any():
        max_val = max(x.real.max(), x.imag.max())
        y = 2 ** (resolution - signed) * x / max_val  # scale to max allowed int value
        quant_real = np.round(y.real).astype(int)
        quant_imag = np.round(y.imag).astype(int)  # round to int
        quant_real.clip(-2 ** (resolution - signed), 2 ** (resolution - signed) - 1, out=quant_real)
        quant_imag.clip(-2 ** (resolution - signed), 2 ** (resolution - signed) - 1, out=quant_imag)
        error = max((y.real - quant_real).max(), (y.imag - quant_imag).max())
        quant = (quant_real << word_length - resolution) + 1j*(quant_imag << word_length - resolution)

    else:
        max_val = x.max()
        y=2 ** (resolution - signed) * x/max_val # scale to max allowed int value
        quant =  np.round(y).astype(int) # round to int
        quant.clip(-2 ** (resolution - signed), 2 ** (resolution - signed)-1, out=quant)
        error = (y-quant).max()
        quant<<=word_length - resolution
    if return_error:
        return quant, error
    else:
        return quant
def complex_scale(z, max_val):
    """
    Returns complex array rescaled so the maximum real or imaginary value is max_val

    inputs:
    - z: list or array of complex number
        input complex array or list
    - max_val: float
        new maximum real or imaginary value
    """
    ensure_array_or_scalar(z)
    input_max = max(z.real.max(), z.imag.max())
    return max_val*z/input_max

def compute_power_sweep_attenuations(start_attn, stop_attn, step_size=IF_ATTN_STEP):
    """
    inputs:
    - start_attn: float [dB]
        the IF board combined starting attenuation
    - stop_attn: float [dB]
        the IF board combined stop attenuation
    - step_size: float [dB]
        attenuation step size
    """
    return np.arange(start_attn,stop_attn+step_size,step_size)

def compute_lo_steps(center, resolution, bandwidth=OS*BANDWIDTH/N_OPFB_CHANNELS):
    """
    inputs:
    - center: float
        center frequency in Hz of the sweep bandwidth
    - resolution: float
        frequency resolution in Hz for the LO sweep
    - bandwidth: float
        bandwidth in Hz for the LO to sweep through
    """
    n_steps = np.round(bandwidth / resolution).astype('int')
    return np.linspace(-bandwidth/2, bandwidth/2, n_steps)+center
def power_sweep_freqs(n_opfb_channels=N_OPFB_CHANNELS, bandwidth=BANDWIDTH):
    """
    inputs:
    - N_OPFB_CHANNELS: int
        Number of channels in the OPFB (all of the places an MKID could be)
    - BANDWIDTH: float
        Full channelizer bandwidth (ADC Nyquist bandwidth) in Hz
    Returns a comb with one frequency at each bin center.
    """
    return(np.linspace(0,n_opfb_channels-1, n_opfb_channels)-n_opfb_channels/2)*(bandwidth/n_opfb_channels)


def generate_waveform(frequencies, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, seed=2, return_merged=True):
    """
    Creates floating-point DAC frequency comb by adding many complex frequencies together with specified amplitudes and phases.
    Frequency values are quantized to the nearest achievable value based on the number of samples and sample rate.
    inputs:
    - frequencies: list/array of floats in (freq, freq)
        list of frequencies in the comb
    - amplitudes: list/array of floats in (0,1]
        list of amplitudes in (0,1] one per frequency. If not specified ones is assumed.
    - phases: list/array of floats in [0, 2*np.pi)
        list of phases for each complex signal, one for each frequency. If None, generates random phases using input seed.
    - iq_ratios: list/array of floats in (0,1]
        list of ratios for IQ values used to help minimize image tones in band. If None, 50:50 (all ones) is assumed.
        TODO: what does this actually do and how does it work
    - phase_offsets: list/array of floats in [0, 2*np.pi)
        TODO: what is this for?
    - seed: int
        random seed to seed phase randomization process
    - return_merged: bool
        whether or not to combine the

    OUTPUT: dict
        ['iq'] - complex64 array of IQ(t) values. If unmerged will have an additional axis corresponding to frequency
        ['frequencies'] - quantized frequencies
        ['phases'] - phases for each frequency
    """
    frequencies, amplitudes, phases, iq_ratios, phase_offsets = tuple(map(ensure_array_or_scalar,(frequencies, amplitudes, phases, iq_ratios, phase_offsets)))

    if amplitudes is None:
        amplitudes = np.ones_like(frequencies)
    if phases is None:
        phases = np.random.default_rng(seed=seed).uniform(0., 2.*np.pi, len(frequencies))
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
        exp = amplitudes[i] * np.exp(1j * (t * quantized_freqs[i] + phases[i]))
        scaled = np.sqrt(2) / np.sqrt(1 + iq_ratios[i] ** 2)
        c1 = iq_ratios[i] * scaled * np.exp(1j * phase_offsets_radians[i])
        if return_merged:
            iq.real += c1.real*exp.real + c1.imag*exp.imag
            iq.imag += scaled * exp.imag
        else:
            iq[i].real = c1.real*exp.real + c1.imag*exp.imag
            iq[i].imag = scaled * exp.imag

    return {'iq': iq, 'frequencies': quantized_freqs, 'phases': phases}

def optimize_random_phase(frequencies, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                          phase_offsets=None, seed=2,
                          max_quant_err=2,
                          max_attempts=10, return_quantized=True):
    """
    #TODO: "waveform" with args(frequencies, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, seed=2) should probably be a class and this function should take it's pre-computed
                      waveform and only recalculate it with different random phases if it has to.
    inputs:
    - max_quant_error: float
        maximum allowable quantization error for real or imaginary samples.
        see predict_quantization_error() for how to estimate this value.
    - max_attempts: int
        Max number of times to recompute the waveform and attempt to get a quantization error below the specified max
        before giving up.

    returns: floating point complex waveform with optimized random phases
    TODO: should support returning quantized waveform
    """
    if max_quant_err is None:
        max_quant_err = 3*predict_quantization_error(resolution=DAC_RESOLUTION)

    waveform = generate_waveform(frequencies, n_samples, sample_rate, amplitudes, phases, iq_ratios,
                      phase_offsets, seed, return_merged=True)['iq']

    # Waveform is ±1 in i and q max complex amplitude of 1
    # Normalizing by the individual I and Q components ensures that at least one dac fully utilizes its dynamic range
    # This is acceptable as on input the attenuators on the RF electronics are set such that the measured ADC values
    # correspond to the maximum readable input signal.
    quant, error = quantize_to_int(waveform, resolution=DAC_RESOLUTION, signed=True, word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                            return_error=True)

    cnt=0
    while error>max_quant_err:
        logging.getLogger(__name__).warning("Max quantization error exceeded. The freq comb's relative phases may have added up sub-optimally."
                                    "Calculating with new random phases")
        seed+=1
        waveform = generate_waveform(frequencies, n_samples, sample_rate, amplitudes, phases, iq_ratios,
                      phase_offsets, seed, return_merged=True)['iq']
        quant, error = quantize_to_int(waveform, resolution=DAC_RESOLUTION, signed=True,
                                       word_length=ADC_DAC_INTERFACE_WORD_LENGTH,
                                       return_error=True)
        cnt+=1
        if cnt>max_attempts:
            raise Exception("Process reach maximum attempts: Could not find solution below max quantization error.")
    error / max_quant_err
    if return_quantized:
        return quant
    else:
        return waveform

def daccomb(frequencies, attenuations, phases=None, iq_ratios=None, phase_offsets=None, max_quant_err=.9,
            globalDacAtten=None, lo=None, return_full=True, max_chan=2048, sample_rate=4.096e9, n_iq_bits=32,
            n_samples=2**19):
    """
    Creates floating-point DAC frequency comb by adding many complex frequencies together with specified amplitudes and phases.

    The attenuations holds the absolute attenuation for each resonator signal coming out of the DAC.
    Zero attenuation means that the tone amplitude is set to the full dynamic range of the DAC and the
    DAC attenuator(s) are set to 0. Thus, all values in attenuations must be larger than globalDacAtten.
    If you decrease the globalDacAtten, the amplitude in the DAC LUT decreases so that the total
    attenuation of the signal is the same.

    Note: Duplicate frequencies in freqList are ignored when making the comb along with their corresponding attenuation, phase, etc...

    INPUTS:
        frequencies - list of frequencies in the comb.
        attenuations - list of absolute attenuation values (dB) for each frequency.
        phases - list of phases for each complex signal. If None, generates random phases.
        iq_ratios - If None, 50:50 is assumed.
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

    max_quant_err=max(max_quant_err, .01)

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
    expectedmax_sig = scipy.special.erfinv((iq.size + max_quant_err - 1) / iq.size) * np.sqrt(2)
    if max_quant_err < 1 and sig_i > 0 and sig_q > 0:
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

def est_loop_centers(iq):
    """
    Finds the (I,Q) centers of the loops via percentile math
    iq - np.complex array[n_loops, n_samples]
    returns centers[iq.shape[0]]

    see mkidgen2.roach2controls.fitLoopCenters for history
    """
    ictr = (np.percentile(iq.real, 95, axis=1) + np.percentile(iq.real, 5, axis=1)) / 2
    qctr = (np.percentile(iq.imag, 95, axis=1) + np.percentile(iq.imag, 5, axis=1)) / 2

    return ictr + qctr * 1j
