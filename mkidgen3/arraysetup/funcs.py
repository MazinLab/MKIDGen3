import numpy as np

def est_loop_centers(iq):
    """
    Finds the (I,Q) centers of the loops via percentile math
    iq - array[n_loops, n_samples]
    returns centers[iq.shape[0]]

    see mkidgen2.roach2controls.fitLoopCenters for history
    """
    ictr = (np.percentile(iq.real, 95, axis=1) + np.percentile(iq.real, 5, axis=1)) / 2
    qctr = (np.percentile(iq.imag, 95, axis=1) + np.percentile(iq.imag, 5, axis=1)) / 2

    return ictr + qctr * 1j


def optimize_comb_power(comb, initial_global_atten, iq_sample_bits, MAX_GLOBAL_ATTENUATION):
    globalDacAtten = initial_global_atten
    # Calculate relative amplitudes for DAC LUT
    maxAmp = int(np.round(2 ** (iq_sample_bits - 1) - 1))  # 1 bit for sign

    iq = comb['iq']

    highestVal = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
    dBexcess = 20 * np.log10(highestVal / maxAmp)
    dBexcess = np.ceil(4 * dBexcess) / 4  # rounded up to nearest 1/4 dB
    globalDacAtten -= dBexcess
    # reduce to fit into DAC dynamic range and quantize to integer

    if globalDacAtten > MAX_GLOBAL_ATTENUATION * 2:
        dB_reduce = globalDacAtten - MAX_GLOBAL_ATTENUATION * 2
        getLogger(__name__).warning(f"Unable to fully utilize DAC dynamic range by {dB_reduce} dB")
        globalDacAtten -= dB_reduce
        dBexcess += dB_reduce
    elif np.max((np.abs(iq.real).max(), np.abs(iq.imag).max())) > maxAmp:
        dBexcess += 0.25  # Since there's some rounding there's a small chance we need to decrease by another atten step

    iq /= 10 ** (dBexcess / 20)
    iq.round(out=iq)

    return comb, globalDacAtten

def find_random_phase_waveform(freq, amplitudes, n_samples, sample_rate, spike_percentile_limit,
                               iq_ratios=None, phase_offsets=None):
    # check that we are utilizing the dynamic range of the DAC correctly
    # 10% of the time there should be a point this many sigmas higher than average
    rstate = np.random.get_state()
    from numpy.random import MT19937, RandomState, SeedSequence
    np.random.set_state(RandomState(MT19937(SeedSequence(123456789))).get_state())
    comb = mkidgen3.daccomb.generate_dac_comb(frequencies=freq, n_samples=n_samples, sample_rate=sample_rate,
                                      amplitudes=amplitudes)
    iq= comb['iq']
    sig_i, sig_q = iq.real.std(), iq.imag.std()

    expectedmax_sig = scipy.special.erfinv((iq.size + spike_percentile_limit - 1) / iq.size) * np.sqrt(2)
    if spike_percentile_limit < 1 and sig_i > 0 and sig_q > 0:
        while max(np.abs(iq.real).max() / sig_i, np.abs(iq.imag).max() / sig_q) >= expectedmax_sig:
            getLogger(__name__).warning("The freq comb's relative phases may have added up sub-optimally. Recomputing.")
            comb = mkidgen3.daccomb.generate_dac_comb(freq, n_samples, sample_rate, amplitudes=amplitudes,
                                                phases=None, iq_ratios=iq_ratios, phase_offsets=phase_offsets,
                                                return_merged=True)
            iq = comb['iq']

    np.random.set_state(rstate)

    getLogger(__name__).debug(msg)

    return comb, expectedmax_sig
