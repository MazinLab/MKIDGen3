from .funcs import *
import mkidgen3 as g3
import scipy
from logging import basicConfig, getLogger
import mkidgen3.daccomb
from mkidgen3.ifboard import IFBoard
import scipy.special

## Load Basic Configuration

bitstream=''

g3.configure(bitstream, ignore_version=False, clocks=False, external_10mhz=False, download=True)

## Find MKID Resonators:
# Find resonant frequency and saturation power. To do this we set the global attenuation
# on the IF board to ??? to set the input power to ?? and play a template waveform out of the DACs.
# The template wavefrom needs to be ???. We then step the LO
# frequency and capture ???? data. We step the LO through the entire range to sweep out all the resonators
# and capture. This process is repeated for various drive powers by changing some to be determined combination
# of the the IF board attenuation and the waveform coefficient magnitude.
# This data is used via machine learning or click-throughs to determine
# the MKID resonator readout frequencies and powers.

sweep_centers =


#Determine random phases to generate template waveform with unity gain
freq=np.zeros()
WAVEFORM_LENGTH = 2**19
n_samples=0
SAMPLE_RATE = 4.096e9
spike_percentile_limit = ?
iq_ratios = None
phase_offsets = None
iq_sample_bits = 16
MAX_GLOBAL_ATTENUATION = 31.75
attenuations = np.ones_like(freq)


# Calculate relative amplitudes for DAC LUT
initial_global_atten = attenuations.min()
maxAmp = int(np.round(2 ** (iq_sample_bits - 1) - 1))  # 1 bit for sign
amplitudes = maxAmp * 10 ** (-(attenuations - initial_global_atten) / 20)
comb, expectedmax_sig = find_random_phase_waveform(freq, amplitudes, WAVEFORM_LENGTH, SAMPLE_RATE,
                                                   spike_percentile_limit, iq_ratios=iq_ratios,
                                                   phase_offsets=phase_offsets)
comb, global_atten = optimize_comb_power(comb, initial_global_atten, iq_sample_bits, MAX_GLOBAL_ATTENUATION)

# highestVal = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
# msg = ('\tGlobal DAC atten: {} dB'.format(globalDacAtten) +
#        '\tUsing {} percent of DAC dynamic range\n'.format(highestVal / maxAmp * 100) +
#        '\thighest: {} out of {}\n'.format(highestVal, maxAmp) +
#        '\tsigma_I: {}  sigma_Q:{}\n'.format(np.std(iq.real), np.std(iq.imag)) +
#        '\tLargest val_I: {} sigma. '.format(np.abs(iq.real).max() / np.std(iq.real)) +
#        'val_Q: {} sigma.\n'.format(np.abs(iq.imag).max() / np.std(iq.imag)) +
#        '\tExpected val: {} sigmas\n'.format(expectedmax_sig))
# getLogger(__name__).debug(msg)
# getLogger(__name__).debug(f"Comb shape: {comb.shape}.\n"
#                           f"Total Samples: {comb.size}. "
#                           f"Memory: {comb.size * 2 * iq_sample_bits / 8 / 1024 ** 2:.0f} MiB\n")


#Now play the wavefrom
g3.play_waveform(comb, **kwargs)

#now sweep
if_board = IFBoard(connect=True)
for if_power in attenuations:
    for lo_position in sweep_centers:
        if_board.set_lo(lo_position)
        if_board.set_attens(output_attens=, input_attens=) # set power level at device
        snap = g3._gen3_overlay.capture.capture_adc(n_samples, duration=False,
                                             complex=False, sleep=True,
                                             use_interrupt=False)
        np.savez(snap, f"snap_{if_power}_{lo_position}.npz")  # NB this is about ??? MiB

#End of initial data setup task


mkid_powers, mkid_frequencies = find_mkid_frequencies_and_powers(data_dir)
mkid_relative_iq = Neelay_IQ_Image_Optimization(mkid_powers, mkid_frequencies)



## Rotate and Center the Loops:
# Create a DAC waveform using a superposition of the previously determined, device / cool-down specific
# readout frequencies and powers (randomize phases, Neelay's IQ imbalance???)
# Play that waveform.
# Program the DDS to down convert each of those channels using the known frequencies. There is no centering
# or phase offset programed in the DDC yet. Only the frequencies.
# Program the matched filters to be doing nothing (unity)
# Capture the IQ loop data for each resonator. Determine the center of the loop.
# Reprogram the DDC with the centers so that the q centers for each channel are subtracted from the q values for each channel
# Capture the phase of each channel and determine the distance between the starting phase and zero.
# Reprogram the DDC with the phase offsets for each channel
# At this point an IQ capture should look like a centered loop with the resonance frequency area on the real axis
# in the IQ plane.



#Determine random phases to generate template waveform with unity gain
freq=mkid_frequencies
spike_percentile_limit = ?
iq_ratios = mkid_relative_iq
phase_offsets = None
attenuations = powers_to_attenuations(mkid_powers)


# Calculate relative amplitudes for DAC LUT
initial_global_atten = attenuations.min()

maxAmp = int(np.round(2 ** (iq_sample_bits - 1) - 1))  # 1 bit for sign
amplitudes = maxAmp * 10 ** (-(attenuations - initial_global_atten) / 20)
comb, expectedmax_sig = find_random_phase_waveform(freq, amplitudes, WAVEFORM_LENGTH, SAMPLE_RATE, spike_percentile_limit,
                                                   iq_ratios=iq_ratios, phase_offsets=phase_offsets)
comb, global_atten = optimize_comb_power(comb, initial_global_atten, iq_sample_bits, MAX_GLOBAL_ATTENUATION)

#Now play the wavefrom
g3.play_waveform(comb, **kwarg)



program_bin2res(opfb_bin_of_freq(mkid_frequencies))
program_ddc(mkid_frequencies)
program_matched_filters(unity)
raw_loops = capture_iq(all_resonator_channels)
centers = compute_center(raw_loops)
program_ddc(mkid_frequencies, centers)
raw_phases = capture_phase(all_resonator_channels)
ddc_phases = compute_ddc_phase(raw_phases)
program_ddc(mkid_frequencies, centers, ddc_phases)



verify_loops_are_centered_and_rotated(captures, plots, logging, etc)


## Generate Optimal Filter Coefficients
# Assuming that the no-photon phase, (randomized phase from previous step + some offset)
# IQ loop rotation, and center haven't changed since the loop rotation
# Generate and play a waveform using this data (correct frequency, power, and ddc rotation + centering).
# Optimal filters should still be programed to do nothing (unity)
# Take a XXX second phase capture on each channel and compute the noise PSD.
# Turn on light source to a laser in the middle of the desired wavelength range.
# Take a XXX second phase capture on each channel and generate the optimal filters.
play_waveform(readout_waveform)
program_bin2res(opfb_bin_of_freq(mkid_frequencies))
program_ddc(mkid_frequencies, centers, ddc_phases)
program_matched_filters(unity)
noise_traces = capture_phase(all_resonator_channels)
psds = compute_psd(noise_traces)
*turn on laser*
raw_pulse_traces = capture_phase(all_resonator_channels)
optimal_filters = compute_optimal_filters(psds, raw_pulse_traces)

## Set Trigger Threshold and Holdoff (By Channel):
# assuming waveform, ddc, etc. is still programmmed...
# keep laser source on (use either middle value or lowest energy value we want to detect)
# Program the optimal filters with their determined optimal value
# Capture phase time stream
# Compute the optimal trigger threshold (standard deviations away from the noise) and holdoff time
# for each channel.
play_waveform(readout_waveform)
program_bin2res(opfb_bin_of_freq(mkid_frequencies))
program_ddc(mkid_frequencies, centers, ddc_phases)
program_matched_filters(optimal_filters)
threshold_traces = capture_phase(all_resonator_channels)
thresholds, holdoffs = optimize_trigger(threshold_traces)
program_trigger(thresholds, holdoffs)

