from .funcs import *
## Load Basic Configuration

## Find MKID Resonators:
# Find resonant frequency and saturation power. To do this we set the global attenuation
# on the IF board to set the input power and play a template waveform out of the DACs. We then step the LO
# frequency and capture ???? data. We step the LO through the entire range to sweep out all of the resonators
# and capture. This process is repeated for various drive powers by changing the IF board attenuation and or
# the waveform coefficient magnitude. This data is used via machine learning or click-throughs to determine
# the MKID resonator readout frequencies and powers.
connect_to_if_board()
template_waveform=compute_waveform(random_phases, powers=ones, frequencies=uniform_comb)
play_waveform(template_waveform)
for if_power in attenuations:
    for LO_position in sweep_centers:
        program_LO(LO_position)
        program_if_attenuation(if_power) # set power level at device
        capture_adc(N_samples)
        save_timeseries()
mkid_powers, mkid_frequencies = find_mkid_frequencies_and_powers("array of [sweep_centers*attenuations] by N_samples time serieses")
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
readout_waveform = compute_waveform(random_phases, mkid_powers, mkid_frequencies, mkid_relative_iq)
play_waveform(readout_waveform)
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

