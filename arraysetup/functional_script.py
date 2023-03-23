import mkidgen3.overlay_helpers
from mkidgen3.funcs import *
import mkidgen3 as g3
from mkidgen3.drivers.ifboard import IFBoard

bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'

ol = mkidgen3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)

n_channels=N_CHANNELS
bandwidth=BANDWIDTH

template_comb_freqs = uniform_freqs(n_channels=N_CHANNELS, bandwidth=BANDWIDTH)
template_waveform = optimize_random_phase(template_comb_freqs, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, seed=2, max_quant_err=predict_quantization_error(),
                          max_attempts=10, return_quantized=True)

#Now play the wavefrom
mkidgen3.overlay_helpers.play_waveform(template_waveform)

# IF Power Sweep Settings
lo_sweep_freqs = compute_lo_steps(center=0, resolution=7.14e3, bandwidth=BANDWIDTH) # TODO what is the right resolution
attenuations = compute_power_sweep_attenuations(0,30,0.25)

#now sweep
if_board = IFBoard(connect=True)
if_board.power_on()
for if_power in attenuations:
    for lo_position in lo_sweep_freqs:
        if_board.set_lo(lo_position[0] * 1e-6)
        program_if_attenuation(if_power) # set power level at device
        capture_adc(N_samples)
        save_timeseries()
play_waveform(template_waveform)
