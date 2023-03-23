import mkidgen3.overlay_helpers
from mkidgen3.funcs import *
import mkidgen3 as g3

bitstream='/home/xilinx/jupyter_notebooks/gen3top_benchmark_0831/cordic_16_15_fir_22_0.bit'

ol = mkidgen3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)

template_comb_freqs = bin_center_freqs()
template_waveform = optimize_random_phase(template_comb_freqs, n_samples=2**19, sample_rate=4.096e9, amplitudes=None, phases=None, iq_ratios=None,
                      phase_offsets=None, seed=2, max_quant_err=predict_quantization_error(),
                          timeout=10, return_quantized=True)
