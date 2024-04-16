import mkidgen3 as g3
from mkidgen3.equipment_drivers.ifboard import IFBoard
from mkidgen3.funcs import *
from logging import getLogger, basicConfig
import matplotlib.pyplot as plt

# basicConfig()
# getLogger("mkidgen3.equipment_drivers.ifboard").setLevel("DEBUG")

# DOWNLOAD OVERLAY
bitstream = '/home/xilinx/bit/cordic_16_15_fir_22_0.bit'
ol = g3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)

# CONNECT IF BOARD
if_board = IFBoard(connect=True)
if_board.power_off()
if_board.power_on()

# MKID PROPERTIES
mkid_f0 = 5.960e9 # Hz

# CONFIGURE IF BOARD
if_board.set_lo(mkid_f0*1e-6)  # set LO to sweep MKID frequency
if_board.set_attens(39.75, 23)

# PROGRAM DAC AND PLAY OUTPUT
tones = np.array([500e6])
amplitudes = np.ones_like(tones) / tones.shape[0]
g3.set_waveform(tones, amplitudes, fpgen='simple')

# ASSIGN RESONATOR CHANNELS
ol.photon_pipe.reschan.resonator_ddc = ol.photon_pipe.reschan.resonator_ddc_control
bins = np.zeros(2048, dtype=int)
bins[:tones.size] = g3.opfb_bin_number(tones, ssr_raw_order=True)
ol.photon_pipe.reschan.bin_to_res.bins = bins

# PROGRAM DDC
ddc_tones = np.zeros(2048)
ddc_tones[:tones.size] = tones
g3.configure_ddc(ddc_tones, quantize=True)


# CAPTURE IQ
def get_iq_point(n=256):
    """
    Args:
        n: int
        how many points to average
    Returns: a single averaged iq data point captured from res channel 0
    """
    x = ol.capture.capture_iq(n, [0, 1], tap_location='iq')
    tmp = np.zeros(x.shape[1:])
    x.mean(axis=0, out=tmp)
    del x
    return tmp[0, 0] + 1j * tmp[0, 1]


#lo_sweep_freqs = compute_lo_steps(center=0, resolution=7.14e3, bandwidth=1e6)
lo_sweep_freqs = compute_lo_steps(center=mkid_f0, resolution=20e3, bandwidth=1e6)

# SWEEP
iq_vals = np.zeros(lo_sweep_freqs.size, dtype=np.complex64)
for x in range(len(lo_sweep_freqs)):
    if_board.set_lo(lo_sweep_freqs[x] * 1e-6)
    iq_vals[x] = get_iq_point()

# PLOT SWEEP
def plot_sweep(freqs, iq_vals, ax=None, fig=None):
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{mkid_f0 * 1e-9} GHz Resonator', fontsize=15)

    ax1.plot(freqs * 1e-9, 20 * np.log10(np.abs(iq_vals)), linewidth=4)
    ax1.set_ylabel('|S21| [dB]')
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_title('Transmission')
    ax2.plot(iq_vals.real, iq_vals.imag, 'o')
    ax2.set_xlabel('Real(S21)')
    ax2.set_ylabel('Imag(S21)')
    ax2.set_title('IQ Loop')

plot_sweep(lo_sweep_freqs, iq_vals)
plt.show()

# CENTER LOOP
def est_loop_center(iq):
    """
    Finds the (I,Q) centers of the loops via percentile math
    iq - np.complex array[n_loops, n_samples]
    returns centers[iq.shape[0]]

    see mkidgen2.roach2controls.fitLoopCenters for history
    """
    ictr = (np.percentile(iq.real, 95) + np.percentile(iq.real, 5)) / 2
    qctr = (np.percentile(iq.imag, 95) + np.percentile(iq.imag, 5)) / 2

    return ictr + qctr * 1j

ddc_centers = np.zeros(2048, dtype=np.complex64)
ddc_centers[:tones.size]=est_loop_center(iq_vals/2**15)
g3.configure_ddc(ddc_tones, phase_offset=None, loop_center=ddc_centers, center_relative=False, quantize=True)

# RESWEEP AND VERIFY LOOP IS CENTERED

iq_vals_centered = np.zeros(lo_sweep_freqs.size, dtype=np.complex64)
for x in range(len(lo_sweep_freqs)):
    if_board.set_lo(lo_sweep_freqs[x] * 1e-6)
    iq_vals_centered[x] = get_iq_point()

plot_sweep(lo_sweep_freqs, iq_vals_centered/2**15)
