import mkidgen3 as g3
import numpy as np
from mkidgen3.drivers.ifboard import IFBoard
from logging import getLogger, basicConfig
import matplotlib.pyplot as plt

#basicConfig()
#getLogger("mkidgen3.drivers.ifboard").setLevel("DEBUG")

# DOWNLOAD OVERLAY
bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'
ol = g3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)

# CONNECT IF BOARD
if_board = IFBoard(connect=True)
if_board.power_off()
if_board.power_on()

# SET LO TO GET MKID FREQUENCY
if_board.set_lo(5960)
if_board.set_attens((31.75,31.75),(20,20))

# PROGRAM DAC AND PLAY OUTPUT
tones = np.array([500e6])
amplitudes = np.ones_like(tones)/tones.shape[0] # CHANGE HERE IF SATURATING
g3.set_waveform(tones,amplitudes,fpgen='simple')






