import mkidgen3 as g3
import numpy as np
from mkidgen3.drivers.ifboard import IFBoard
from logging import getLogger, basicConfig
import matplotlib.pyplot as plt

#basicConfig()
#getLogger("mkidgen3.drivers.ifboard").setLevel("DEBUG")


bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'

ol = g3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)
# Connect IF Board
if_board = IFBoard(connect=True)
if_board.power_off()
if_board.power_on()
# SET LO TO GET MKID FREQUENCY
if_board.set_lo(5960)
if_board.set_attens((31.75,31.75),(20,20))
a = if_board.status()

tones = np.array([532.435e6])
amplitudes = np.ones_like(tones)/tones.shape[0]
g3.set_waveform(tones,amplitudes,fpgen='simple')

N = 2**19 # Number of samples to capture (full waveform)
Fs = 4.096e9 # ADC Sample Rate [Hz]
Tc = N/Fs # total collection time (seconds)
# Trigger Capture
adc_capture_data1 = ol.capture.capture_adc(N, complex=True)
adc_capture_data1/=2**16  #Normalize to 1/2 V
# Plot ADC Data

#if_board.set_attens((20,20),(20,20))

adc_capture_data2 = ol.capture.capture_adc(N, complex=True)
adc_capture_data2/=2**16

data1 = 20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(adc_capture_data1))))
data2 = 20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(adc_capture_data2))))

plt.plot(np.linspace(-2.048e9,2.048e9,2**19), data1 - data1.max())
plt.plot(np.linspace(-2.048e9,2.048e9,2**19), data2 - data1.max())
plt.grid(True)
#plt.xlim([200e6,700e6])
plt.show()


