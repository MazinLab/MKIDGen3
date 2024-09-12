
import os
from mkidanalysis import QuasiparticleTimeStream, Resonator, RFElectronics, FrequencyGrid, ReadoutPhotonResonator
import numpy as np
import matplotlib.pyplot as plt

# Generate a timestream proportional to the change in quasiparticle density
quasiparticle_timestream = QuasiparticleTimeStream(fs = 4.096e9, ts = (2**19)/4.096e9)
# Define a sudden change in quasiparticle density (caused by a photon)
quasiparticle_timestream.gen_quasiparticle_pulse(tf=100) # 30 is realistic
quasiparticle_timestream.data[quasiparticle_timestream.data.shape[0]//4:quasiparticle_timestream.data.shape[0]//4 + quasiparticle_timestream.photon_pulse.shape[0]] = quasiparticle_timestream.photon_pulse

# Create resonator and compute S21
resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1) #1e2
rf = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9)
freq = FrequencyGrid( fc=4.0012e9, points=1000, span=500e6)

plt.figure()
quasiparticle_timestream.photon_pulse.shape
plt.plot(quasiparticle_timestream.data)

plt.figure()
lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)
theta1, d1 = lit_res_measurment.basic_coordinate_transformation()

plt.plot(theta1)
np.save("waveform_phase_pulse.npy", theta1)

# To load data into a waveform:
# theta = np.load("waveform_phase_pulse.npy")
# theta[200_000:] += theta[:-200_000] / 2
# theta[100_000:] += theta[:-100_000] / 2
# theta2 = np.load("waveform_phase_pulse.npy")
# theta2[150_000:] += theta2[:-150_000] / 3
# theta*=2*np.pi
# theta2*=2*np.pi
# tones = np.array([250.0e6 + 107e3, 2.00062500e+08])
# amplitudes = np.full(tones.size, fill_value=0.8/tones.shape[0])
# phases = [-theta, -theta2]
# waveform = WaveformConfig(waveform=WaveformFactory(frequencies=tones, amplitudes=amplitudes,
#                                                    phases=phases, maximize_dynamic_range=False))
