import threading
import numpy as np
import copy
import zmq
import matplotlib.pyplot as plt

from mkidgen3.server.feedline_config import (IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig,
                                             ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig)
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest, StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.util import setup_logging

setup_logging('feedlineclient')
# ctx = zmq.Context.instance()
# ctx.linger = 0


frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

large_job_test = False
send_wave = ''
frsu = frsa


def synthetic_photon_waveform_generator():
    import importlib.resources
    pulse_file = importlib.resources.path('mkidgen3.config', 'waveform_phase_pulse.npy')
    theta = np.load(pulse_file)
    theta *= 2 * np.pi
    theta2 = theta.copy()
    theta[200_000:] += theta[:-200_000] / 2
    theta[100_000:] += theta[:-100_000] / 2
    theta2[150_000:] += theta2[:-150_000] / 3

    tones = np.array([250.0e6 + 107e3, 2.00062500e+08])
    amplitudes = np.full(tones.size, fill_value=0.8 / tones.shape[0])
    phases = [-theta, -theta2]
    waveform = WaveformConfig(waveform=WaveformFactory(frequencies=tones, amplitudes=amplitudes,
                                                       phases=phases, maximize_dynamic_range=False))
    return waveform


# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)

# RFDC Clocking Config, clock source should default to external 10 MHz
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
if_board = IFConfig(lo=6000, adc_attn=10, dac_attn=50)

# DAC Config
waveforms = {}
waveforms['fake_photon'] = synthetic_photon_waveform_generator()
waveforms['2tone'] = WaveformConfig(waveform=WaveformFactory(frequencies=[250.0e6 + 107e3, 2.00062500e+08]))

# Channel config
chan = waveforms['fake_photon'].default_channel_config
ddc = waveforms['fake_photon'].default_ddc_config
thresholds = -0.5 * np.ones(2048)
thresholds[2:] = -0.99

# holdoffs = np.full(2048, fill_value=20, dtype=np.uint16)
trig = TriggerConfig(holdoffs=[20] * 2048, thresholds=thresholds)

# Feedline Config
for k in waveforms:
    if send_wave == 'hash':
        waveforms[k] = waveforms[k].hashed_form
    elif send_wave == 'computed':
        waveforms[k] = waveforms[k].output_waveform  # trigger waveform computation

fc_adconly = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc)

test_adc_only_job = CaptureJob(CaptureRequest(2 ** 19, 'adc', fc_adconly, frsu))

fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    filter=FilterConfig(coefficients='unity20'),
                    if_board=if_board, waveform=waveforms['fake_photon'], chan=chan, ddc=ddc,
                    trig=trig)

test_large_file_jobs = list(map(CaptureJob, (CaptureRequest(5 * 1024 ** 3 // 4, 'adc', fc, frsu,
                                                            file='file:///nfs/wheatley/adc5GiB.npz'),
                                             CaptureRequest(5 * 1024 ** 3 // 4 // 2048, 'ddciq', fc, frsu,
                                                            file='file:///nfs/wheatley/iq5GiB.npz'),
                                             CaptureRequest(5 * 1024 ** 3 // 2 // 2048, 'filtphase', fc, frsu,
                                                            file='file:///nfs/wheatley/phase5GiB.npz'))))

test_eng_jobs = list(map(CaptureJob, (CaptureRequest(1024, 'adc', fc, frsu),
                                      CaptureRequest(1024, 'ddciq', fc, frsu),
                                      CaptureRequest(1024, 'filtphase', fc, frsu))))

# gsm = StatusListener(b'', frsb.status_url)


test_adc_only_job.submit(True, True)

for j in test_eng_jobs:
    j.submit(True, True)

if large_job_test:
    for j in test_large_file_jobs:
        j.submit(True, True)


def hammer(sendhashed=False, precomputewave=True):
    """Run several tests in rapid succession (e.g. emulating a powersweep)"""
    waveform = WaveformConfig(waveform=WaveformFactory(n_uniform_tones=512))
    if precomputewave:
        waveform.waveform.output_waveform
    bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)
    rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop',
                                  clock_source=None)  # clock source should default to external 10 MHz
    rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
    if_board = IFConfig(lo=6000, adc_attn=50, dac_attn=50)
    chan = waveform.default_channel_config
    ddc = waveform.default_ddc_config
    filtercfg = FilterConfig(coefficients=f'unity{2048}')
    trigconfig = TriggerConfig()

    fc = FeedlineConfig(bitstream=bitstream,rfdc_clk=rfdc_clk,rfdc=rfdc,if_board=if_board,
                        waveform=waveform,chan=chan,ddc=ddc,filter=filtercfg,trig=trigconfig)
    steps=np.arange(0,400,10)
    for i, step in enumerate(steps):
        print(f'\nHammering: {i+1}/{len(steps)}\n')
        fc.if_board = IFConfig(lo=if_board.lo + step)
        j = CaptureJob(CaptureRequest(1024, 'ddciq', fc, frsu, channels=None))
        try:
            j.submit(True, True)
            d = j.data(60 * 3).data
            np.mean(d.real, axis=0), np.mean(d.imag, axis=0), np.std(d.real, axis=0), np.std(d.imag, axis=0)
        except KeyboardInterrupt:
            j.cancel()
            raise KeyboardInterrupt
        if sendhashed and i==0:
            fc = fc.hashed_form

hammer(sendhashed=False, precomputewave=True)
hammer(sendhashed=True, precomputewave=True)


channels_plt = [0, 1]

# Capture phase with no offsets
no_offset_ddc = copy.copy(ddc)
no_offset_ddc.center_relative = False
no_offset_ddc.quantize = True

j = CaptureJob(CaptureRequest(2 ** 19, 'filtphase', fc, frsu, channels=channels_plt))
j.submit(True, True)

# Compute Phase Needed to Move average to Zero
phase = j.data()
phase_offsets = -phase.data.mean(axis=0)
phase_offsets[2:] = 0

# Capture phase with new offsets
offset_ddc = copy.copy(no_offset_ddc)
offset_ddc.phase_offset = phase_offsets
offset_fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                           filter=FilterConfig(coefficients='unity20'),
                           if_board=if_board, waveform=waveforms['fake_photon'], chan=chan, ddc=offset_ddc,
                           trig=trig)

j2 = CaptureJob(CaptureRequest(2 ** 10, 'filtphase', offset_fc, frsu, channels=channels_plt))
j2.submit(True, True)
moved_phase = j2.data()

# Capture Data
fig, axes = plt.subplots(1, len(channels_plt), figsize=(15, 5))
for i, ax in zip(channels_plt, axes.T):
    plt.sca(ax)
    plt.plot(phase.data[:600, i] / np.pi, label='unbiased')
    plt.plot(moved_phase.data[:600, i] / np.pi, label='biased')
    plt.ylim(-1, 1)
    plt.ylabel('Phase (Scaled Radians)')
    plt.xlabel('Samples (Scaled Radians)')
axes.ravel()[-1].legend()
plt.show()

postage_job = CaptureJob(CaptureRequest(6730, 'postage', offset_fc, frsu, channels=[0, 1, 2]))
photon_job = CaptureJob(CaptureRequest(1200, 'photon', offset_fc, frsu))

postage_job.submit(True, True)
photon_job.submit(True, True)
threading.Timer(15, lambda: photon_job.cancel(False, False)).start()

# Postage
monitor_channels = [0, 1, 2]
postage = postage_job.data(timeout=5)  # parallel get_postage
for i in monitor_channels:
    print(f'Got {(postage.ids == i).sum()} photons in channel {i}')

fig, ax = plt.subplots(2, 4, figsize=(14, 10))
for i, (chan, ax) in enumerate(zip(monitor_channels, ax.ravel())):
    plt.sca(ax)
    plt.title(f'Chan {chan}')
    plt.imshow(np.abs(postage.iqs[postage.ids == chan][:200]), origin='lower')
    plt.ylabel('Event #')
    plt.xlabel('t (us)')
plt.tight_layout()
plt.show()

photons = photon_job.data(timeout=20)
from mkidgen3.server.aggregator import make_image, PixelMap
map = PixelMap(np.arange(2048, dtype=int), 2048)
image = make_image(map, photons.data, timestep_us=1000)

photons.data.shape,set(photons.data['phase']/(2**15-1)), set(np.diff(photons.data['time']))



print(0)

# Visualize Thresholds and Holdoffs
# x = ol.capture.capture_phase(2**10, [0,1], tap_location='filtphase')
# phase = np.array(x)/(2**15-1)
# x.freebuffer()
# chan_plt=[0,1]
# fig, axes = plt.subplots(1, len(chan_plt), figsize=(15,5))
# samples = np.arange(128)
# for j,(i, ax) in enumerate(zip(chan_plt,axes.T)):
#     trig = phase[:128,i]<thresholds[i]
#     holdoff = slice(samples[trig][i], samples[trig][i]+holdoffs[i])
#     plt.sca(ax)
#     plt.plot(samples, phase[:128,i], label='phase')
#     plt.plot(samples[trig][i], phase[:128,i][trig][i], marker='x', label='trigger')
#     plt.axhline(thresholds[i], linestyle='--', color='black', label='threshold')
#     plt.plot(samples[holdoff], phase[:128,i][holdoff], color='cyan', label='holdoff')
#     plt.title(f'Channel {i}')
#
#     plt.ylim(-1, 1)
#     plt.ylabel('Phase (Scaled Radians)')
#     plt.xlabel('Samples (Scaled Radians)')
#     if j==0:
#         ax.legend()


plt.plot(accum[1]['time'][:20], "o");
plt.xlabel('Photon')
plt.ylabel('timestamp')
slept = time.perf_counter()
x = accum.image(timestep_us=1000)
slept = time.perf_counter() - slept
print(f'Imaged in {slept * 1e3:.1f} ms, {accum[1].size * 12 / 1024 ** 2:.2f} MiB in chan1')
# No droped photons, but depends on what the threshold was set to!
x.shape, set(accum[1]['phase'] / (2 ** 15 - 1)), set(np.diff(accum[0]['time'])), set(np.diff(accum[0]['time'])) == {128}
