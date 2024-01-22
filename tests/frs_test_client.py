import zmq
from mkidgen3.server.feedline_config import IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest,StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.opfb import opfb_bin_number
import numpy as np
from mkidgen3.util import setup_logging
setup_logging('feedlineclient')
# ctx = zmq.Context.instance()
# ctx.linger = 0

frs = 'a'

if frs == 'a':
    server = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
elif frs == 'b':
    server = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
# Status listener
gsm = StatusListener(b'', server.status_url)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)
# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz
# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
# IF Config
if_board0 = IFConfig(lo=6000, adc_attn=50, dac_attn=50)
if_board1 = IFConfig(lo=7000, adc_attn=40, dac_attn=30)
if_cfgs = [if_board0, if_board1]
# DAC Config
waveform0 = WaveformConfig(waveform=WaveformFactory(n_uniform_tones=512))
waveform1 = WaveformConfig(waveform=WaveformFactory(n_uniform_tones=1024))
output_waveform = waveform0.waveform.output_waveform # trigger waveform computation
output_waveform = waveform1.waveform.output_waveform # trigger waveform computation
wvfm_cfgs = [waveform0, waveform1]
# Bin2Res Config
chan0 = waveform0.default_channel_config
chan1 = waveform1.default_channel_config
b2r_cfgs = [chan0, chan1]
# DDC Config
ddc0 = waveform0.default_ddc_config
ddc1 = waveform1.default_ddc_config
ddc_cfgs = [ddc0, ddc1]
# Feedline Config
fcs = []
for i in [0, 1]:
    fcs.append(FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_cfgs[i], waveform=wvfm_cfgs[i], chan=b2r_cfgs[i], ddc=ddc_cfgs[i]))

crs = []
crs.append(CaptureRequest(2**14, 'adc', fcs[0], server))
crs.append(CaptureRequest(2**34, 'adc', fcs[0], server))
crs.append(CaptureRequest(2**7, 'adc', fcs[1], server))
crs.append(CaptureRequest(2**14, 'ddciq', fcs[0], server))
crs.append(CaptureRequest(2, 'ddciq', fcs[0], server))
crs.append(CaptureRequest(2, 'ddciq', fcs[0], server, channels=[0, 2, 3]))
crs.append(CaptureRequest(2**16, 'ddciq', fcs[1], server, channels=[0, 2, 3]))
crs.append(CaptureRequest(2**14, 'filtphase', fcs[0], server))


#cr = CaptureRequest(1024**3//4//2048, 'iq', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/iq1024MiB.npz')
#cr = CaptureRequest(1024**3//2//2048, 'phase', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/phase1024MiB.npz')
#cr = CaptureRequest(3024**3//2//2048, 'phase', fc, frsa)
#cr = CaptureRequest(2**19, 'adc', fc, frsa)

j = CaptureJob(crs[-1])
j.submit(True, True)

print('hi')
print('hi')
