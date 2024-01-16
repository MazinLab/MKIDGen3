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



frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)

# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz

# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)

# IF Config
if_board = IFConfig(lo=6000, adc_attn=10, dac_attn=50)

# DAC Config
waveform_vals = WaveformFactory(n_uniform_tones=2048)

waveform = WaveformConfig(waveform=waveform_vals)
output_waveform = waveform.waveform.output_waveform # trigger waveform computation

# Bin2Res Config
chan = waveform.default_channel_config

# DDC Config
ddc = waveform.default_ddc_config

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)
#fc2= FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
#                    if_board=if_board, waveform=waveform2, chan=chan, ddc=ddc,
#                    filter=FilterConfig(coefficients='unity20'),
#                    trig=TriggerConfig(holdoffs=[20]*2048, thresholds=[0]*2048))
gsm = StatusListener(b'', frsb.status_url)
cr = CaptureRequest(3*1024**3//4, 'adc', fc, frsb)
#cr = CaptureRequest(1024**3//4//2048, 'iq', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/iq1024MiB.npz')
#cr = CaptureRequest(1024**3//2//2048, 'phase', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/phase1024MiB.npz')
#cr = CaptureRequest(3024**3//2//2048, 'phase', fc, frsa)
#cr = CaptureRequest(2**19, 'adc', fc, frsa)

# j = CaptureJob(cr)
# j.submit(True, True)

print('hi')