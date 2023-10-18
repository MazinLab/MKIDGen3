import zmq
from mkidgen3.server.feedline_config import IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig
from mkidgen3.server.feedline_client_objects import CaptureJob, FRSClient, CaptureRequest
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.opfb import opfb_bin_number
import numpy as np

# ctx = zmq.Context.instance()
# ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://mkidrfsoc4x2.physics.ucsb.edu:8888'
capture_data_server = 'tcp://mkidrfsoc4x2.physics.ucsb.edu:8889'
status_server = 'tcp://mkidrfsoc4x2.physics.ucsb.edu:8890'

frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top.bit', ignore_version=True)

# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz

# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)

# IF Config
if_board = IFConfig(lo=3000, adc_attn=20, dac_attn=20)

# DAC Config
waveform_vals = WaveformFactory(n_uniform_tones=512)
waveform = WaveformConfig(waveform=waveform_vals)
freqs = waveform.waveform.freqs

# Bin2Res Config
bins = np.zeros(2048, dtype=int)
bins[:freqs.size] = opfb_bin_number(freqs, ssr_raw_order=True)
chan = ChannelConfig(frequencies=bins)

# DDC Config
ddc_tones = np.zeros(2048)
ddc_tones[:freqs.size]=freqs
ddc = DDCConfig(tones=ddc_tones)

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc, if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)

cr = CaptureRequest(1024, 'adc', fc, frsa)
j = CaptureJob(cr)
j.submit(True, True)
