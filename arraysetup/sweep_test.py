import zmq
from mkidgen3.server.feedline_config import IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, DDCConfig, FeedlineConfig
from mkidgen3.server.feedline_client_objects import CaptureJob, FRSClient, CaptureRequest
from mkidgen3.server.waveform import WaveformFactory

# ctx = zmq.Context.instance()
# ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8888'
capture_data_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8889'
status_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8890'

frs = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top.bit', ignore_version=True)
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None) # clock source should default to external 10 MHz
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
if_board = IFConfig(lo=3000, adc_attn=20, dac_attn=20)
waveform_vals = WaveformFactory(n_uniform_tones=512)
waveform = WaveformConfig(waveform=waveform_vals)
chan = waveform.default_channel_config
ddc = DDCConfig(tones=chan.frequencies)
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc, if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)

cr = CaptureRequest(1024, 'adc', fc, frs)
j = CaptureJob(cr)
