from logging import getLogger
import zmq
from mkidgen3.server.feedline_config import IFConfig, FeedlineConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest, StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.server.feedline_config import FeedlineConfigManager
import numpy as np
from mkidgen3.util import setup_logging
setup_logging('feedlineclient')

frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)
# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source='external')
# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
# IF Config
if_board = IFConfig(lo=6000, adc_attn=50, dac_attn=50)

#feedline config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=WaveformConfig(), chan=ChannelConfig(), ddc=DDCConfig(),
                    filter=FilterConfig(), trig=TriggerConfig())

one_gib_adc = 2**28
four_gib_adc = 4*2**28

size = four_gib_adc
tap = 'adc'
cr = CaptureRequest(n=4*2**28, tap=tap, feedline_config=fc, feedline_server=frsa)
j = CaptureJob(cr)
j.submit(True, True)
result = j.data(timeout=300)
print('hi')
