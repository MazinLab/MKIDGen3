import zmq
from mkidgen3.server.feedline_client_objects import CaptureRequest, CaptureJob
from mkidgen3.server.feedline_objects import IFConfig, DACConfig, PhotonPipeConfig, ChannelConfig, DDCConfig, FeedlineConfig


#ctx = zmq.Context.instance()
#ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://localhost:8888'
capture_data_server = 'tcp://localhost:8889'
status_server = 'tcp://localhost:8890'

if_config = IFConfig(lo=3000, adc_attn=20, dac_attn=20)
dac_config = DACConfig(n_uniform_tones=512)

frequencies = dac_config._waveform.freqs

chan_config = ChannelConfig(frequencies)
ddc_config = DDCConfig(tones=frequencies)
pp_config = PhotonPipeConfig(chan_config, ddc_config)

fl_config = FeedlineConfig(if_config, dac_config, pp_config)


#test_cr = CaptureRequest(1024, 'adc', fl_config, feedline_server)