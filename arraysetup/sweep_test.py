import zmq
from mkidgen3.server.feedline_config import IFConfig, DACConfig, ADCConfig, PhotonPipeConfig, DDCConfig, FeedlineConfig
from mkidgen3.server.feedline_client_objects import CaptureJob, FRSClient, CaptureRequest



#ctx = zmq.Context.instance()
#ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8888'
capture_data_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8889'
status_server = 'tcp://rfsoc4x2b.physics.ucsb.edu:8890'

frs_client = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

if_config = IFConfig(lo=3000, adc_attn=20, dac_attn=20)
dac_config = DACConfig(n_uniform_tones=512)
adc_config = ADCConfig()

chan_config = dac_config.default_channel_config
ddc_config = DDCConfig(tones=chan_config.frequencies)
pp_config = PhotonPipeConfig(chan_config=chan_config, ddc_config=ddc_config)

fl_config = FeedlineConfig(if_config=if_config, dac_config=dac_config, pp_config=pp_config, adc_config=adc_config)


cr = CaptureRequest(1024, 'adc', fl_config, frs_client)
j = CaptureJob(cr)
