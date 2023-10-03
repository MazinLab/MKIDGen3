#import zmq
#from mkidgen3.server.feedline_client_objects import CaptureRequest, CaptureJob, FeedlineConfig
from mkidgen3.server.feedline_objects import IFConfig, DACConfig


#ctx = zmq.Context.instance()
#ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://localhost:8888'
capture_data_server = 'tcp://localhost:8889'
status_server = 'tcp://localhost:8890'

ifconfig = IFConfig(lo=3000, adc_attn=20, dac_attn=20)
dacconfig = DACConfig(n_uniform_tones=512)



#fc = FeedlineConfig(ifconfig,)

#test_cr = CaptureRequest(1024, 'adc', fc, feedline_server)