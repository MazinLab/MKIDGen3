import zmq
import logging
logging.basicConfig(level=logging.DEBUG)
from mkidgen3.server.feedline_objects import DACConfig, PhotonPipeConfig
from mkidgen3.server.feedline_client_objects import CaptureRequest, CaptureJob, FeedlineConfig


ctx = zmq.Context.instance()
ctx.linger = 0

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://localhost:8888'
capture_data_server = 'tcp://localhost:8889'
status_server = 'tcp://localhost:8890'


frequencies = []
coefficients = []
thresholds= []
holdoffs = []
fc = FeedlineConfig(dac_config=dict(frequencies=None, amplitudes=None, phases=None,
                                    iq_ratios=None, phase_offsets=None, maximize_dynamic_range=True),
                    pp_config=PhotonPipeConfig(chan_config=dict(frequencies=frequencies),
                                               ddc_config=dict(tones=None, loop_center=None,
                                                               phase_offset=None, center_relative=None, quantize=None),
                                               filter_config=dict(coefficients=coefficients),
                                               trig_config=dict(thresholds=thresholds, holdoffs=holdoffs)))
cr = CaptureRequest(1337, 'iq', fc)
cj = CaptureJob(cr, feedline_server, capture_data_server, status_server, submit=False)
cj.submit()
x = cj.data(timeout=5)  #should be almost
print(cj.status_history())
