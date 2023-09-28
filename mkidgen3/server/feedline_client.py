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

#Start the capture of photons from an MKID array, assume FeedlineReadoutServers are running on all the boards
feedline_server_urls = []
data_server_urls = []
status_server_urls = []

#Generate a suitable FeedlineConfig for each server
fc = FeedlineConfig() # TODO
buff_duration_ms = 100
jobs = []
for a,b,c in zip(feedline_server_urls, data_server_urls, status_server_urls):
    cr = CaptureRequest(buff_duration_ms, 'photons', fc, feedline_server=a)
    jobs.append(CaptureJob(cr, a, b, c, submit=True))




cap_data_urls = []
from zmq.devices import ThreadDevice

data_server_internal = 'inproc://cap_data.xpub'
# Set up a proxy for routing all the capture requests
dtd = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
dtd.setsockopt_in(zmq.LINGER, 0)
dtd.setsockopt_out(zmq.LINGER, 0)
for url in cap_data_urls:
    dtd.connect_in(url)
dtd.bind_out(data_server_internal)
dtd.daemon = True
dtd.start()
getLogger(__name__).info(f'Relaying all capture data from {cap_data_urls} to {data_server_internal}')
