import zmq
ctx = zmq.Context.instance()
ctx.linger = 0
import logging
logging.basicConfig(level=logging.DEBUG)
from mkidgen3.feedline_objects import CaptureRequest, FeedlineConfig
from mkidgen3.feedline_client_objects import CaptureJob, PowerSweepJob

# cap command default 8888
# cap data 8889
# cap status 9000

feedline_server = 'tcp://localhost:8888'
capture_data_server = 'tcp://localhost:8889'
status_server = 'tcp://localhost:8890'

#start a listner for status
# pd = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
# pd.bind_in(f'tcp://localhost:9000')
# pd.bind_out('inproc://cap_status')
# # pd.setsockopt_in(zmq.SUBSCRIBE, b"B")
# pd.start()

#start a listner for data

fc = FeedlineConfig()
cr = CaptureRequest(1337, 'iq', fc)
cj = CaptureJob(cr, feedline_server, capture_data_server, status_server, submit=False)
cj.submit()


ps = PowerSweepJob()
jobs = ps.generate_jobs(submit=True)
for j in jobs:
    j.data()   # Will block until data is ready
