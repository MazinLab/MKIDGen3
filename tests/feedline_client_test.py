from mkidgen3.objects import *

# cap command default 8888
# cap data 8889
# cap status 9000

capture_port

#start a listner for status
pd = ThreadDevice(zmq.QUEUE, zmq.XSUB, zmq.XPUB)
pd.bind_in(f'tcp://localhost:9000')
pd.bind_out('inproc://cap_status')
# pd.setsockopt_in(zmq.SUBSCRIBE, b"B")
pd.start()

#start a listner for data

s=''

cr = CaptureRequest(1337, 'adc', FeedlineConfig(), 'localhost:8888', 'localhost:9000')
cj = CaptureJob(cr, 'localhost:8889', 'inproc://cap_status', submit=False)
