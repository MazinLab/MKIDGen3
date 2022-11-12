from time import sleep
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
print('Binding to port 5555')
socket.bind("tcp://*:5555")
sleep(1)

while True:
    tap = socket.recv_string()
    if tap == "iq":
        data = ol.capture.capiq()
        socket.send_pyobj(data)
    if tap == "adc":
        data = ol.capture.capture_adc()
        socket.send_pyobj(data)
    if tap == 'stop':
        socket.send_string('Stopping server')
        break