import zmq

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://128.111.23.124:5555")
    socket.send_string('adc')
    capture = socket.recv_pyobj()
    socket.send_string('stop')
    print(socket.recv_string())