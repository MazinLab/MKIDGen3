from time import sleep
import mkidgen3 as g3
import zmq

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    print('Binding to port 5555')
    socket.bind("tcp://*:5555")
    sleep(1)


    ol = g3.configure('/home/xilinx/bit/cordic_16_15_fir_22_0.bit', clocks=False, external_10mhz=False, ignore_version=True, download=False)

    while True:
        tap = socket.recv_string()
        if tap == "iq":
            data = ol.capture.capiq(2**19)
            socket.send_pyobj(data)
        if tap == "adc":
            data = ol.capture.capture_adc(2**19, complex=True)
            socket.send_pyobj(data)
        if tap == 'stop':
            socket.send_string('Stopping server')
            break