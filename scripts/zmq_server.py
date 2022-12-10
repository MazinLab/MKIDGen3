from time import sleep

import mkidgen3 as g3
import zmq
import time

import mkidgen3.overlay_helpers
from mkidgen3.objects import *
from mkidgen3.objects import CaptureRequest


def do(socket, ol):
    data = socket.recv_string()
    if data == "iq":
        # data = ol.capture.capiq(2 ** 19)
        socket.send_pyobj('Not supported')
    elif data == "powersweep":
        pending_captures.extend([CaptureRequest(2 * 19, ol) for _ in range(100)])
        bin2res=
    elif data == 'stop':
        socket.send_string('Stopping server')
        run=False


def capture_arbitrator(capture_request: CaptureRequest, pl_ddr4_max_bytes=2147000000):
    """

    Args:
        capture_request (CaptureRequest): a requested capture with no knowledge of memory bandwidth limitations
        pl_ddr4_max_bytes (int): max bytes to utilize in Pl DRAM before copying data to PS RAM and sending it off the board

    Returns:
        list of capture requests with size comptible with memory bandwidth limitations
    """
    if pl_ddr4_max_bytes > 4095000000:
        raise ValueError('Pl DDR4 is only 4 GiB')

    return capture_request


def command_available(socket):
    return bool(socket.peek())


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    print('Binding to port 5555')
    socket.bind("tcp://*:5555")
    sleep(1)

    ol = mkidgen3.overlay_helpers.configure('/home/xilinx/bit/cordic_16_15_fir_22_0.bit', clocks=False,
                                            external_10mhz=False, ignore_version=True, download=False)
    xmitThread = None
    capture = None
    last = None
    pending_captures = []
    run = True
    while run:
        if command_available(socket):
            do(socket, ol)
        if capture and capture.complete and (last is None or last.sent()):
            del last
            last=capture
            capture.send(socket)
            capture=None
        if not capture and pending_captures:
            try:
                capture=pending_captures.pop(0)
                capture.start()
            except IndexError:                pass
        time.sleep(.01)
