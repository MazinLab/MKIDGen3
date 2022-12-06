from time import sleep

import numpy as np

import mkidgen3 as g3
import zmq
import threading
import time

from mkidgen3.configuration_objects import *



class CaptureRequest:
    def __init__(self, n, if_setup: IFSetup, pipe_setup, dac_setup: DACOutputSpec, channel_spec, ):
        self.ol =ol
        self._buffer=None
        self._thread=None
        self.points = n
        self.tap_location=None
        self.points=n
        self.if_setup=if_setup
        self.pipe_setup=pipe_setup
        self.dac_setup=dac_setup
        self.channel_spec=channel_spec


    @property
    def complete(self):
        return self.ol.capture.axis2mm.complete

    def start(self):
        g3.apply_setup(ifsetup=self.if_setup, pipe=self.pipe_setup, dac=self.dac_setup)
        self._buffer = self.ol.capture.capture_adc(2 ** 19, complex=False, sleep=False, use_interrupt=False)

    def send(self, socket):
        def send_data(socket, capture_data):
            socket.send_pyobj(capture_data)
        self._thread = threading.Thread(target=send_data, args=(socket, self._buffer.copy()), daemon=True,
                                        name=f'CapXmit: {self}')
        self._thread.start()

    def sent(self):
        return self._thread is not None and not self._thread.is_alive()

    def __del__(self):
        if self._buffer is not None:
            self._buffer.free()


class PowerSweepRequest:
    def __init__(self, natten, nfreq):
        self.natten=natten
        self.nfreq=nfreq
        self.samples=2**19
        self.attens=np.linspace(20).reshape((2,10))
        self.bandwidth=BANDWIDTH
        self.lo_centers = compute_lo_steps(center=0, resolution=7.14e3, bandwidth=self.bandwidth)

    def capture_requests(self):
        dacsetup=DACOutputSpec('regular_normalized_comb', n_tines=200)
        return [CaptureRequest(self.samples, dac_setup=dacsetup,
                               if_setup=IFSetup(lo=freq, adc_attn=adc_atten,dac_attn=dac_atten))
                for (adc_atten,dac_atten) in self.attens for freq in self.lo_centers]


def do(socket, ol):
    data = socket.recv_string()
    if data == "iq":
        # data = ol.capture.capiq(2 ** 19)
        socket.send_pyobj('Not supported')
    elif data == "powersweep":
        pending_captures.extend([CaptureRequest(2*19, ol) for _ in range(100)])
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

    ol = g3.configure('/home/xilinx/bit/cordic_16_15_fir_22_0.bit', clocks=False,
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
            except IndexError:
                pass
        time.sleep(.01)



