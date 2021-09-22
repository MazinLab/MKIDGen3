from logging import getLogger

import numpy as np
from pynq import allocate

from mkidgen3.mkidpynq import FP16_15
from pynq import(DefaultIP)

class DACTableAXIM(DefaultIP):
    bindto = ['mazinlab:mkidgen3:dac_table_axim:0.6']

    def __init__(self, description):
        super().__init__(description=description)
        self._buffer=None

    def replay(self, data, tlast=True, tlast_every=256, length=None, start=True, fpgen=lambda x: FP16_15(x)):
        """
        data - an array of complex numbers, nominally 2^18 samples
        tlast - Set to true to emit a tlast pulse every tlast_every transactions
        length - Number of groups of 16 (DAC takes 16/clock) in the data to use.
            e.g. if data[N].reshape(N//16,16). Note that the user channel will count to length-1
        fpgen - set to a function to convert floating point numbers to integers. if None data will be truncated.
        start - start the replay immediately
        """
        # Data has right shape
        if data.size < 2 ** 18:
            getLogger(__name__).warning('Insufficient data, padding with zeros')
            x = np.zeros(2 ** 18, dtype=np.complex64)
            x[:data.size] = data
            data = x

        if data.size > 2 ** 18:
            getLogger(__name__).warning('Data will be truncated to 2^18 samples')
            data = data[:2 ** 18]

        # Process Length
        if length is None:
            length = data.size // 8
        length = int(length)
        if length > 2 ** 15:  # NB 15 as length is in sets of 8
            getLogger(__name__).warning('Clipping replay length to 2^15 sets of 8')
        if length < 1:
            raise ValueError('Replay length must be at least 1')

        # Process tlast_every
        tlast_every = int(tlast_every)

        if tlast and not 1 <= tlast_every <= length:
            raise ValueError('tlast_every must be in [1-length] when tlast is set')
        if not tlast:
            tlast_every = 256  # just assign some value to ensure we didn't get handed garbage

        self._buffer = allocate(2 ** 20, dtype=np.uint16)
        self._buffer[:data.size * 2:2] = [fpgen(x) for x in data.real] if fpgen is not None else data.real
        self._buffer[1:data.size * 2:2] = [fpgen(x) for x in data.imag] if fpgen is not None else data.imag

        self.register_map.a1 = self._buffer.device_address
        self.register_map.length_V = length
        self.register_map.tlast = bool(tlast)
        self.register_map.replay_length_V = tlast_every
        self.register_map.run = True
        if start:
            self.register_map.CTRL.AP_START = 1

    def stop(self):
        self.register_map.run = False

    def start(self):
        #TODO probably need to check for the case where not idle and run = False (axis stall to completion)
        if self._buffer is None:
            raise RuntimeError('Must call replay first to configure the core')
        self.register_map.CTRL.AP_START = 1