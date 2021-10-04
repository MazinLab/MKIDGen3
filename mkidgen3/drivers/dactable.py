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
        data - an array of complex numbers, nominally 2^19 samples
        tlast - Set to true to emit a tlast pulse every tlast_every transactions
        length - Number of groups of 16 (DAC takes 16/clock) in the data to use.
            e.g. if data[N].reshape(N//16,16). Note that the user channel will count to length-1
        fpgen - set to a function to convert floating point numbers to integers. if None data will be truncated.
        start - start the replay immediately
        """
        # Data has right shape
        if data.size < 2 ** 19:
            getLogger(__name__).warning('Insufficient data, padding with zeros')
            x = np.zeros(2 ** 19, dtype=np.complex64)
            x[:data.size] = data
            data = x

        if data.size > 2 ** 19:
            getLogger(__name__).warning('Data will be truncated to 2^19 samples')
            data = data[:2 ** 19]

        # Process Length
        if length is None:
            length = data.size // 16
        length = int(length)

        if length > 2 ** 15:  # NB 15 as length is in sets of 16
            getLogger(__name__).warning('Clipping replay length to 2^15 sets of 16')
            length = 2**15
        if length % 2:
            raise ValueError('Replay length must be a multiple of 2')

        # Process tlast_every
        tlast_every = int(tlast_every)

        if tlast and not (1 <= tlast_every <= length):
            raise ValueError('tlast_every must be in [1-length] when tlast is set')
        if not tlast:
            tlast_every = 256  # just assign some value to ensure we didn't get handed garbage

        self._buffer = allocate(2 ** 20, dtype=np.uint16)
        iload = [fpgen(x).__index__() for x in data.real] if fpgen is not None else data.real
        qload = [fpgen(x).__index__() for x in data.imag] if fpgen is not None else data.imag
        for i in range(16):
            self._buffer[i:data.size * 2:32] = iload[i::16]
            self._buffer[i+16:data.size * 2:32] = qload[i::16]

        self.register_map.a_1 = self._buffer.device_address
        self.register_map.length_r = length-1  # length counter
        self.register_map.tlast = bool(tlast)
        self.register_map.replay_length = tlast_every-1
        self.register_map.run = True
        if start:
            self.register_map.CTRL.AP_START = 1

    def stop(self):
        # Note the dac still outputs stuff (harmonics of last bunnfer)
        # Need to load URAM with zeros if you want it to be quiet (see quiet)
        self.register_map.run = False

    def quiet(self):
        self.replay(np.zeros(16, dtype=np.complex64), tlast=False, length=16)

    def start(self):
        #TODO probably need to check for the case where not idle and run = False (axis stall to completion)
        if self._buffer is None:
            raise RuntimeError('Must call replay first to configure the core')
        self.register_map.CTRL.AP_START = 1