import logging
from logging import getLogger
import numpy as np
import time
import pynq
from mkidgen3.fixedpoint import FP16_15


class DACTableURAM(pynq.DefaultHierarchy):
    @staticmethod
    def checkhierarchy(description):
        # found = check_description_for(description, ('xilinx.com:axis_switch', 'mazinlab:mkidgen3:filter_iq',
        #                                             'xilinx.com:module_ref:axis2mm'))
        return description['fullpath'] == 'dactable'  # and bool(len(found['xilinx.com:module_ref:axis2mm']))

    def __init__(self, description):
        super().__init__(description)
        self._buffer = None

    def replay_ramp(self):
        ramp = np.arange(2 ** 19, dtype=np.complex64)
        ramp.real %= (2 ** 16)
        ramp.imag = ramp.real
        self.replay(ramp, fpgen=None)

    def replay(self, data, fpgen=lambda x: [FP16_15(v).__index__() for v in x]):
        """
        data - an array of complex numbers, nominally 2^19 samples
        fpgen - set to a function to convert floating point numbers to integers. must work on arrays of data
        if None data will be truncated.
        """
        if fpgen == 'simple':
            data = (data * 8192).round().clip(-8192, 8191) * 4
            fpgen = None
        # Data has right shape
        if data.size < 2 ** 19:
            getLogger(__name__).warning('Insufficient data, padding with zeros')
            x = np.zeros(2 ** 19, dtype=np.complex64)
            x[:data.size] = data
            data = x

        if data.size > 2 ** 19:
            getLogger(__name__).warning('Data will be truncated to 2^19 samples')
            data = data[:2 ** 19]

        self._buffer = np.zeros(2 ** 20, dtype=np.uint16)
        iload = fpgen(data.real) if fpgen is not None else data.real.astype(np.int16)
        qload = fpgen(data.imag) if fpgen is not None else data.imag.astype(np.int16)
        for i in range(8):
            self._buffer[i:data.size * 2:16] = iload[i::8]
            self._buffer[i + 8:data.size * 2:16] = qload[i::8]
        self.axi_bram_ctrl_0.mmio.array[:] = np.frombuffer(self._buffer, dtype=np.uint32)
        time.sleep(1)

    def stop(self):
        self._buffer = np.zeros(2 ** 20, dtype=np.uint16)
        self.axi_bram_ctrl_0.mmio.array[:] = 0

    def quiet(self):
        """Replay 0s"""
        self.stop()

    def status(self)->dict:
        """
        Generate a dictionary with whether the core is running and a copy of the current buffer
        Returns: dict with keys running and buffer
        """
        return {'running': True, 'buffer': self._buffer.copy() if self._buffer is not None else None}

    def configure(self, waveform=None):
        """
        Args:
            waveform_values: complex values quantized to integer real and imaginary parts
            max val, min val: [2**15-1, -2**15] corresponding to [1.0, -1.0] & [max dac V, min dac V]
        Returns: None
        """
        try:
            waveform_values = waveform.output_waveform
        except AttributeError:
            getLogger(__name__).info('Interpreting waveform as array')
            waveform_values = waveform

        if waveform_values is None:
            raise ValueError('waveform_values is None')

        self.replay(waveform_values, fpgen=None)


class DACTableAXIM(pynq.DefaultIP):
    bindto = ['mazinlab:mkidgen3:dac_table_axim:1.33']

    def __init__(self, description):
        super().__init__(description=description)
        self._buffer = None

    def replay_ramp(self, tlast_every=256):
        ramp = np.arange(2 ** 19, dtype=np.complex64)
        ramp.real %= (2 ** 16)
        ramp.imag = ramp.real
        self.stop()
        self.replay(ramp, tlast_every=tlast_every, fpgen=None)

    def replay(self, data, tlast=True, tlast_every=256, replay_len=None, start=True,
               fpgen=lambda x: [FP16_15(v).__index__() for v in x], stop_if_needed=False):
        """
        data - an array of complex numbers, nominally 2^19 samples
        tlast - Set to true to emit a tlast pulse every tlast_every transactions
        tlast_every - assert tlast every tlast_every*16 th sample.
        replay_len - Number of groups of 16 (DAC takes 16/clock) in the data to use.
            e.g. if data[N].reshape(N//16,16). Note that the user channel will count to length-1
        start - start the replay immediately
        fpgen - set to a function to convert floating point numbers to integers. must work on arrays of data
        if None data will be truncated.
        """
        if self.register_map.run.run and not stop_if_needed:
            raise RuntimeError('Replay in progress. Call .stop() or with stop_if_needed=True')
        if fpgen == 'simple':
            data = (data * 8192).round().clip(-8192, 8191) * 4
            fpgen = None
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
        if replay_len is None:
            replay_len = data.size // 16
        replay_len = int(replay_len)

        if replay_len > 2 ** 15:  # NB 15 as length is in sets of 16
            getLogger(__name__).warning('Clipping replay length to 2^15 sets of 16')
            replay_len = 2 ** 15
        if replay_len % 2:
            raise ValueError('Replay length must be a multiple of 2')

        # Process tlast_every
        tlast_every = int(tlast_every)

        if tlast and not (1 <= tlast_every <= replay_len):
            raise ValueError('tlast_every must be in [1-length] when tlast is set')
        if not tlast:
            tlast_every = 256  # just assign some value to ensure we didn't get handed garbage

        if self._buffer is not None:
            self._buffer.freebuffer()
        self._buffer = pynq.allocate(2 ** 20, dtype=np.uint16)

        iload = fpgen(data.real) if fpgen is not None else data.real.astype(np.int16)
        qload = fpgen(data.imag) if fpgen is not None else data.imag.astype(np.int16)
        for i in range(16):
            self._buffer[i:data.size * 2:32] = iload[i::16]
            self._buffer[i + 16:data.size * 2:32] = qload[i::16]

        if self.register_map.run.run:
            self.stop()
            while not self.register_map.CTRL.AP_IDLE:
                time.sleep(.0001)
        self.register_map.a_1 = self._buffer.device_address
        self.register_map.length_r = replay_len - 1  # length counter
        self.register_map.tlast = bool(tlast)
        self.register_map.replay_length = tlast_every - 1
        self.register_map.run = True
        if start:
            self.register_map.CTRL.AP_START = 1

    def stop(self):
        # Note the dac still outputs stuff (harmonics of last buffer)
        # Need to load URAM with zeros if you want it to be quiet (see quiet)
        self.register_map.run = False

    def quiet(self):
        """Replay 0s"""
        self.replay(np.zeros(16, dtype=np.complex64), tlast=False, replay_len=16, stop_if_needed=True)
        self.stop()

    def status(self)->dict:
        """
        Generate a dictionary with whether the core is running and a copy of the current buffer
        Returns: dict with keys running and buffer
        """
        return {'running': self.register_map.run.run and self.register_map.CTRL.AP_START,
                'buffer': self._buffer.copy() if self._buffer is not None else None}

    def configure(self, waveform=None):
        """
        Args:
            waveform_values: complex values quantized to integer real and imaginary parts
            max val, min val: [2**15-1, -2**15] corresponding to [1.0, -1.0] & [max dac V, min dac V]
        Returns: None
        """
        try:
            waveform_values = waveform.output_waveform
        except AttributeError:
            getLogger(__name__).info('Interpreting waveform as array')
            waveform_values = waveform

        if waveform_values is None:
            raise ValueError('waveform_values is None')

        self.replay(waveform_values, tlast=True, tlast_every=256, replay_len=None, start=True,
                    fpgen=None, stop_if_needed=True)
