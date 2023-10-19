import json
import logging
from logging import getLogger
import threading
import serial
from typing import Tuple, List
import numpy as np

MAX_OUT_ATTEN = 31.75  # dB
MAX_IN_ATTEN = 31.75  # dB
MAX_ATTN_RESOLUTION = 0.25  # dB

def escape_nline_creturn(string):
    """Escape \n and \r in a string"""
    return string.replace('\n', '\\n').replace('\r', '\\r')


class SerialDevice:
    def __init__(self, port, baudrate=115200, timeout=0.1, name=None, terminator='\n', response_terminator=''):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.name = name if name else self.port
        self.terminator = terminator
        self._response_terminator = response_terminator
        self._rlock = threading.RLock()

    def _preconnect(self):
        """
        Override to perform an action immediately prior to connection.
        Function should raise IOError if the serial device should not be opened.
        """
        pass

    def _postconnect(self):
        """
        Override to perform an action immediately after connection. Default is to sleep for twice the timeout
        Function should raise IOError if there are issues with the connection.
        Function will not be called if a connection can not be established or already exists.
        """
        pass

    def _predisconnect(self):
        """
        Override to perform an action immediately prior to disconnection.
        Function should raise IOError if the serial device should not be opened.
        """
        pass

    def connect(self, reconnect=False, raise_errors=True):
        """
        Connect to a serial port. If reconnect is True, closes the port first and then tries to reopen it. First asks
        the port if it is already open. If so, returns nothing and allows the calling function to continue on. If port
        is not already open, first attempts to create a serial.Serial object and establish the connection.
        Raises an IOError if the serial connection is unable to be established.
        """
        if reconnect:
            self.disconnect()

        try:
            if self.ser.isOpen():
                return
        except Exception:
            pass

        getLogger(__name__).debug(f"Connecting to {self.port} at {self.baudrate}")
        try:
            self._preconnect()
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
            self._postconnect()
            getLogger(__name__).debug(f"port {self.port} connection established")
            return True
        except (serial.SerialException, IOError) as e:
            getLogger(__name__).error(f"Connecting to port {self.port} failed: {str(e)}, will attempt disconnect.")
            if self.ser is not None:
                self.disconnect()
            if raise_errors:
                raise e
            return False

    def disconnect(self):
        """
        First closes the existing serial connection and then sets the ser attribute to None. If an exception occurs in
        closing the port, log the error but do not raise.
        """
        try:
            self._predisconnect()
            self.ser.close()
            self.ser = None
        except Exception as e:
            getLogger(__name__).info(f"Exception during disconnect: {e}")

    def format_msg(self, msg: str):
        """Subclass may implement to apply hardware specific formatting"""
        if msg and msg[-1] != self.terminator:
            msg = msg + self.terminator
        return msg.encode('utf-8')

    def send(self, msg: str, connect=True):
        """
        Send a message to a serial port. If connect is True, try to connect to the serial port before sending the
        message. Formats message according to the class's format_msg function before attempting to write to serial port.
        If IOError or SerialException occurs, first disconnect from the serial port, then log and raise the error.
        """
        with self._rlock:
            if connect:
                self.connect()
            try:
                msg = self.format_msg(msg)
                getLogger(__name__).getChild('io').debug(f"Sending '{msg}'")
                self.ser.write(msg)
                resp, success = self.receive()
                if not success:
                    raise IOError(resp)
                return resp
            except (serial.SerialException, IOError) as e:
                self.disconnect()
                getLogger(__name__).error(f"...failed: {e}")
                raise e

    def receive(self):
        """
        Receive data until a timeout or a :/n or ?/n is received.
        received, decode it and strip it of the confirmation/rejection. In the case of a serialException,
        disconnects from the serial port and raises an IOError.

        returns a tuple any respponse (minus the confiramtion/rejection) and a boolean indicating
        confirmation/rejection
        """
        with self._rlock:
            try:
                lines = []
                more = True
                while more:
                    lines.append(self.ser.readline().decode("utf-8"))
                    if lines[-1].endswith('?\r\n') or lines[-1].endswith(':\r\n'):
                        more = False
                response = ''.join(lines)
                getLogger(__name__).getChild('io').debug(f"Read {escape_nline_creturn(response)} from {self.name}")
                return response.strip()[:-1], response.endswith(':\r\n')
            except (IOError, serial.SerialException) as e:
                self.disconnect()
                getLogger(__name__).debug(f"Send failed {e}")
                raise IOError(e)


class IFBoard(SerialDevice):
    def __init__(self, port='/dev/ifboard', timeout=0.1, connect=True):
        super().__init__(port, 115200, timeout, response_terminator='\r\n')
        self.firmware = None
        self._monitor_thread = None
        self._initialized = False
        self.initialized_at_last_connect = False
        self.rebooted = None
        if connect:
            self.connect(raise_errors=False)

    def set_lo(self, freq_mhz, fractional: bool = True, full_calibration=True, g2_mode=False):
        try:
            freq_mhz = float(freq_mhz)
            assert 9600 > float(freq_mhz) > 0
        except (TypeError, AssertionError):
            raise ValueError('Frequency must be a float in (0, 9600)')

        self.send('G2' + ('T' if g2_mode else 'F'))
        self.send('FM' + ('T' if fractional else 'F'))
        self.send('CA' + ('T' if full_calibration else 'F'))
        resp = self.send(f'LO{freq_mhz}')
        if 'ERROR' in resp:
            getLogger(__name__).error(resp)
            raise ValueError(resp)

    def set_attens(self, output_attens: (float, Tuple[float], List[float], None) = None,
                   input_attens: (float, Tuple[float], List[float], None) = None):
        """
        Set attenuator values. Values are set in the order of the signal path. None implies a value is not changed.
        A single number sets both attenuation values.

        If individual attenuations are specified they are passed directly to the board without alteration.
        It is intended for low-level debugging use. Generally if setting individual attens,
        the majority of the attenuation should be in the first attenuator for the DAC and the second for the ADC.

        Allowed values are documented in the ifboard arduino codebase or the attenuator chips datasheet.

        Raises IOError on com failure, ValueError on invalid value or setting failure.
        """
        attens = self.attens
        current = [attens[v] for v in ('dac1', 'dac2', 'adc1', 'adc2')]

        if not isinstance(input_attens, (tuple, list)):
            x = min(input_attens, 2 * MAX_IN_ATTEN)

            if x!=input_attens:
                getLogger(__name__).warning(f'Max output attenuation exceeded. Clipping each value to {MAX_IN_ATTEN} dB.')

            input_attens = [max(x - MAX_IN_ATTEN, 0), min(x, MAX_IN_ATTEN)]

        if len(input_attens) != 2:
            raise ValueError('Incorrect number of input attenuations')

        if not isinstance(output_attens, (tuple, list)):
            x = min(output_attens, 2 * MAX_OUT_ATTEN)

            if x != output_attens:
                getLogger(__name__).warning(f'Max input attenuation exceeded. Clipping each value to {MAX_OUT_ATTEN} dB.')

            output_attens = [min(x, MAX_OUT_ATTEN), max(x - MAX_OUT_ATTEN, 0)]

        if len(output_attens) != 2:
            raise ValueError('Incorrect number of output attenuations')

        if (((np.asarray(input_attens) % MAX_ATTN_RESOLUTION).any() != 0) or ((np.asarray(output_attens) % MAX_ATTN_RESOLUTION).any() != 0)).any():
            getLogger(__name__).warning(f'Exact attenuation not achievable. Clipping value to nearest {MAX_ATTN_RESOLUTION} dB. Check status() for exact value.')

        new = output_attens + input_attens
        try:
            attens = [f'{float(n if n is not None else c):.2f}' for n, c in zip(new, current)]
        except TypeError:
            raise ValueError('Attenuations must be float or None')

        self.send('AT' + ','.join(attens))

    @property
    def powered(self):
        powered = self.send('IO?')
        if powered == '1':
            return True
        elif powered == '0':
            return False
        else:
            raise IOError("Board did not adhere to protocol")

    @property
    def attens(self):
        """
        Returns a dict of attenuator values in dB: adc1, adc2, dac1, dac2

        These values are either in effect or will take effect when powered
        """
        atten = self.send('AT?')
        try:
            attens = json.loads(atten)
        except json.JSONDecodeError:
            raise IOError("IF board did not adhere to protocol")
        return attens

    def _postconnect(self):
        data = self.ser.readall().decode('utf-8')
        if 'Enter PC for cmd list' in data:
            self.rebooted = True
            self.ser.write('x\n'.encode('utf-8'))  # a throwaway string just to stop any boot message
        self.ser.readall()

    @property
    def lo_freq(self):
        """Get the LO frequency in use while powered"""
        return float(self.send('LO?'))

    def save_settings(self):
        """Save current settings as power-on defaults"""
        self.send('SV')

    def power_off(self, save_settings=True):
        """Turn off IF board power, optionally saving the active settings"""
        self.send('IO0' + ('S' if save_settings else ''))

    def power_on(self):
        """Power up and load saved settings"""
        self.send('IO1')

    def raw(self, cmd):
        """Send a raw serial command to the IF board and return the response"""
        return self.send(cmd)

    def status(self):
        response = self.send('TS')
        getLogger(__name__).debug(response)
        lines = response.split('\n')
        lines = [l for l in lines if not l.startswith('#')]
        return IFStatus(''.join(lines))

    def stop_lo(self):
        """
        Stop the LO from oscillating
        Returns:

        """
        getLogger(__name__).warning('Requested to stop LO, but function is presently NOOP')
        pass

    def configure(self, lo=None, dac_attn=None, adc_attn=None):
        self.power_on()
        self.set_lo(lo)
        self.set_attens(dac_attn, adc_attn)

class IFStatus:
    def __init__(self, jsonstr):
        self._data = d = json.loads(jsonstr)
        self.general = g = d['global']
        self.fresh_boot = not g['coms']
        self.boot = g['boot']
        self.power = g['power']
        self.trf_general = t = d['trf'][0]
        self.trf_regs = d['trf'][1:]
        self.dac_attens = (d['attens']['dac1'], d['attens']['dac2'])
        self.adc_attens = (d['attens']['adc1'], d['attens']['adc2'])
        partialcal = not (g['gen2'] or g['g3fcal'])
        s = 'LO gen{} {} mode, {} calibration. PLL {}locked.\n\tReq: {} MHz Attained: {} MHz Err: {} MHz'
        self.lo_mode = s.format('32'[g['gen2']], ('integer', 'fractional')[g['fract']], ('full', 'partial')[partialcal],
                                ('un', '')[t['pll_locked']], g['lo'], t['f_LO'], (t['f_LO'] or np.nan) - g['lo'])

    def __str__(self):
        stat = 'IFStatus: {}, boot {}{}. {}\n\tDAC attens: {}\n\tADC Attens: {}'
        return stat.format('Powered' if self.power else 'Unpowered', self.boot,
                           ', freshly booted' if self.fresh_boot else '',
                           self.lo_mode, self.dac_attens, self.adc_attens)

# import logging
# logging.basicConfig()
# logging.getLogger('mkidgen3').setLevel('DEBUG')
# import mkidgen3.ifboard
# d='/dev/cu.usbmodem14401'
# b=mkidgen3.ifboard.IFBoard(d)
