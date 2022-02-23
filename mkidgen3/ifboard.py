import json
from logging import getLogger
import threading
import serial
from typing import Tuple, List


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
            getLogger(__name__).info(f"port {self.port} connection established")
            return True
        except (serial.SerialException, IOError) as e:
            getLogger(__name__).error(f"Connecting to port {self.port} failed: {e}, will attempt disconnect.")
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

    def format_msg(self, msg:str):
        """Subclass may implement to apply hardware specific formatting"""
        if msg and msg[-1] != self.terminator:
            msg = msg+self.terminator
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
                getLogger(__name__).debug(f"Sending '{msg}'")
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
        Receive data until a timeout or a :/n or ?/n is recieved.
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
                getLogger(__name__).debug(f"Read {escape_nline_creturn(response)} from {self.name}")
                return response.strip()[:-1], response.endswith(':\r\n')
            except (IOError, serial.SerialException) as e:
                self.disconnect()
                getLogger(__name__).debug(f"Send failed {e}")
                raise IOError(e)

class IFBoard(SerialDevice):
    def __init__(self, port, timeout=0.1, connect=True):
        """The initialize callback is called after _simspecificconnect iff _initialized is false. The callback
        will be passed this object and should raise IOError if the device can not be initialized. If it completes
        without exception (or is not specified) the device will then be considered initialized
        The .initialized_at_last_connect attribute may be checked to see if initilization ran.
        """

        super().__init__(port, 115200, timeout, response_terminator='\r\n')

        self.firmware = None
        self._monitor_thread = None
        self._initialized = False
        self.initialized_at_last_connect = False
        if connect:
            self.connect(raise_errors=False)

    def set_lo(self, freq_mhz, fractional:bool=True, full_calibration=True, g2_mode=False):
        try:
            freq_mhz=float(freq_mhz)
            assert 9600>float(freq_mhz)>0
        except (TypeError, AssertionError):
            raise ValueError('Frequency must be a float in (0, 9600)')

        if g2_mode:
            self.send('G2T')
        if fractional:
            self.send('FMT')
        if full_calibration:
            self.send('CAT')
        resp = self.query(f'LO{freq_mhz}')
        if 'ERROR' in resp:
            getLogger(__name__).error(resp)
            raise ValueError(resp)

    def set_attens(self, output_attens=None, input_attens:(float, Tuple[float], List[float], None)=None):
        """Set attenuator values. Values are set in the order of the signal path. None implies a value is not set. A
        a single number sets both attenuation values.

        Raises IOError on com failure, ValueError on invalid value or setting failure.
        """
        output_attens = output_attens if isinstance(output_attens, (tuple, list)) else [output_attens]*2
        if len(output_attens) !=2:
            raise ValueError('Incorrect number of output attenuations')

        input_attens = input_attens if isinstance(input_attens, (tuple, list)) else [input_attens]*2
        if len(input_attens) !=2:
            raise ValueError('Incorrect number of input attenuations')

        try:
            attens = [float(a) if a is not None else 'x' for a in output_attens+input_attens]
        except TypeError:
            raise ValueError('Attenuations must be float or None')

        if not self.send('AT'+','.join(attens)):
            raise ValueError('Failed to set attenuations')

    @property
    def powered(self):
        powered = self.query('IO?')
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
            self.ser.write('x\n'.encode('utf-8'))  #a throwaway string just to stop any boot message
        self.ser.readall()

    @property
    def lo_freq(self):
        return float(self.send('LO?'))

    def save_settings(self):
        self.send('SV')

    def power_off(self, save_settings=True):
        self.send('IO0'+('S' if save_settings else ''))

    def power_on(self):
        self.send('IO1')

    def raw(self, cmd):
        return self.send(cmd)

    def status(self):
        response = self.send('TS')
        getLogger(__name__).debug(response)
        lines = response.split('\n')
        lines = [l for l in lines if not l.startswith('#')]
        return json.loads(''.join(lines))

        # powered = json.loads(lines[1].partition(':')[2].strip())
        # attens = json.loads(lines[2])
        # lo_data = lines[5:-2]
        # version = lines[-2]
        # return {'device': 'online',
        #         'device_firmware': response_fwv or 'unknown',
        #         'target_frequency': self.target,
        #         'programmed_frequency': response_freq or 'unknown',
        #         'locked': response_locked or 'unknown',
        #         'output_atten': response_atten_out or 'unknown',
        #         'input_atten': response_atten_in or 'unknown',
        #         'f_ref': response_fref or 'unknown'}



# import logging
# logging.basicConfig()
# logging.getLogger('mkidgen3').setLevel('DEBUG')
# import mkidgen3.ifboard
# d='/dev/cu.usbmodem14401'
# b=mkidgen3.ifboard.IFBoard(d)