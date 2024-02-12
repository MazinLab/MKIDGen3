import json
import threading
import math
import serial
import time
import numpy as np

from logging import getLogger
from typing import Tuple, List
from enum import IntFlag, auto
from secrets import token_bytes
from dataclasses import dataclass
from fractions import Fraction

from mkidgen3.registers import MetaRegister, field, field_bool, field_enum

MAX_OUT_ATTEN = 31.75  # dB
MAX_IN_ATTEN = 31.75  # dB
IF_ATTN_STEP = 0.25  # dB IF attenuator step size (minimum dB attenuation step, limited by a single attenuator)



def _escape_nline_creturn(x):
    """Escape \n and \r in a string"""
    return x.replace('\n', '\\n').replace('\r', '\\r')


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
        Function should raise IOError in the event of any error.
        """
        pass

    def connected(self):
        return True if self.ser is not None and self.ser.isOpen() else False

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

        log=getLogger(__name__).getChild('io')
        log.debug(f"Connecting to {self.port} at {self.baudrate}")
        try:
            self._preconnect()
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
            self._postconnect()
            log.debug(f"port {self.port} connection established")
            return True
        except (serial.SerialException, IOError) as e:
            log.error(f"Connecting to port {self.port} failed: {str(e)}, will attempt disconnect.")
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
        except Exception as e:
            getLogger(__name__).getChild('io').info(f"Exception during disconnect: {e}")
        self.ser = None

    def format_msg(self, msg: str):
        """Subclass may implement to apply hardware specific formatting"""
        if msg and msg[-1] != self.terminator:
            msg = msg + self.terminator
        return msg.encode('utf-8')

    def send(self, msg: str, connect=True):
        """
        Send a message to  the serial device and get a response, optionally connecting as needed.
        Formats message according to `self.format_msg()` function before sending.

        Serial/IO errors will be raised. An IOError will be raised if the device does not indicate success.

        In the event of an IOError, disconnect from the port before logging and raising the error.

        Args:
            msg: the message to send
            connect: Try to connect before sending, if not set an attempt to send will raise an IOError

        Returns: the response from the board, possibly ''

        Raises: IOError in the event of a failure.

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
                getLogger(__name__).getChild('io').debug(f"Read {_escape_nline_creturn(response)} from {self.name}")
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
        self.trf_control = TRF3765((lambda a: self._trf_get_reg(a), lambda a, v: self._trf_set_reg(a, v)), 10)
        if connect:
            self.connect(raise_errors=False)

    def __str__(self):
        con= 'Connected' if self.connected() else 'Disconnected'
        return f'{self.port}@{self.baudrate} ({con})'

    def settle(self, timeout=None):
        """
        Wait for the IF board to be fully settled
        Returns: None
        """
        # Per Aled/Jenny 150us is the maximum settling time
        time.sleep(.000150)

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

        if (((np.asarray(input_attens) % IF_ATTN_STEP).any() != 0) or ((np.asarray(output_attens) % IF_ATTN_STEP).any() != 0)).any():
            getLogger(__name__).warning(f'Exact attenuation not achievable. Clipping value to nearest {IF_ATTN_STEP} dB. Check status() for exact value.')

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
        getLogger(__name__).info(f'Configuring {self} to lo={lo} DAC Atten:{dac_attn} ADC Atten: {adc_attn}')
        self.power_on()
        self.set_lo(lo)
        self.set_attens(dac_attn, adc_attn)

    def _trf_get_reg(self, address):
        resp = self.send("RR{:d}".format(address))
        return int(resp.strip(), 16)

    def _trf_set_reg(self, address, value):
        self.send("WR{:d}{:s}".format(address, hex(value)[2:]))

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


# It's ugly but its the best we can reasonably do atm
class _TRFReg(MetaRegister):
    def __init__(self, address, rwhandle):
        self.address = address
        self.rwhandle = rwhandle

    def __len__(self):
        return 32

    def __get__(self, obj, objtype=None):
        if obj is None or ((objtype is not None) and issubclass(objtype, MetaRegister)):
            return self
        return self.rwhandle[0](self.address)

    def __set__(self, obj, val: int):
        self.rwhandle[1](self.address, val)

class TRFPowerDown(IntFlag):
    PLL = auto()
    CP = auto()
    VCO = auto()
    VCOMUX = auto()
    DIV124 = auto()
    PRESC = auto()
    LO_DIV = auto()
    BUFF1 = auto()
    BUFF2 = auto()
    BUFF3 = auto()
    BUFF4 = auto()

@dataclass
class TRFVCOCalibration:
    vco_sel: int
    vco_trim: int

@dataclass
class TRFFractionalCalibration:
    mod_ord: int
    isource_sink: bool
    isource_trim: int

DEFAULT_FRACCAL = TRFFractionalCalibration(0b10, False, 0b100)

@dataclass
class TRFDividerConfig:
    lo_div_sel: int
    pll_div_sel: int
    rdiv: int
    nint: int
    nfrac: int
    prsc_sel: int
    f_ref: int | Fraction
    
    @property
    def frequency(self) -> Fraction:
        base = Fraction(self.f_ref * 1 * (1 << self.pll_div_sel), self.rdiv)
        integer = base*self.nint
        fractional = base*Fraction(self.nfrac, 1<<25)
        return integer + fractional

    @property
    def step(self) -> Fraction:
        return TRFDividerConfig(
            self.lo_div_sel,
            self.pll_div_sel,
            self.rdiv,
            self.nint,
            self.nfrac + 1,
            self.prsc_sel,
            self.f_ref
        ).frequency - self.frequency

    @classmethod
    def from_target(cls, freq_unscaled: np.float64 | Fraction, f_ref: int | Fraction = 10) -> "TRFDividerConfig":
        if freq_unscaled is np.float64:
            freq = Fraction(freq_unscaled)
        else:
            freq = freq_unscaled

        if freq > 4800 or freq < 300:
            raise ValueError("TRF3765 (undoubled) output frequency not in [300, 4800] MHz")
        if freq >= 2400:
            lo_div_sel = 0
        elif freq >= 1200:
            lo_div_sel = 1
        elif freq >= 600:
            lo_div_sel = 2
        else:
            lo_div_sel = 3
        f_vco_target = freq * (1 << lo_div_sel)

        pll_div_sel = 0
        if f_vco_target > 3000:
            pll_div_sel = 1
        if f_vco_target / 2 > 3000:
            pll_div_sel = 2
        if f_vco_target / 4 > 3000:
            raise ValueError(
                "Unable to compute pll_div_sel (see TRF3765 datasheet section 7.4.4.1 b ) with lo_div_sel = {:d}, freq_unscaled = {:f}, f_ref = {:d}".format(
                    lo_div_sel,
                    float(freq_unscaled),
                    f_ref
                )
            )

        rdiv = 1
        if f_ref >=  48:
            rdiv = math.ceil(48 / f_ref)
            if rdiv > (1 << 13) - 1:
                raise ValueError("Reference out of range, cannot make f_pfd <= 48MHz using rdiv")

        while pll_div_sel <= 2:
            pll_div = 1 << pll_div_sel
            nint = math.floor(f_vco_target * rdiv / (f_ref * pll_div))
            nfrac = round((f_vco_target * rdiv / (f_ref * pll_div) - nint) * (1 << 25))

            if nint >= 75:
                prsc_sel = 1
                p = 8
            elif nint >= 23:
                prsc_sel = 0
                p = 4
            else:
                pll_div_sel += 1
                continue

            f_n = f_vco_target / (p * pll_div)
            if f_n <= 375:
                return TRFDividerConfig(
                    lo_div_sel,
                    pll_div_sel,
                    rdiv,
                    nint,
                    nfrac,
                    prsc_sel,
                    f_ref
                )
        raise ValueError("Couldn't make digital divider frequency (f_N) <= 375 MHz, see TRF3765 Datasheet section 7.4.4.1 c.")

@dataclass
class TRFCalibrationCertificate:
    token: int
    divider_config: TRFDividerConfig
    vco_calibration: TRFVCOCalibration

    @property
    def frequency(self) -> Fraction:
        return self.divider_config.frequency

class TRF3765:
    def __init__(self, rwhandle, f_ref):
        self.r0 = _TRFReg(0, rwhandle)
        self.r1 = _TRFReg(1, rwhandle)
        self.r2 = _TRFReg(2, rwhandle)
        self.r3 = _TRFReg(3, rwhandle)
        self.r4 = _TRFReg(4, rwhandle)
        self.r5 = _TRFReg(5, rwhandle)
        self.r6 = _TRFReg(6, rwhandle)

        self.__token = int.from_bytes(token_bytes(16), byteorder='big')
        self.f_ref = f_ref

    def get_certificate(self) -> TRFCalibrationCertificate:
        return TRFCalibrationCertificate(
            self.__token,
            TRFDividerConfig(
                self.lo_div_sel,
                self.pll_div_sel,
                self.rdiv,
                self.nint,
                self.nfrac,
                self.prsc_sel,
                self.f_ref
            ),
            TRFVCOCalibration(
                self.vco_sel_readback,
                self.vco_trim_readback,
            )
        )

    def program(self, divider_config: TRFDividerConfig):
        assert divider_config.f_ref == self.f_ref
        old_power = self.powerdown_parts
        self.powerdown_parts = old_power | TRFPowerDown.BUFF1 | TRFPowerDown.BUFF2 | TRFPowerDown.BUFF3 | TRFPowerDown.BUFF4
        self.lo_div_sel = divider_config.lo_div_sel
        self.pll_div_sel = divider_config.pll_div_sel
        self.rdiv = divider_config.rdiv
        self.nint = divider_config.nint
        self.nfrac = divider_config.nfrac
        self.prsc_sel = divider_config.prsc_sel
        self.en_cal = True
        while self.en_cal:
            pass
        self.powerdown_parts = old_power

    def _set_fractional_cal(self, frac_calibration: TRFFractionalCalibration):
        self.isource = True
        self.dith = True
        self.mod_ord = frac_calibration.mod_ord
        self.dith_sel = False
        self.del_sd_clk = 0b10
        self.en_frac = True
        self.ld_isource = False
        self.isource_sink = frac_calibration.isource_sink
        self.isource_trim = frac_calibration.isource_trim
        self.icpdouble = False

    @field(slice(5, 18))
    def rdiv(self):
        return self.r1

    @field_bool(19)
    def ref_inv(self):
        return self.r1

    @field_bool(20)
    def neg_vco(self):
        return self.r1

    @field(slice(21, 26))
    def icp(self):
        return self.r1

    @field_bool(26)
    def icpdouble(self):
        return self.r1

    @field(slice(27, 31))
    def cal_clk_sel(self):
        return self.r1

    @field(slice(5, 21))
    def nint(self):
        return self.r2

    @field(slice(21, 23))
    def pll_div_sel(self):
        return self.r2

    @field(slice(23, 24))
    def prsc_sel(self):
        return self.r2

    @field(slice(26, 28))
    def vco_sel(self):
        return self.r2

    @field_bool(28)
    def vco_sel_mode(self):
        return self.r2

    @field(slice(29, 31))
    def cal_acc(self):
        return self.r2

    @field_bool(31)
    def en_cal(self):
        return self.r2

    @field(slice(5, 30))
    def nfrac(self):
        return self.r3

    @field_enum(slice(5, 16), TRFPowerDown)
    def powerdown_parts(self):
        return self.r4

    @field_bool(18)
    def isource(self):
        return self.r4

    @field_bool(25)
    def dith(self):
        return self.r4

    @field(slice(26, 28))
    def mod_ord(self):
        return self.r4

    @field_bool(28)
    def dith_sel(self):
        return self.r4

    @field(slice(29, 31))
    def del_sd_clk(self):
        return self.r4

    @field_bool(31)
    def en_frac(self):
        return self.r4

    @field_bool(31)
    def ld_isource(self):
        return self.r5

    @field(slice(7, 13))
    def vco_trim(self):
        return self.r6

    @field_bool(19)
    def isource_sink(self):
        return self.r6

    @field(slice(20, 23))
    def isource_trim(self):
        return self.r6

    @field(slice(23, 25))
    def lo_div_sel(self):
        return self.r6

    @field(slice(15, 21))
    def vco_trim_readback(self):
        return self.r0

    @field(slice(22, 24))
    def vco_sel_readback(self):
        return self.r0

# import logging
# logging.basicConfig()
# logging.getLogger('mkidgen3').setLevel('DEBUG')
# import mkidgen3.ifboard
# d='/dev/cu.usbmodem14401'
# b=mkidgen3.ifboard.IFBoard(d)
