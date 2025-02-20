import json
import threading
import math
import serial
import time
import numpy as np

from logging import getLogger
from typing import Tuple, List, Optional
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

    def set_lo(self, freq, fractional: Optional[bool] = None):
        if type(freq) is TRFCalibrationCertificate:
            assert fractional is None
            self.trf_control.set_output(freq, wait_for_lock=False)
        else:
            self.trf_control.set_output(freq / 2, wait_for_lock=False, fractional = fractional)

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
    cacheacc = 0
    totalacc = 0
    readacc = 0
    readcacc = 0
    def __init__(self, address, rwhandle, cached=True, nocache=0, readcache=False):
        self.address = address
        self.rwhandle = rwhandle
        self.cached = cached
        self.nocache = 0
        self.cache = None
        self.readcache = readcache

    def __len__(self):
        return 32

    def __get__(self, obj, objtype=None):
        if obj is None or ((objtype is not None) and issubclass(objtype, MetaRegister)):
            return self
        self.readacc += 1
        if self.readcache and self.cache is not None:
            self.readcacc += 1
            return self.cache
        self.cache = self.rwhandle[0](self.address)
        return self.cache

    def __set__(self, obj, val: int):
        self.totalacc += 1
        if self.cache is not None:
            if self.cache == val and self.cached and not (val & self.nocache):
                self.cacheacc += 1
                return
        self.cache = val
        self.rwhandle[1](self.address, val)

    def invalidate(self):
        self.cache = None

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
    
    def __post_init__(self):
        self.validate(False)

    def validate(self, calibrate: bool):
        if self.prsc_sel < 0 or self.prsc_sel > 1:
            raise ValueError("PRSC_SEL must be 0 or 1")
        if self.rdiv < 1 or self.rdiv > (1 << 13) - 1:
            raise ValueError("RDIV is not in [1, 1 << 13 - 1]")
        if self.nint < 1 or self.nint > (1 << 16) - 1:
            raise ValueError("NINT is not int [1, 1 << 16 - 1]")
        if self.nfrac < 0 or self.nfrac > (1 << 25) - 1:
            raise ValueError("NFRAC is not in [0, 1 << 25 - 1]")
        if self.pll_div_sel < 0 or self.pll_div_sel > 2:
            raise ValueError("PLL_DIV_SEL must be 0 1 or 2")
        if self.lo_div_sel < 0 or self.lo_div_sel > 3:
            raise ValueError("LO_DIV_SEL must be 0 1 2 or 3")

        if self.frequency > 4800 or self.frequency < 300:
            raise ValueError("Frequency {:f} out of range [300, 4800] MHz".format(float(self.frequency)))
        if self.f_vco / self.pll_div > 3000:
            raise ValueError("PLL Input frequency {:f} > 3000 MHz".format(float(self.f_vco / self.pll_div)))
        if self.f_pfd > 48:
            raise ValueError("Phase detector frequency ({:f} MHz) > 48 MHz".format(float(self.f_pfd)))
        if self.f_ref < 0.5 or self.f_ref > 350:
            raise ValueError("Reference frequency ({:f} MHz) < 0.5MHz or < 350 MHz".format(float(self.f_ref)))
        if self.f_n > 375:
            raise ValueError("Integer divider frequency (f_n = {:f}) > 375 MHz".format(float(self.f_n)))

        if self.prsc_p == 8:
            if self.nfrac == 0 and self.nint < 72:
                raise ValueError("PRSC is 8/9 with NINT < 72 in integer mode")
            elif self.nfrac != 0 and self.nint < 75:
                raise ValueError("PRSC is 8/9 with nint < 75 in fractional mode")
        if self.prsc_p == 4:
            if self.nfrac == 0 and (self.nint >= 72 or self.nint < 20):
                raise ValueError("PRSC is 4/5 with nint not in [20, 72)")
            if self.nfrac == 1 and (self.nint >= 75 or self.nint < 23):
                raise ValueError("PRSC is 4/5 with nint not in [23, 75)")

        if calibrate:
            _ = self.cal_clk

    @property
    def frequency(self) -> Fraction:
        return self.f_vco / self.lo_div

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

    @property
    def prsc_p(self) -> int:
        if self.prsc_sel:
            return 8
        return 4

    @property
    def pll_div(self) -> int:
        return 1 << self.pll_div_sel

    @property
    def lo_div(self) -> int:
        return 1 << self.lo_div_sel

    @property
    def f_vco(self) -> Fraction:
        base = Fraction(self.f_ref * self.pll_div, self.rdiv)
        integer = base*self.nint
        fractional = base*Fraction(self.nfrac, 1<<25)
        return integer + fractional

    @property
    def f_pfd(self) -> Fraction:
        return self.f_vco / (self.nint * self.pll_div)

    @property
    def f_n(self) -> Fraction:
        return self.f_vco / (self.prsc_p * self.pll_div)

    @property
    def cal_clk_factor(self) -> Fraction:
        pfd = self.f_pfd
        ref = self.f_ref
        factor = Fraction(128, 1)
        while pfd * factor > Fraction(6, 10):
            factor = factor / 2
            if factor < Fraction(1, 128):
                raise ValueError("Unable to produce a cal_clk frequency < 600 KHz")
        while ref * factor > Fraction(1, 100) and pfd * factor > Fraction(5, 100) and factor >= Fraction(1, 128):
            if ref / (pfd * factor) < 8000:
                return factor
            factor = factor / 2
        raise ValueError("Unable to find valid cal clk")

    @property
    def cal_clk(self) -> Fraction:
        return self.cal_clk_factor * self.f_pfd

    @property
    def cal_clk_sel(self) -> int:
        n, d = self.cal_clk_factor.as_integer_ratio()
        if n == d and n == 1:
            return 0b1000
        if n == 1:
            return 0b1000 + round(math.log2(d))
        return 0b1000 - round(math.log2(n))

    @classmethod
    def from_target(cls, freq_unscaled: float | Fraction, f_ref: int | Fraction = 10) -> "TRFDividerConfig":
        if freq_unscaled is float:
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
        while f_vco_target / (1 << pll_div_sel) > 3000:
            pll_div_sel += 1
            if pll_div_sel > 2:
                raise ValueError(
                    "Unable to compute pll_div_sel (see TRF3765 datasheet section 7.4.4.1 b ) with lo_div_sel = {:d}, freq_unscaled = {:f}, f_ref = {:f}".format(
                        lo_div_sel,
                        float(freq_unscaled),
                        float(f_ref)
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
    tcal_total = 0.0

    def __init__(self, rwhandle, f_ref, outputs = TRFPowerDown.BUFF1):
        self.r0 = _TRFReg(0, rwhandle, cached=False)
        self.r1 = _TRFReg(1, rwhandle, readcache=True)
        self.r2 = _TRFReg(2, rwhandle, nocache=1<<31, readcache=True)
        self.r3 = _TRFReg(3, rwhandle, readcache=True)
        self.r4 = _TRFReg(4, rwhandle, readcache=True)
        self.r5 = _TRFReg(5, rwhandle, readcache=True)
        self.r6 = _TRFReg(6, rwhandle, readcache=True)

        self.__token = int.from_bytes(token_bytes(16), byteorder='big')
        self.f_ref = f_ref
        self.outputs = outputs

    def set_output(self, frequency: float | Fraction | TRFCalibrationCertificate, wait_for_lock: bool = True, fractional: Optional[bool] = None):
        if isinstance(frequency, TRFCalibrationCertificate):
            if frequency.token != self.__token:
                raise ValueError("Calibration certificate is not for this LO")
            self.__program_divider(frequency.divider_config)
            self.vco_sel = frequency.vco_calibration.vco_sel
            self.vco_trim = frequency.vco_calibration.vco_trim
        else:
            divider_config = TRFDividerConfig.from_target(frequency, self.f_ref)
            self.__program(divider_config, True, True, fractional)     
        if wait_for_lock:
            raise ValueError("Currently unimplemented")

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

    def __program_divider(self, divider_config: TRFDividerConfig):
        self.lo_div_sel = divider_config.lo_div_sel
        self.pll_div_sel = divider_config.pll_div_sel
        self.rdiv = divider_config.rdiv
        self.nint = divider_config.nint
        self.nfrac = divider_config.nfrac
        self.prsc_sel = divider_config.prsc_sel

    def __program(
        self,
        divider_config: TRFDividerConfig,
        calibrate: bool = True,
        powerdown: bool = True,
        fractional: Optional[bool] = None,
        fractional_cal: Optional[TRFFractionalCalibration] = None,
    ):
        assert divider_config.f_ref == self.f_ref
        if calibrate:
            self.cal_clk_sel = divider_config.cal_clk_sel
        if fractional is not None and not fractional:
            if divider_config.nfrac != 0:
                raise ValueError("Requested integer calibration with nfrac != 0")
        if fractional is None:
            fractional = divider_config.nfrac != 0
        self.powerdown_parts = TRFPowerDown.BUFF1 | TRFPowerDown.BUFF2 | TRFPowerDown.BUFF3 | TRFPowerDown.BUFF4
        self.__program_divider(divider_config)
        if fractional_cal is not None:
            self.__set_fractional_cal(fractional_cal)
        else:
            self.__set_fractional_cal(DEFAULT_FRACCAL)
        self.en_frac = fractional
        if calibrate:
            start_cal = time.time()
            self.en_cal = True
            while self.en_cal:
                self.r2.invalidate()
                pass
            self.tcal_total += time.time() - start_cal
        self.powerdown_parts = (TRFPowerDown.BUFF1 | TRFPowerDown.BUFF2 | TRFPowerDown.BUFF3 | TRFPowerDown.BUFF4) ^ self.outputs

    def __set_fractional_cal(self, frac_calibration: TRFFractionalCalibration):
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
