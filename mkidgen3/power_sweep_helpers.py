import copy
from logging import getLogger
import numpy.typing as nt
import numpy as np
from mkidgen3.system_parameters import ADC_DAC_INTERFACE_WORD_LENGTH, DAC_RESOLUTION, DAC_LUT_SIZE, N_CHANNELS, \
    SYSTEM_BANDWIDTH, IF_ATTN_STEP, ADC_MAX_V, PHASE_FRACTIONAL_BITS


class Component:
    def __init__(self, name: str | None, gain: float):
        self.name = name
        self._gain = gain

    @property
    def gain(self):
        return self._gain

    def __add__(self, other):
        return Component(None, self.gain + other.gain)


class Attenuator(Component):
    def __init__(self, name: str, gain: float):
        super().__init__(name, gain)
        assert gain <= 0, "gain value should be negative for an attenutor"

    def output(self, input_power: float) -> float:
        return input_power + self.gain

    def __repr__(self):
        return f'{self.name} {self.gain} dB attn'


class ProgrammableAttenuator(Attenuator):
    def __init__(self, name: str, gain: float):
        super().__init__(name, gain)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        assert gain <= 0, "gain value should be negative for an attenutor"
        self._gain = gain


class Device(Component):
    def __init__(self, name: str, gain: float):
        super().__init__(name, gain)

    def output(self, input_power: float) -> float:
        return input_power + self.gain

    def __repr__(self):
        return f"Device ({self.gain} dB attn)"


class Amplifier(Component):
    def __init__(self, name: str, gain: float, saturation: float):
        super().__init__(name, gain)
        assert gain >= 0, "gain value should be positive for an amplifier"
        self.saturation = saturation  # dBm

    def __repr__(self):
        return f'{self.name}, gain: {self.gain} dB, saturation: {self.saturation} dBm'

    def output(self, input_power: float) -> float:
        """

        Args:
            input_power: dBm

        Returns: output power in dBm

        """
        return input_power + self.gain

    def is_saturated(self, input_power: float) -> bool:
        return self.output(input_power) > self.saturation

    @property
    def max_input(self):
        return self.saturation - self.gain


class ROChain:
    def __init__(self, *ro_chain):
        self.ro_chain = ro_chain
        self._index = 0

    @property
    def device_idx(self) -> int:
        return next(i for i, v in enumerate(self.ro_chain) if isinstance(v, Device))

    @property
    def dac_attn_idx(self) -> int:
        return next(
            i for i, v in enumerate(self.ro_chain) if (isinstance(v, ProgrammableAttenuator) & ('dac' in v.name)))

    @property
    def adc_attn1_idx(self) -> int:
        return next(
            i for i, v in enumerate(self.ro_chain) if (isinstance(v, ProgrammableAttenuator) & ('adc' in v.name)))

    @property
    def adc_attn2_idx(self) -> int:
        return len(self.ro_chain) - next(i for i, v in enumerate(self.ro_chain[::-1]) if
                                         (isinstance(v, ProgrammableAttenuator) & ('adc' in v.name))) - 1

    def __repr__(self):
        string = ''
        for component in self.ro_chain:
            string += component.__repr__() + '\n'
        return string

    def __getitem__(self, item):
        return self.ro_chain[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.ro_chain):
            item = self.ro_chain[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def validate(self, input_power):
        for component in self.ro_chain:
            try:
                if component.is_saturated(input_power):
                    raise ValueError(f'{component.name} receiving {input_power} dBm is saturated.')
            except AttributeError:
                pass
            input_power = component.output(input_power)

    def calculate_gain(self, start, stop):
        gain = 0
        for i in range(start, stop):
            gain += self.ro_chain[i].gain
        return gain

    def pre_device_gain(self):
        return self.calculate_gain(self.dac_attn_idx + 1, self.device_idx)

    def post_device_gain(self):
        return self.calculate_gain(self.device_idx, self.adc_attn1_idx)

    def post_adc_attn2_gain(self):
        return self.calculate_gain(self.adc_attn2_idx, len(self.ro_chain))


def calculate_adc_dac_attn(dac_output_power: float, device_power: float, signal_chain: ROChain,
                            adc_input_power: float) -> tuple[float, tuple[float, float]]:
    """
    Args:
        dac_output_power: dac output power [dBm]
        device_power: target power at MKID device [dBm]
        signal_chain: list of components
        adc_input_power: target adc input power [dBm]

    Returns: tuple where the first element is the total dac attenuation and the second element is a tuple where the
    first element is the adc attenuation of the first programmable attenuator in the receive chain and the second
    element is the second programmable attenuator in the receive chain.

    """
    chain = copy.deepcopy(signal_chain)
    dac_attn = -device_power + dac_output_power + chain.pre_device_gain()
    if dac_attn < 0:
        getLogger(__name__).warning(
            f'Cannot acheive device power {device_power} dBm, setting DAC attn to 0, device power will be {dac_output_power + chain.pre_device_gain()} dBm')
        dac_attn = 0
    chain[chain.dac_attn_idx].gain = -dac_attn
    post_adc_attns = adc_input_power - chain.post_adc_attn2_gain()
    pre_adc_attns = device_power + chain.post_device_gain()
    total_adc_attenuation = pre_adc_attns - post_adc_attns

    amp_between_adc_attns = chain.ro_chain[chain.adc_attn1_idx + 1]
    assert isinstance(amp_between_adc_attns, Amplifier), "adc attenuator 1 needs to be followed by amplifier"
    if pre_adc_attns > amp_between_adc_attns.max_input:
        chain.ro_chain[chain.adc_attn1_idx].gain = pre_adc_attns - amp_between_adc_attns.max_input
        chain.ro_chain[chain.adc_attn2_idx].gain = -total_adc_attenuation + chain.ro_chain[chain.adc_attn1_idx].gain
    else:
        chain.ro_chain[chain.adc_attn1_idx].gain = 0
        chain.ro_chain[chain.adc_attn2_idx].gain = -total_adc_attenuation

    chain.validate(dac_output_power)
    adc_attn = -chain.ro_chain[chain.adc_attn1_idx].gain + -chain.ro_chain[chain.adc_attn2_idx].gain  # It may make sense to return these separately

    return -chain.ro_chain[chain.dac_attn_idx].gain, adc_attn


def compute_power_sweep_attenuations(dac_attn_start: float, adc_attn_start: float,
                                     dac_attn_stop: float | None = None, points: int | None = None,
                                     step_size=IF_ATTN_STEP) -> list[tuple[float | int, float | int]]:
    """
    Args:
        dac_attn_start: starting dac attenuation [dB]
        adc_attn_start: starting adc attenuation [dB]
        dac_attn_stop: *last dac attenuation [dB]
        points: *number of attenuation steps, overrides step_size
        step_size: *attenuation step size

    Returns: list of tuples containing DAC and ADC attns
    For Ex: [(dac_attn0, adc_attn0), (dac_attn1, adc_attn1), ...]

    *A subset of these is required. Precedence is resolved like so:
    If dac_attn_stop:
        if points:
            dac_attn_stop and points determine the list and step_size is ignored
        elif step_size:
            dac_attn_stop and step_size determine the list
    else:
        step size and points determine the list


    """
    for attn in [dac_attn_start, adc_attn_start, dac_attn_stop]:
        if attn:
            assert attn >= 0, "attenuation must be positive"

    if dac_attn_stop:
        if points:
            dac_attns, step = np.linspace(dac_attn_start, dac_attn_stop, points, retstep=True)
            adc_attns = np.linspace(adc_attn_start, adc_attn_start - step * (len(dac_attns) - 1), points)
        else:
            dac_attns = np.arange(dac_attn_start, dac_attn_stop + step_size, step_size)
            adc_attns = np.arange(adc_attn_start, adc_attn_start - step_size * len(dac_attns), -step_size)
    else:
        dac_attns = np.arange(dac_attn_start, dac_attn_start + step_size * points, step_size)
        adc_attns = np.arange(adc_attn_start, adc_attn_start - step_size * points, -step_size)

    return [(dac, adc) for dac, adc in zip(dac_attns, adc_attns)]


def single_res_sweep(if_board, lo_sweep_freqs, ol):
    iq_vals = np.zeros(lo_sweep_freqs.size, dtype=np.complex64)
    for x in range(len(lo_sweep_freqs)):
        if_board.set_lo(lo_sweep_freqs[x] * 1e-6)
        iq_vals[x] = get_iq_point(ol)
    return iq_vals


def get_iq_point(ol, n=256):
    """
    Args:
        n: int
        how many points to average
    Returns: a single averaged iq data point captured from res channel 0
    """
    x = ol.capture.capture_iq(n, [0, 1], tap_location='ddciq')
    tmp = np.zeros(x.shape[1:])
    x.mean(axis=0, out=tmp)
    del x
    return tmp[0, 0] + 1j * tmp[0, 1]

def compute_lo_steps(center: float | int, resolution: float | int, bandwidth: float | int) -> nt.NDArray[float | int]:
    """
    Compute LO steps.
    Args:
        center: center frequency of the sweep bandwidth [Hz]
        resolution: frequency resolution in for the LO sweep [Hz]
        bandwidth: bandwidth for the LO to sweep through [Hz]

    Returns: list of LO settings

    """
    n_steps = np.round(bandwidth / resolution).astype('int')
    return np.linspace(-bandwidth / 2, bandwidth / 2, n_steps) + center
