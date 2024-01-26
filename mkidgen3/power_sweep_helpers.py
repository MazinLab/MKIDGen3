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
        return input_power+self.gain

    def __repr__(self):
        return f'{self.name} {self.gain} dB atten'


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
        return input_power+self.gain

    def __repr__(self):
        return f"Device ({self.gain} dB atten)"


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
        return input_power+self.gain

    def is_saturated(self, input_power: float) -> bool:
        return self.output(input_power) > self.saturation

    @property
    def max_input(self):
        return self.saturation - self.gain


class ROChain:
    def __init__(self, ro_chain: list):
        self.ro_chain = ro_chain
        self._index = 0

    @property
    def device_idx(self) -> int:
        return next(i for i, v in enumerate(self.ro_chain) if isinstance(v, Device))

    @property
    def dac_atten_idx(self) -> int:
        return next(i for i, v in enumerate(self.ro_chain) if (isinstance(v, ProgrammableAttenuator) & ('dac' in v.name)))

    @property
    def adc_atten1_idx(self) -> int:
        return next(i for i, v in enumerate(self.ro_chain) if (isinstance(v, ProgrammableAttenuator) & ('adc' in v.name)))

    @property
    def adc_atten2_idx(self) -> int:
        return len(self.ro_chain) - next(i for i, v in enumerate(self.ro_chain[::-1]) if (isinstance(v, ProgrammableAttenuator) & ('adc' in v.name)))-1

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
        return self.calculate_gain(self.dac_atten_idx+1, self.device_idx)

    def post_device_gain(self):
        return self.calculate_gain(self.device_idx, self.adc_atten1_idx)

    def post_adc_atten2_gain(self):
        return self.calculate_gain(self.adc_atten2_idx, len(self.ro_chain))