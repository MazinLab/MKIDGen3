"""Off-board tests for the BxBFFT mode 12 gain controller.

A fake GPIO models the BxBFFT detector semantics from user guide v3.2:
detector i reads the pre-shift peak at scaling position i, so runtime
shift j only lowers readings at positions > j; readings are sticky
maxima until the detector reset bit is pulsed.
"""
import pytest

from mkidgen3.drivers.fftgain import FFTGainControl


def _reading(headroom_bits):
    """Peak reading for a given headroom in bits (guide table 5.5)."""
    return max(0, min(3, 4 - headroom_bits))


class FakeGpio:
    """Duck-typed AxiGPIO with the BxBFFT behind it.

    base[i] is detector i's headroom in bits when all runtime shifts are
    zero (the compile time static shift is upstream of every detector's
    dependence on runtime shifts, so it is folded into base).
    """

    def __init__(self, base):
        self.base = base
        self.ch1_state = 0x0FF  # hardware reset default: shifts 3,3,3,3
        self.resets = 0
        self._latched = self._current()
        gpio = self

        class Ch1:
            def write(self, val, mask):
                old = gpio.ch1_state
                gpio.ch1_state = (old & ~mask) | (val & mask)
                if gpio.ch1_state & 0x100 and not old & 0x100:
                    gpio.resets += 1
                    gpio._latched = gpio._current()
                gpio._latch()

        class Ch2:
            def read(self):
                gpio._latch()
                return gpio._pack(gpio._latched)

        self.channel1 = Ch1()
        self.channel2 = Ch2()

    def shifts(self):
        return [(self.ch1_state >> (2 * i)) & 3 for i in range(4)]

    def _current(self):
        s = self.shifts()
        return [
            _reading(self.base[i] + sum(s[:i]))
            for i in range(4)
        ]

    def _latch(self):
        self._latched = [
            max(a, b) for a, b in zip(self._latched, self._current())
        ]

    @staticmethod
    def _pack(readings):
        word = 0
        for i, r in enumerate(readings):
            word |= (r & 3) << (2 * i)
        return word


def test_pack_unpack_roundtrip():
    for shifts in [(0, 0, 0, 0), (3, 3, 3, 3), (2, 1, 0, 3)]:
        word = FFTGainControl.pack(shifts)
        assert FFTGainControl.unpack(word) == tuple(shifts)
    with pytest.raises(ValueError):
        FFTGainControl.pack((4, 0, 0, 0))
    with pytest.raises(ValueError):
        FFTGainControl.pack((0, 0, 0))


def test_set_shifts_preserves_reset_bit():
    gpio = FakeGpio(base=(4, 4, 4, 4))
    ctl = FFTGainControl(gpio)
    gpio.channel1.write(0x100, 0x100)  # hold detector reset high
    ctl.set_shifts((1, 2, 3, 0))
    assert gpio.ch1_state & 0x100
    assert gpio.shifts() == [1, 2, 3, 0]
    assert ctl.shifts == (1, 2, 3, 0)


def test_reset_overflow_pulses():
    gpio = FakeGpio(base=(4, 4, 4, 4))
    ctl = FFTGainControl(gpio)
    ctl.reset_overflow()
    assert gpio.resets == 1
    assert not gpio.ch1_state & 0x100


def test_readings_sticky_until_reset():
    gpio = FakeGpio(base=(1, 1, 1, 1))  # hot input: reading 3 at loc 0
    ctl = FFTGainControl(gpio)
    assert ctl.read_overflow()[0] == 3
    gpio.base = (4, 4, 4, 4)  # signal removed; sticky bits must persist
    assert ctl.read_overflow()[0] == 3
    assert ctl.measure(dwell=0)[0] == 0  # reset clears the latch


def test_autoset_nominal():
    gpio = FakeGpio(base=(4, 1, 0, -1))
    ctl = FFTGainControl(gpio)
    shifts, readings, input_hot = ctl.autoset(target=1, dwell=0)
    assert shifts == (2, 1, 1, 0)
    assert readings == (0, 1, 1, 1)
    assert not input_hot
    assert ctl.total_shift == 1 + 4
    assert ctl.gain == 2 ** -5


def test_autoset_cold_signal_reaches_minimum_shifts():
    gpio = FakeGpio(base=(10, 10, 10, 10))
    ctl = FFTGainControl(gpio)
    shifts, readings, input_hot = ctl.autoset(target=1, dwell=0)
    assert shifts == (0, 0, 0, 0)
    assert not input_hot
    assert ctl.total_shift == 1
    assert ctl.gain == 0.5


def test_autoset_flags_hot_input():
    gpio = FakeGpio(base=(1, 5, 5, 5))
    ctl = FFTGainControl(gpio)
    shifts, readings, input_hot = ctl.autoset(target=1, dwell=0)
    assert input_hot  # detector 0 is upstream of every runtime shift
    assert readings[0] == 3


def test_safe_restores_full_shifts():
    gpio = FakeGpio(base=(4, 1, 0, -1))
    ctl = FFTGainControl(gpio)
    ctl.autoset(target=1, dwell=0)
    ctl.safe()
    assert gpio.shifts() == [3, 3, 3, 3]
    assert ctl.total_shift == 13


class FakeOverlay:
    def __init__(self, params):
        self.ip_dict = {
            "photon_pipe/opfb/fft/axi_gpio_0": {"parameters": params}
        }


def test_present_detects_mode12_gpio():
    assert FFTGainControl.present(
        FakeOverlay({"C_IS_DUAL": "1", "C_GPIO_WIDTH": "9"})
    )
    # legacy SysGen scale GPIO: 12 wide, single channel
    assert not FFTGainControl.present(
        FakeOverlay({"C_IS_DUAL": "0", "C_GPIO_WIDTH": "12"})
    )
    assert not FFTGainControl.present(FakeOverlay({}))

    class NoGpio:
        ip_dict = {}

    assert not FFTGainControl.present(NoGpio())


# --- detector order probe -----------------------------------------------------
# The 2026-07-24 bench session claimed read_overflow comes back reversed, but
# its dataset held location 0 at shift 3 in every schedule, so it could not
# identify the mapping. probe_detector_order varies exactly that.

class _ProbeGPIO:
    """Fake GPIO whose ch2 readings respond to ch1's location-0 shift.

    `reversed_hw` picks which end of the returned word location 0 lands on.
    """

    def __init__(self, reversed_hw):
        self._reversed = reversed_hw
        self._word = 0
        outer = self

        class _Ch:
            def __init__(self, n):
                self.n = n

            def write(self, val, mask):
                if self.n == 1:
                    outer._word = (outer._word & ~mask) | (val & mask)

            def read(self):
                s0 = outer._word & 3
                # location 0 shift attenuates everything downstream of it;
                # its own detector is upstream and never moves.
                per_loc = [3, max(0, 3 - s0), max(0, 3 - s0), max(0, 3 - s0)]
                if outer._reversed:
                    per_loc = per_loc[::-1]
                return sum(v << (2 * i) for i, v in enumerate(per_loc))

        self.channel1 = _Ch(1)
        self.channel2 = _Ch(2)


def test_probe_detects_documented_order():
    g = FFTGainControl(_ProbeGPIO(reversed_hw=False))
    low, high, verdict = g.probe_detector_order(dwell=0)
    assert low[0] == high[0]          # location 0's own detector is fixed
    assert "documented order confirmed" in verdict


def test_probe_detects_reversed_order():
    g = FFTGainControl(_ProbeGPIO(reversed_hw=True))
    low, high, verdict = g.probe_detector_order(dwell=0)
    assert low[3] == high[3]
    assert "REVERSED" in verdict
