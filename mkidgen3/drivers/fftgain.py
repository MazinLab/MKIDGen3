"""Runtime gain control for the BxBFFT (GAIN_CONTROL_STRATEGY=12 builds).

The OPFB FFT (BxBFFT 4096-pt, 16 samples/clock, stages radix-16,4,4,4,4)
exposes four runtime shifter locations, one after each stage except the
last: ``fft_shifts_i`` packs 2 bits per location (right shift 0-3), added
to the compile time static shift of 1 at location 0 (which covers the
radix-16 first stage's 4-bit growth). With monitoring enabled, each
location also has 2 sticky peak-detect bits: 0 = peak more than 18 dB
below overflow, 1 = 12-18 dB, 2 = 6-12 dB, 3 = within 6 dB. Detector i
reads the PRE-shift peak at location i, so runtime shift j only affects
readings at locations > j; detector 0 responds only to the input drive.

Wiring (gen3_top fft hier): GPIO ch1 out [7:0] = shifts, [8] = detector
reset, through a CDC into the 512 MHz domain; sticky ``overflow_detect_o``
returns on GPIO ch2. The GPIO resets to 0x0FF = all shifts 3 (maximum
compensation, overflow proof) so a freshly loaded overlay is always safe.

Shift changes are asynchronous to the stream and corrupt the FFT frames
in flight: tune during setup, never during an observation. Changing the
shifts rescales every downstream amplitude (DDC, matched filters,
thresholds) by the reported :attr:`FFTGainControl.gain`.
"""
import time

STATIC_SHIFTS = (1, 0, 0, 0)
N_LOCATIONS = 4
_SHIFT_MASK = 0xFF
_RESET_BIT = 0x100
_GPIO_IP_PATH = "photon_pipe/opfb/fft/axi_gpio_0"
# Reading -> headroom in bits (reading 0 means "more than 3 bits"; treat
# as 4 for the last-stage shift heuristic).
_HEADROOM = {3: 1, 2: 2, 1: 3, 0: 4}


class FFTGainControl:
    """Wraps the fft hier gain GPIO (pynq ``AxiGPIO`` or any object with
    ``channel1.write(val, mask)`` / ``channel2.read()``)."""

    STATIC_SHIFTS = STATIC_SHIFTS

    def __init__(self, gpio):
        self._gpio = gpio
        self._shifts = None  # unknown until first set_shifts

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def present(overlay):
        """True when the overlay carries the mode 12 gain GPIO.

        The GPIO address is shared with the legacy SysGen scale GPIO; the
        mode 12 build is identified by its shape (9-bit dual channel).
        """
        try:
            params = overlay.ip_dict[_GPIO_IP_PATH]["parameters"]
            return (int(params.get("C_IS_DUAL", 0)) == 1
                    and int(params.get("C_GPIO_WIDTH", 0)) == 9)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    @classmethod
    def from_overlay(cls, overlay):
        if not cls.present(overlay):
            raise RuntimeError(
                "this overlay does not carry the BxBFFT mode 12 gain GPIO"
            )
        return cls(overlay.photon_pipe.opfb.fft.axi_gpio_0)

    # -- shift word --------------------------------------------------------
    @staticmethod
    def pack(shifts):
        shifts = tuple(int(s) for s in shifts)
        if len(shifts) != N_LOCATIONS:
            raise ValueError(f"need {N_LOCATIONS} shifts, got {len(shifts)}")
        if not all(0 <= s <= 3 for s in shifts):
            raise ValueError(f"shifts must be 0..3, got {shifts}")
        word = 0
        for i, s in enumerate(shifts):
            word |= s << (2 * i)
        return word

    @staticmethod
    def unpack(word):
        return tuple((int(word) >> (2 * i)) & 3 for i in range(N_LOCATIONS))

    # -- control -----------------------------------------------------------
    def set_shifts(self, shifts):
        """Program the four runtime shifts (corrupts in-flight FFT frames)."""
        word = self.pack(shifts)
        self._gpio.channel1.write(word, _SHIFT_MASK)
        self._shifts = self.unpack(word)

    def safe(self):
        """All shifts 3: maximum compensation, overflow proof (the
        hardware reset default)."""
        self.set_shifts((3,) * N_LOCATIONS)

    @property
    def shifts(self):
        return self._shifts

    @property
    def total_shift(self):
        """Total right shift including the compile time static base."""
        if self._shifts is None:
            raise RuntimeError("shifts unknown; call set_shifts/safe first")
        return sum(STATIC_SHIFTS) + sum(self._shifts)

    @property
    def gain(self):
        """FFT amplitude scaling relative to an uncompensated transform."""
        return 2.0 ** -self.total_shift

    # -- monitoring --------------------------------------------------------
    def reset_overflow(self):
        """Pulse the sticky detector reset."""
        self._gpio.channel1.write(_RESET_BIT, _RESET_BIT)
        self._gpio.channel1.write(0, _RESET_BIT)

    def read_overflow(self):
        """Four sticky peak readings (0-3), location 0 first.

        The ordering follows BxBFFT User Guide v3.2 s5.5 -- "The first
        scaling position has bits [1:0], the second has bits [3:2], etc."
        -- and the fft hier wires overflow_detect_o[7:0] straight through
        the CDC to gpio2_io_i, so no reordering happens in gateware.

        DISPUTED: the 2026-07-24 bench session concluded these come back
        reversed, on the grounds that position 0 moved when only downstream
        shifts changed, which nothing upstream can explain. That dataset
        cannot actually settle it -- every schedule it tested held location
        0 at shift 3, so the one variable that would identify the mapping
        was never varied. Run :meth:`probe_detector_order` to settle it
        before trusting either interpretation, and fix it in exactly one
        place (here or the gateware), never both.
        """
        return self.unpack(self._gpio.channel2.read() & _SHIFT_MASK)

    def probe_detector_order(self, dwell=0.01):
        """Identify which reading index responds to location 0's shift.

        Varies ONLY location 0 and reports which index moves. Location 0's
        shift is upstream of every detector except its own, so under the
        documented ordering index 0 must be the one that stays put while
        1..3 move together; under the disputed reversed ordering index 3
        stays put instead.

        Returns (readings_at_shift0_low, readings_at_shift0_high, verdict).
        Needs a live, steady input -- run it with the comb on and the drive
        unchanged for the duration.
        """
        rest = self._shifts[1:] if self._shifts is not None else (0, 0, 0)
        self.set_shifts((0,) + tuple(rest))
        low = self.measure(dwell)
        self.set_shifts((3,) + tuple(rest))
        high = self.measure(dwell)
        moved = tuple(i for i in range(N_LOCATIONS) if low[i] != high[i])
        if moved and 0 not in moved:
            verdict = "documented order confirmed (index 0 fixed, downstream moved)"
        elif moved and 3 not in moved:
            verdict = "REVERSED: index 3 is location 0 -- reverse the unpack"
        elif not moved:
            verdict = ("inconclusive: nothing moved. Detectors may be "
                       "saturated or the input is not driving the FFT")
        else:
            verdict = (f"inconclusive: indices {moved} moved, which fits "
                       f"neither ordering")
        return low, high, verdict

    def measure(self, dwell=0.01):
        """Reset detectors, dwell (an FFT frame is 500 ns; the 10 ms
        default averages ~20k frames), read the peaks."""
        self.reset_overflow()
        if dwell:
            time.sleep(dwell)
        return self.read_overflow()

    # -- closed loop -------------------------------------------------------
    def autoset(self, target=1, dwell=0.01):
        """Walk the shifts down for maximum gain at ``target`` headroom.

        target is the highest acceptable peak reading downstream of each
        shifter (1 = 12-18 dB below overflow, Ross's suggested operating
        point). Starts from the safe all-3 state, then lowers locations
        0..2 greedily while every downstream detector stays at or below
        target. Location 3 has no downstream detector (it protects the
        unmonitored last stage and the 18->16 bit output conversion), so
        it is set from detector 3's final reading: enough shift to keep
        the last stage's 2-bit growth clear of the output's headroom.

        Returns (shifts, readings, input_hot); input_hot warns that
        detector 0 - upstream of every runtime shift - exceeds target,
        which only the analog/DAC drive can fix.
        """
        self.safe()
        readings = self.measure(dwell)
        shifts = list(self._shifts)
        for loc in range(N_LOCATIONS - 1):
            while shifts[loc] > 0:
                trial = list(shifts)
                trial[loc] -= 1
                self.set_shifts(trial)
                r = self.measure(dwell)
                if any(r[k] > target for k in range(loc + 1, N_LOCATIONS)):
                    self.set_shifts(shifts)  # revert
                    break
                shifts = trial
                readings = r
        shifts[3] = max(0, 3 - _HEADROOM[readings[3]])
        self.set_shifts(shifts)
        readings = self.measure(dwell)
        input_hot = readings[0] > target
        return self._shifts, tuple(readings), input_hot
