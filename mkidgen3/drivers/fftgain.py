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

        The hardware word is REVERSED relative to BxBFFT User Guide v3.2
        s5.5, which says the first scaling position occupies bits [1:0].
        It does not: location 0 arrives in the HIGH pair, so this unpacks
        and then reverses. Two independent bench results agree, 2026-07-24:

          * Across the original schedule scan, the raw index 0 varied while
            only locations 1 and 2 changed. Nothing upstream of location 0
            moved, so under the documented order that reading could not
            have changed at all.
          * Pinning location 0 at shift 3 and sweeping locations 1-2, raw
            indices 2 and 3 held constant while 0 and 1 swept their full
            range -- the exact opposite of what the documented order
            requires, and index 2 sat at reading 1 with room to move.

        FIX THIS IN ONE PLACE ONLY. It is corrected here, so the gateware
        must keep wiring overflow_detect_o[7:0] straight through to
        gpio2_io_i. Reversing there too would cancel this out.
        """
        return self.unpack(self._gpio.channel2.read() & _SHIFT_MASK)[::-1]

    def probe_detector_order(self, dwell=0.01):
        """Check that :meth:`read_overflow` really is location ordered.

        Varies ONLY location 0. Its shift is upstream of every detector
        except its own, so in a correctly ordered result index 0 holds
        still while 1..3 move together. If index 3 is the one holding
        still, the returned word is reversed relative to what this driver
        assumes and the reversal in read_overflow is wrong.

        A reading railed at 0 or 3 cannot move either, so "did not move"
        is only evidence when the candidate is strictly between 0 and 3.
        Ignoring that made this probe report the wrong answer at 11.4% and
        18.2% ADC fill, where index 0 sits floored at 0 in both states
        (bench, 2026-07-24). Drive the FFT into midscale before trusting
        it, and prefer a fill where the fixed index reads 1 or 2.

        Returns (readings_at_shift0_low, readings_at_shift0_high, verdict).
        Needs a live, steady input -- comb on, drive unchanged throughout.
        """
        rest = self._shifts[1:] if self._shifts is not None else (0, 0, 0)
        self.set_shifts((0,) + tuple(rest))
        low = self.measure(dwell)
        self.set_shifts((3,) + tuple(rest))
        high = self.measure(dwell)
        moved = tuple(i for i in range(N_LOCATIONS) if low[i] != high[i])

        def informative(i):
            """A candidate 'fixed' index only counts if it had room to move."""
            return 0 < low[i] < 3 and 0 < high[i] < 3

        if not moved:
            return low, high, ("inconclusive: nothing moved. Detectors may be "
                               "saturated or the input is not driving the FFT")
        if 0 not in moved and informative(0):
            verdict = "ORDER OK: index 0 held (with room to move), downstream moved"
        elif 3 not in moved and informative(3):
            verdict = ("REVERSED: index 3 held (with room to move) -- "
                       "read_overflow's reversal is wrong, drop it")
        elif 0 not in moved or 3 not in moved:
            stuck = 0 if 0 not in moved else 3
            verdict = (f"inconclusive: index {stuck} held but is railed at "
                       f"{low[stuck]}, so it could not have moved regardless. "
                       f"Re-run at an ADC fill that puts it mid-range")
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
