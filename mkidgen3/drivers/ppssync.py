import asyncio
import logging
import time

from enum import Enum, IntFlag
from math import floor

from pynq import DefaultIP

NS_PER_SEC = 1000 * 1000 * 1000
SUBNS_PER_NS = 256


def _field_get(reg, lsb, mask):
    return (reg >> lsb) & mask


def _field_set(reg, lsb, mask, val):
    current = reg & (0xFFFFFFFF ^ (mask << lsb))
    return current | ((val & mask) << lsb)


def _make_time_ns(secs, ns, subns):
    return secs * NS_PER_SEC + ns + subns / SUBNS_PER_NS


class PPSMode(Enum):
    """Enumerate PPS synchronizer modes."""

    STOPPED = 0
    """Stops the PPS synchronization engine"""

    FORCE_START = 1
    """Immediatly starts the PPS Engine ignoring PPS pulses"""

    PPS_FREERUN = 2
    """Starts on the next PPS edge and then counts ignoring all further PPS
    edges
    """

    PPS_SYNC = 3
    """Probably the most useful mode, starts on the next PPS edge and rolls
    over on subsequent PPS edges
    """

    SYSREF_SYNC = 4
    """Starts on the next SYSREF edge and rolls over on subsequent PPS edges,
    this is only really useful with the SYNC gated Sysref of the LMK04828 to
    sync boards to < 1 clock period
    """

    SYSREF_FREERUN = 5
    """Starts on the next SYSREF edge then lets the counter run free"""


class CaptureMode(Enum):
    """Enumerates counter capture modes."""

    CLEAR = 0
    """Clear the current capture and initilize the capture engine for another
    capture.
    """

    PPS_EDGE = 1
    """Capture the counter value on the next PPS Edge"""

    SYSREF_EDGE = 2
    """Capture the counter value on the next SYSREF Edge"""


class SanityMode(IntFlag):
    """Enumartes options for sanity checking on PPS rollover."""

    LOCKOUT = 1
    """Effectively debouncing, ignores PPS Edges for a period of time
    after the counter rolls over
    """

    SELF_ROLLOVER = 2
    """If the counter exceeds a certain threshold then rollover without a PPS
    Edge
    """

    RESYNC = 4
    """Handy for a flaky PPS signal that drops in and out; enables both LOCKOUT
    and MISSDETECT, but if a PPS edge is detected inside the debounce window
    then we disable the next MISSDETCT window and wait for the next pulse.

    If the PPS generator is really bad you probably want PPSMode.PPS_FREERUN
    and you can use the capture engine and live setting of the clock period
    to implement a software PLL.
    """


class PPSSource(IntFlag):
    """As of 9/13/23 the pps engine has three PPS inputs connected as described,
    they can be or-ed together to generate an internal PPS pulse by or-ing
    several of these enums together (IE `PPSSource.PPS0 | PPSSource.PPS1`)
    """

    PPS0 = 1
    """As of 9/13/23 this comes from the schmidt trigger on the RFSoC4x2"""

    PPS1 = 2
    """As of 9/13/23 this comes from the comparator on the RFSoC4x2"""

    PPS2 = 4
    """As of 9/13/23 this is unconnected on the RFSoC4x2, in the future it
    may be connected to the slow ADC that reads the PPS input"""


class PPSSync(DefaultIP):
    bindto = ["MazinLab:mkidgen3:pps_synchronizer_control:0.3"]

    def __init__(self, description):
        super().__init__(description=description)
        self.capture_event = asyncio.Event()
        self.started_event = asyncio.Event()
        self._shadow_mode = 0
        self._shadow_counter_config = 0
        self._shadow_delay_ns = 0
        self._shadow_rollover = 0
        self._shadow_lockout = 0
        self._shadow_load_time = [0, 0, 0]

    # High-level api
    async def start_engine(
        self,
        mode=PPSMode.PPS_SYNC,
        skew=None,
        load_time=None,
        lockout=None,
        rollover_thresh=None,
        resync=False,
        start_after=None,
        pps_source=PPSSource.PPS0,
        clk_period_ns=1.953125,
        timeout=5,
        poll=0,
    ):
        self.mode = PPSMode.STOPPED
        self.pps_source = pps_source
        self.delay_ns = 0
        self.started_event.clear()

        if skew is None:
            skew = await self.sample_skew(timeout, poll)
        self.delay_ns = NS_PER_SEC - skew

        self.ns_per_clk = int(floor(clk_period_ns))
        self.subns_per_clk = int((clk_period_ns % 1.0) * 256)

        if load_time is None:
            load_time = time.time_ns() + NS_PER_SEC
        elif type(load_time) == float:
            load_time = int(load_time)
        if type(load_time) == int:
            load_time = [load_time // NS_PER_SEC, load_time % NS_PER_SEC, 0]
        self.load_secs, self.load_ns, self.load_subns = list(load_time)

        sanity_mode = SanityMode(0)

        if lockout is not None:
            self.lockout = lockout
            sanity_mode |= SanityMode.LOCKOUT
        if rollover_thresh is not None:
            self.rollover_thresh = rollover_thresh
            sanity_mode |= SanityMode.SELF_ROLLOVER
        if resync:
            assert (
                sanity_mode == SanityMode.LOCKOUT | SanityMode.SELF_ROLLOVER
            ), "Resync requires both a lockout value and rollover threshold"
            sanity_mode |= SanityMode.RESYNC
        self.sanity_mode = sanity_mode

        if start_after is not None:
            while time.time_ns() < start_after:
                asyncio.sleep(poll)

        self.mode = mode
        if mode == PPSMode.STOPPED:
            return

        start_time = time.time()
        while timeout == 0 or time.time() - start_time < timeout:
            if self.started:
                self.started_event.set()
                return
            await asyncio.sleep(poll)
        raise TimeoutError(
            "PPS Engine did not start before timeout, is the PPS source correct?"
        )

    def stop_engine(self):
        self.mode = PPSMode.STOPPED

    async def capture(
        self, capture_mode=CaptureMode.PPS_EDGE, timeout=5, poll=0, fmt=float
    ):
        assert fmt in (
            tuple,
            float,
        ), "Valid capture timestamp formats are tuple and float"
        self.capture_mode = CaptureMode.CLEAR
        self.capture_event.clear()
        self.capture_mode = capture_mode
        start_time = time.time()
        while timeout == 0 or time.time() - start_time < timeout:
            if self.captured:
                self.capture_event.set()
                if fmt == tuple:
                    return (self.captured_secs, self.captured_ns, self.captured_subns)
                if fmt == float:
                    return self.captured_time_ns
            await asyncio.sleep(poll)
        raise TimeoutError("Capture didn't occur before timeout")

    async def sample_skew(self, timeout=5, poll=0):
        assert (
            self.mode == PPSMode.STOPPED and self.delay_ns == 0
        ), "Turn off the counter, and set the delay to zero"
        await self.start_engine(PPSMode.PPS_SYNC, timeout, poll)
        pps_time = time.time_ns()
        self.stop_engine()
        return pps_time % NS_PER_SEC

    # Individual register fields, quasi user-accessible
    @property
    def started(self):
        """Is 1 if the PPS counter engine is started, 0 otherwise"""
        return _field_get(self.register_map.counter_status_reg.counter_status_reg, 0, 1)

    @property
    def captured(self):
        """Us 1 if the capture engine has captured, 0 otherwise"""
        return _field_get(self.register_map.counter_status_reg.counter_status_reg, 8, 1)

    @property
    def mode(self):
        """This is the PPS Counter engines mode, of type :py:class:`mkidgen3.drivers.ppssync.PPSMode`"""
        return PPSMode(_field_get(self._mode_reg, 0, 0b1111))

    @mode.setter
    def mode(self, value):
        self._mode_reg = _field_set(self._mode_reg, 0, 0b1111, value.value)

    @property
    def capture_mode(self):
        """The Capture Mode of the capture engine, of type :py:class:`mkidgen3.drivers.ppssync.CaptureMode`"""
        return CaptureMode(_field_get(self._mode_reg, 8, 0b11))

    @capture_mode.setter
    def capture_mode(self, value):
        self._mode_reg = _field_set(self._mode_reg, 8, 0b11, value.value)

    @property
    def sanity_mode(self):
        """The Sanity Mode of the counter engine, of type :py:class:`mkidgen3.drivers.ppssync.SanityMode`"""
        return SanityMode(_field_get(self._mode_reg, 16, 0b111))

    @sanity_mode.setter
    def sanity_mode(self, value):
        self._mode_reg = _field_set(self._mode_reg, 16, 0b111, value.value)

    @property
    def pps_source(self):
        return PPSSource(_field_get(self._mode_reg, 24, 0b111))

    @pps_source.setter
    def pps_source(self, value):
        self._mode_reg = _field_set(self._mode_reg, 24, 0b111, value.value)

    @property
    def ns_per_clk(self):
        return _field_get(self._counter_config, 8, 0b111111)

    @ns_per_clk.setter
    def ns_per_clk(self, value):
        self._counter_config = _field_set(self._counter_config, 8, 0b111111, value)

    @property
    def subns_per_clk(self):
        return _field_get(self._counter_config, 0, 0xFF)

    @subns_per_clk.setter
    def subns_per_clk(self, value):
        self._counter_config = _field_set(self._counter_config, 0, 0xFF, value)

    # Shadow Registers
    @property
    def _mode_reg(self):
        """Provides a read write interface to the underlying mode register
        which is write-only, should not be called by user code as _mode_reg
        contains several config registers one of which is also confusingly
        what is set by PPSSync.mode, I'm very sorry."""
        return self._shadow_mode

    @_mode_reg.setter
    def _mode_reg(self, value):
        self._shadow_mode = value
        self.register_map.mode_reg.mode_reg = value

    @property
    def _counter_config(self):
        """See PPSSynch._mode_reg"""
        return self._shadow_counter_config

    @_counter_config.setter
    def _counter_config(self, value):
        self._shadow_counter_config = value
        self.register_map.counter_config_reg.counter_config_reg = value

    @property
    def lockout(self):
        """The lockout interval in ns (See :py:class:`mkidgen3.drivers.ppssync.SanityMode`)"""
        return self._shadow_lockout

    @lockout.setter
    def lockout(self, value):
        self._shadow_lockout = value
        self.register_map.lockout_reg.lockout_reg = value

    @property
    def rollover_thresh(self):
        """The rollover threshold in ns (See :py:class:`mkidgen3.drivers.ppssync.SanityMode`)"""
        return self._shadow_rollover

    @rollover_thresh.setter
    def rollover_thresh(self, value):
        self._shadow_rollover = value
        self.register_map.rollover_thresh_reg.rollover_thresh_reg = value

    @property
    def load_secs(self):
        """The seconds counter value to load when we start the PPS engine"""
        return self._shadow_load_time[0]

    @load_secs.setter
    def load_secs(self, value):
        self._shadow_load_time[0] = value
        self.register_map.load_secs_reg.load_secs_reg = value

    @property
    def load_ns(self):
        """The ns counter value to load when we start the PPS engine"""
        return self._shadow_load_time[1]

    @load_ns.setter
    def load_ns(self, value):
        self._shadow_load_time[1] = value
        self.register_map.load_ns_reg.load_ns_reg = value

    @property
    def load_subns(self):
        """The subns counter value to load when we start the PPS engine"""
        return self._shadow_load_time[2]

    @load_subns.setter
    def load_subns(self, value):
        self._shadow_load_time[2] = value
        self.register_map.load_subns_reg.load_subns_reg = value

    @property
    def delay_ns(self):
        """The ns counter value to load when we start the PPS engine"""
        return self._shadow_delay_ns

    @delay_ns.setter
    def delay_ns(self, value):
        self._shadow_delay_ns = value
        self.register_map.delay_ns_reg.delay_ns_reg = value

    # Timer readout stuffs
    @property
    def captured_time_ns(self):
        """The time captured by the core in floating point nanoseconds"""
        if not self.captured:
            logging.getLogger().warning(
                "Captured time read while registers are invalid"
            )
        return _make_time_ns(self.captured_secs, self.captured_ns, self.captured_subns)

    @property
    def captured_time(self):
        """The Time captured by the core in floating point seconds"""
        return self.captured_time_ns / (1000 * 1000 * 1000)

    @property
    def current_time_ns(self):
        """The current time according to the core in nanoseconds"""
        secs = self.secs
        # subns just won't be accurate
        ns = self.ns
        after_secs = self.secs
        if after_secs != self.secs:
            return self.current_time_ns
        return _make_time_ns(secs, ns, 0)

    @property
    def current_time(self):
        """The current time according to the core in floating point seconds"""
        return self.current_time_ns / (1000 * 1000 * 1000)

    @property
    def captured_subns(self):
        """The raw captured subns counter value"""
        return self.register_map.capture_subns_reg.capture_subns_reg

    @property
    def captured_ns(self):
        """The raw captured ns counter value"""
        return self.register_map.capture_ns_reg.capture_ns_reg

    @property
    def captured_secs(self):
        """The raw captured seconds counter value"""
        return self.register_map.capture_secs_reg.capture_secs_reg

    @property
    def subns(self):
        """The raw current subns counter value"""
        return self.register_map.current_subns_reg.current_subns_reg

    @property
    def ns(self):
        """The raw current ns counter value"""
        return self.register_map.current_ns_reg.current_ns_reg

    @property
    def secs(self):
        """The raw current seconds counter value"""
        return self.register_map.current_secs_reg.current_secs_reg