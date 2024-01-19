import threading
import logging
import time

import mkidgen3.registers as registers

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
    """Immediately starts the PPS Engine ignoring PPS pulses"""

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
        self.capture_event = threading.Event()
        self.started_event = threading.Event()

    # High-level api
    def start_engine(
        self,
        mode=PPSMode.PPS_SYNC,
        start_second=None,
        skew=None,
        lockout=None,
        rollover_thresh=None,
        resync=False,
        pps_source=PPSSource.PPS0,
        load_time=None,
        clk_period_ns=1.953125,
        timeout=5,
        poll=0,
    ):
        """Starts the PPS engine in varius modes

        This will configure the PPS engine and provides the tools needed to
        achieve board to board timestamp synchronization.

        If you have a reliable PPS Source with PPS edges on the second edge
        and assumptions about board clock accuracy are met then you can simply
        do `start_engine(skew = 0)` on all of the boards you want to
        synchronize.

        If your PPS Edges are not aligned then you will likely want go choose
        a board and measure the skew relative to the system clock using
        `PPSSync.sample_skew` and pass that skew to all the boards you want
        to start with `start_engine(skew = skew_measured)`. If you are
        using only one board then `start_engine()` will work

        If you do not have a PPS Source available then using
        `start_engine(PPSMode.FORCE_START)` will start the PPS Engine
        with a the system time plus or minus around a dozen microseconds

        Parameters
        ----------
        mode : PPSMode
            The mode you would like to start the PPS engine, SYSREF modes are
            not considered part of the stable API and should not be used unless
            you know what you are doing
        start_second : int or NoneType
            The UTC second you would like PPS engine to begin counting at,
            should be at least 4 seconds in the future or None
        skew : int or NoneType
            The skew between the true second and the PPS Signal edge
            in nanoseconds. If the PPS signal comes from a GPS source set this
            to zero otherwise set it to a common value accross all boards. If
            this is set to None the skew will be sampled relative to the system
            clock in PPS based modes.
        lockout : int or NoneType
            The debounce interval in nanoseconds during which PPS edges will be
            ignored. For the FS725 this can likely safely be left as None,
            otherwise set to the risetime of your PPS Signal.
        rollover : int or NoneType
            The number of nanoseconds after which the nanosecond counter should
            rollover without the presence of a PPSEdge. For the FS725 this can
            be safely set to 1000*1000*1000 if your clocking configuration
            is correct. If this and lockout are set resync should likely be set
        resync : bool
            If this is set and a PPS edge is detected in the lockout period
            after an automatic rollover then the counter will not rollover
            until another PPS Edge has been seen
        pps_source : PPSSource
            The PPS Source selected, see enum for documentation, PPS Signals
            can be ord together
        load_time : int or float or list or tuple or NoneType
            The time in nanoseconds (float or int), or a tuple/list of seconds,
            nanoseconds, and subns which you would like to load into the engine
            when it is started. If None the utc second on which we are starting
            will be chosen.
        clk_period_ns : float
            The clock period of the clock domain the block exists in. This
            should be 1/512MHz as of Nov 2023
        timeout : int or float
            The maximum number of seconds to wait for a pps edge before timing
            out worst case execution time is less than 4 + 2*timeout seconds
        poll : float
            The number of seconds to wait between polling for the status of
            the engine

        Assumptions
        -----------
        - The systems clock is accurate to < 0.2 seconds and that a leap
          second is not about to occur
        - The system is fast enough that ~12 writes to registers and
          a small amount of floating point math can be done in python in
          under a second
        """
        self.mode = PPSMode.STOPPED
        self.pps_source = pps_source
        self.delay_ns = 0
        self.started_event.clear()

        if lockout == 0:
            lockout = None

        if mode == PPSMode.PPS_SYNC or mode == PPSMode.PPS_FREERUN:
            if skew is None:
                skew = self.sample_skew(timeout, poll)
            if start_second is None:
                start_second = time.time() + 1
                if start_second % 1.0 > 0.75:
                    start_second = start_second + 1
                start_second = start_second // 1
        else:
            if skew is None:
                skew = 0
            if start_second is None:
                start_second = time.time()
        start_after = start_second - 0.2
        self.delay_ns = (NS_PER_SEC - skew) % NS_PER_SEC

        self.ns_per_clk = int(floor(clk_period_ns))
        self.subns_per_clk = int((clk_period_ns % 1.0) * 256)

        if load_time is None:
            load_time = int(start_second * NS_PER_SEC)
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
                time.sleep(poll)

        self.mode = mode
        if mode == PPSMode.STOPPED:
            self.started_event.clear()
            return

        start_time = time.time()
        while timeout == 0 or time.time() - start_time < timeout:
            if self.started:
                self.started_event.set()
                return
            time.sleep(poll)
        raise TimeoutError(
            "PPS Engine did not start before timeout, is the PPS source correct?"
        )

    def stop_engine(self):
        """Stop the PPS Engine"""
        self.started_event.clear()
        self.mode = PPSMode.STOPPED

    def capture(
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
            time.sleep(poll)
        raise TimeoutError("Capture didn't occur before timeout")

    def sample_skew(self, timeout=5, poll=0):
        """Captures the sub-second portion of the system time at a PPS Edge
        The PPS Engine should be stopped before you call this. Should be
        accurate to less than 50 microseconds with the system under reasonable
        load.

        Parameters
        ----------
        timeout : int or float
            The amount of time to wait for a PPS Edge in seconds
        poll : float
            The time between checking for a capture in seconds

        Returns
        -------
        skew : int
            The skew between the PPS Edge and the system time in nanoseconds
        """
        assert (
            self.mode == PPSMode.STOPPED and self.delay_ns == 0
        ), "Turn off the counter, and set the delay to zero"
        self.start_engine(PPSMode.PPS_SYNC, timeout, poll)
        pps_time = time.time_ns()
        self.stop_engine()
        return pps_time % NS_PER_SEC

    # Base Registers
    @registers.register_ro
    def _counter_status_reg(self):
        return self.register_map.counter_status_reg

    @registers.register_shadow(0)
    def _mode_reg(self):
        """Provides a read write interface to the underlying mode register
        which is write-only, should not be called by user code as _mode_reg
        contains several config registers one of which is also confusingly
        what is set by PPSSync.mode, I'm very sorry."""
        return self.register_map.mode_reg

    @registers.register_shadow(0)
    def _counter_config_reg(self):
        return self.register_map.counter_config_reg

    # Quasi user-accessible registers
    @registers.register_shadow(0)
    def lockout(self):
        """The lockout interval in ns (See :py:class:`mkidgen3.drivers.ppssync.SanityMode`)"""
        return self.register_map.lockout_reg

    @registers.register_shadow(0)
    def rollover_thresh(self):
        """The rollover threshold in ns (See :py:class:`mkidgen3.drivers.ppssync.SanityMode`)"""
        return self.register_map.rollover_thresh_reg

    @registers.register_shadow(0)
    def load_secs(self):
        """The seconds counter value to load when we start the PPS engine"""
        return self.register_map.load_secs_reg

    @registers.register_shadow(0)
    def load_ns(self):
        """The ns counter value to load when we start the PPS engine"""
        return self.register_map.load_ns_reg

    @registers.register_shadow(0)
    def load_subns(self):
        """The subns counter value to load when we start the PPS engine"""
        return self.register_map.load_subns_reg

    @registers.register_shadow(0)
    def delay_ns(self):
        """The time in ns to delay each PPS Edge"""
        return self.register_map.delay_ns_reg

    @registers.register_ro
    def captured_subns(self):
        """The raw captured subns counter value"""
        return self.register_map.capture_subns_reg

    @registers.register_ro
    def captured_ns(self):
        """The raw captured ns counter value"""
        return self.register_map.capture_ns_reg

    @registers.register_ro
    def captured_secs(self):
        """The raw captured secs counter value"""
        return self.register_map.capture_secs_reg

    @registers.register_ro
    def subns(self):
        """The raw current subns counter value"""
        return self.register_map.current_subns_reg

    @registers.register_ro
    def ns(self):
        """The raw current ns counter value"""
        return self.register_map.current_ns_reg

    @registers.register_ro
    def secs(self):
        """The raw current secs counter value"""
        return self.register_map.current_secs_reg

    # Individual register fields, quasi user-accessible    
    started = registers.field_bool(0, _counter_status_reg)
    captured = registers.field_bool(8, _counter_status_reg)

    mode = registers.field_enum(slice(0, 4), PPSMode, _mode_reg)
    capture_mode = registers.field_enum(slice(8, 10), CaptureMode, _mode_reg)
    sanity_mode = registers.field_enum(slice(16, 19), SanityMode, _mode_reg)
    pps_source = registers.field_enum(slice(24, 27), PPSSource, _mode_reg)

    ns_per_clk = registers.field(slice(8, 14), _counter_config_reg)
    subns_per_clk = registers.field(slice(0, 8), _counter_config_reg)

    # Timer readout stuffs
    @property
    def captured_time_ns(self):
        """The time captured by the core in floating point nanoseconds"""
        if not self.captured:
            logging.getLogger(__name__).warning(
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
