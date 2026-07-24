"""Driver for the iq_sweep_acc hardware sweep accumulator.

The core accumulates per channel [sumI, sumQ, sumI**2, sumQ**2] int64 sums
over n_frames full frames of the ddciq stream (each channel visited at
2 MS/s), then streams 2048 beats of {sumQQ, sumII, sumQ, sumI} (64 bit
fields, one channel per beat, 64 KB total) into capture switch input 4 for
DMA by axis2mm. One ap_start = one sweep point. The result path has no
backpressure: arm the capture BEFORE ap_start.
"""
import numpy as np

try:
    from pynq import DefaultIP
    _PYNQ = True
except ImportError:
    DefaultIP = object
    _PYNQ = False


def sums_to_mean_rms(sums, n):
    """Convert (N, 4) int64 [sumI, sumQ, sumII, sumQQ] to (mean, rms).

    Algebraically identical to the complex mean and per component population
    std (np.std default) of the raw int16 samples, computed exactly from the
    integer sums. Variance is clamped at zero against float rounding.
    """
    s = np.asarray(sums, dtype=np.float64)
    n = float(n)
    mi, mq = s[:, 0] / n, s[:, 1] / n
    mean = mi + 1j * mq
    var_i = np.maximum(s[:, 2] / n - mi * mi, 0.0)
    var_q = np.maximum(s[:, 3] / n - mq * mq, 0.0)
    rms = np.sqrt(var_i) + 1j * np.sqrt(var_q)
    return mean, rms


class IQSweepAccumulator(DefaultIP):
    bindto = ['mazinlab:mkidgen3:iq_sweep_acc:0.1']

    ADDR_CTRL = 0x00           # HLS ap_ctrl_hs: start/done/idle/ready
    ADDR_N_FRAMES = 0x10
    ADDR_DISCARD_FRAMES = 0x18
    AP_START = 0x1
    AP_DONE = 0x2
    AP_IDLE = 0x4
    AP_READY = 0x8

    MAX_FRAMES = 2 ** 20       # matches gen3d's MAX_SWEEP_AVERAGE

    def __init__(self, description):
        super().__init__(description=description)

    def start_point(self, n_frames, discard_frames=0):
        """Program one accumulation and pulse ap_start.

        The caller must have armed the capture DMA on switch source
        'iqsweep' first; the 2048 beat result burst follows ap_done with no
        flow control.
        """
        n_frames = int(n_frames)
        discard_frames = int(discard_frames)
        if not 0 < n_frames <= self.MAX_FRAMES:
            raise ValueError(f'n_frames must be in 1..{self.MAX_FRAMES}')
        if discard_frames < 0:
            raise ValueError('discard_frames must be nonnegative')
        self.write(self.ADDR_N_FRAMES, n_frames)
        self.write(self.ADDR_DISCARD_FRAMES, discard_frames)
        self.write(self.ADDR_CTRL, self.AP_START)

    @property
    def status(self):
        """Decode the ap_ctrl_hs control register in a single read.

        ap_done is clear on read, so prefer this whenever more than one bit
        matters: each individual property below costs its own read, and any
        of them can consume the done latch.
        """
        r = self.read(self.ADDR_CTRL)
        return {'start': bool(r & self.AP_START),
                'done': bool(r & self.AP_DONE),
                'idle': bool(r & self.AP_IDLE),
                'ready': bool(r & self.AP_READY)}

    @property
    def done(self):
        """True only if ap_done is latched.

        Deliberately does not fold in ap_idle. A core that was never started
        is idle, and reporting that as done sends anyone debugging a stalled
        capture looking in the wrong place -- use `status` to tell the two
        apart, since idle alone cannot distinguish "never ran" from
        "finished, and the done latch was already read".
        """
        return bool(self.read(self.ADDR_CTRL) & self.AP_DONE)

    @property
    def idle(self):
        """True if no accumulation is in progress (never started, or done)."""
        return bool(self.read(self.ADDR_CTRL) & self.AP_IDLE)

    @property
    def busy(self):
        """True while an accumulation/dump is in progress."""
        return not bool(self.read(self.ADDR_CTRL) & self.AP_IDLE)
