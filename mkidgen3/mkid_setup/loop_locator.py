import logging
import time

import numpy as np
import numpy.typing as nt
import scipy.optimize as opt
import scipy.signal as sig

from typing import Optional, Callable
from dataclasses import dataclass

from mkidgen3.server.feedline_config import WaveformConfig, DDCConfig
from mkidgen3.drivers.ddc import ThreepartDDC

LO_QUANT = 0.596


@dataclass
class SweepConfig:
    steps: nt.NDArray[np.float64]
    waveform: WaveformConfig
    lo_center: float | np.float64 = 6000.0
    average: int = 1024
    tap: str = "ddciq"

    _idle = 0.01
    _flush = 64

    @classmethod
    def from_bandwidth(
        cls,
        bandwidth: float | np.float64,
        points: int,
        waveform: WaveformConfig,
        lo_center: float | np.float64 = 6000.0,
        average: int = 1024,
    ) -> "SweepConfig":
        spacing = LO_QUANT * np.floor(((bandwidth / points) / LO_QUANT))

        # These are not going to be actually quantized properly because FP but the LO algo on the IF Board is FP too.
        # We will probably want to fix both of these at some point in the near future as the IF Board is FP32
        steps = (np.arange(points) * spacing - spacing * points // 2) / 1e6

        return SweepConfig(steps, waveform, lo_center, average)

    @classmethod
    def from_comb(
        cls,
        points: int,
        waveform: WaveformConfig,
        lo_center: float | np.float64 = 6000.0,
        average: int = 1024,
    ) -> "SweepConfig":
        tones = waveform.waveform.freqs
        assert np.allclose(
            tones, tones[0] + np.arange(tones.size) * (tones[1] - tones[0])
        ), "Expected waveform to have evenly spaced comb"

        steps = np.linspace(0, tones[1] - tones[0], points, endpoint=False) / 1e6
        return SweepConfig(steps, waveform, lo_center, average)

    def reset_ddc(self, ddccontrol) -> "SweepConfig":
        ddccontrol.configure(**self.waveform.default_ddc_config.settings_dict())
        return self

    def run_sweep(self, ifboard, capture) -> "Sweep":
        tones = self.waveform.waveform.freqs
        iq = np.empty((tones.size, self.steps.size), np.complex64)
        rms = np.empty((tones.size, self.steps.size), np.complex64)
        for i, step in enumerate(self.steps):
            ifboard.set_lo(
                step + self.lo_center,
                fractional=True,
                g2_mode=False,
                full_calibration=True,
            )
            piq, prms = self._get_iq_point_rms(capture)
            iq[::, i] = piq[: tones.size]
            rms[::, i] = prms[: tones.size]
        ifboard.set_lo(
            self.lo_center, fractional=True, g2_mode=False, full_calibration=True
        )
        return Sweep(iq, rms / np.sqrt(self.average), self)

    def frequencies(self) -> nt.NDArray[np.float64]:
        tones = self.waveform.waveform.freqs
        freqs = np.zeros((tones.size, self.steps.size), dtype=np.float64)
        freqs += self.lo_center * 1e6
        freqs += self.steps * 1e6
        freqs += tones.reshape((tones.size, 1))
        return freqs

    def _get_iq_point_rms(self, capture):
        """
        Args:
            n: int
            how many points to average
        Returns: a single averaged iq data point captured from res channel 0
        """
        tones = self.waveform.waveform.freqs
        time.sleep(self._idle)
        if self._flush:
            x = capture.capture_iq(self._flush, tap_location=self.tap)
            del x

        # CortexA53 is not good at math... try and keep to f32
        x = capture.capture_iq(self.average, tap_location=self.tap)
        ps_buf = np.empty((self.average, tones.size, 2), dtype=np.float32)
        ps_buf[::, ::, ::] = np.ascontiguousarray(
            x[::, : tones.size, ::], dtype=np.int16
        )
        if hasattr(self, "verify"):
            ps_buf_old = np.copy(ps_buf)
        del x

        means = np.sum(ps_buf, axis=0) / np.float32(self.average)
        ps_buf -= means.reshape((1, tones.size, 2))
        ps_buf *= ps_buf
        rmses = np.sqrt(np.sum(ps_buf, axis=0) / np.float32(self.average))

        if hasattr(self, "verify"):
            means_golden = np.empty((tones.size, 2), dtype=np.float64)
            rmses_golden = np.empty((tones.size, 2), dtype=np.float64)
            ps_buf_old.mean(axis=0, out=means_golden)
            ps_buf_old.std(axis=0, out=rmses_golden)
            assert np.allclose(means_golden, means)
            assert np.allclose(rmses_golden, rmses)

        return means[::, 0] + 1j * means[::, 1], rmses[::, 0] + 1j * rmses[::, 1]


@dataclass
class Sweep:
    iq: nt.NDArray[np.complex64]
    iqsigma: nt.NDArray[np.complex64]
    config: "SweepConfig"

    @property
    def frequencies(self) -> nt.NDArray[np.float64]:
        return self.config.frequencies()

    def plot(
        self,
        ax,
        stacked: bool = False,
        newtones: Optional[list[float] | nt.NDArray[np.float64]] = None,
    ):
        for i in range(self.iq.shape[0]):
            l = ax.semilogy(
                (self.frequencies[i] / 1e6 if not stacked else self.config.steps),
                np.abs(self.iq[i]),
                label="Tone: {:.3f} MHz".format(
                    self.config.waveform.default_ddc_config.tones[i] / 1e6
                ),
            )
            if newtones:
                if stacked:
                    ax.axvline(
                        newtones[i] / 1e6
                        - self.config.waveform.default_ddc_config.tones[i] / 1e6,
                        color=l[0].get_color(),
                        linestyle="--",
                    )
                else:
                    ax.axvline(
                        newtones[i] / 1e6 + self.config.lo_center,
                        color=l[0].get_color(),
                        linestyle="--",
                    )
        if stacked:
            ax.axvline(0, color="black", lw=0.1, label="LO")
        ax.legend()
        ax.set_ylabel("S21 (Magnitude)")
        ax.set_xlabel("Frequency (MHz)")

    def plot_loops(
        self,
        ax
    ):
        for i in range(self.iq.shape[0]):
            ax.plot(self.iq[i].real, self.iq[i].imag)
        ax.set_aspect('equal')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')


def pointslope_intersect(
    m0, x0, y0, m1, x1, y1, m0r=0, x0r=0, y0r=0, m1r=0, x1r=0, y1r=0
):
    xin = y0 - y1 + m1 * x1 - m0 * x0
    xid = m1 - m0
    xi = xin / xid

    yip = xi - x0
    yi = y0 + m0 * yip

    xirn = np.sqrt(
        y0r**2
        + y1r**2
        + (m1 * x1 * np.sqrt((m1r / m1) ** 2 + (x1r / x1) ** 2)) ** 2
        + (m0 * x0 * np.sqrt((m0r / m0) ** 2 + (x0r / x0) ** 2)) ** 2
    )
    xird = np.sqrt(m1r**2 + m0r**2)
    xir = xi * np.sqrt((xirn / xin) ** 2 + (xird / xid) ** 2)
    yirp = np.sqrt(xir**2 + x0r**2)
    yir = np.sqrt(y0r**2 + ((m0r / m0) ** 2 + (yirp / yip) ** 2) * ((yip * m0) ** 2))

    return xi, yi, xir, yir


def slope_midpoint(x0, x1, y0, y1, x0r, x1r, y0r, y1r):
    md = x1 - x0
    mn = y1 - y0

    mnr = np.sqrt(y1r**2 + y0r**2)
    mdr = np.sqrt(x1r**2 + x0r**2)

    m = mn / md
    mr = m * np.sqrt((mnr / mn) ** 2 + (mdr / md) ** 2)

    ymidr = mnr / 2
    xmidr = mdr / 2

    xmid = (x1 + x0) / 2
    ymid = (y1 + y0) / 2

    return m, mr, xmid, ymid, xmidr, ymidr


def approximate_center(i, q, ir, qr, plot=None, holdoff=6):
    assert i.size == q.size

    mi = q.size // 2

    cestimates = np.zeros((q.size // 2 - 1 - holdoff, 2))
    cestimaters = np.zeros((q.size // 2 - 1 - holdoff, 2))

    for p in range(holdoff, q.size // 2 - 1):
        mp, mpr, xmp, ymp, xmpr, ympr = slope_midpoint(
            i[mi], i[mi + p], q[mi], q[mi + p], ir[mi], ir[mi + p], qr[mi], qr[mi + p]
        )
        mn, mnr, xmn, ymn, xmnr, ymnr = slope_midpoint(
            i[mi], i[mi - p], q[mi], q[mi - p], ir[mi], ir[mi - p], qr[mi], qr[mi - p]
        )
        mpi, mpir = -1 / mp, mpr / (mp**2)
        mni, mnir = -1 / mn, mnr / (mn**2)
        xc, yc, xcr, ycr = pointslope_intersect(
            mpi, xmp, ymp, mni, xmn, ymn, mpir, xmpr, ympr, mnir, xmnr, ymnr
        )
        cestimates[p - holdoff][0] = xc
        cestimates[p - holdoff][1] = yc
        cestimaters[p - holdoff][0] = xcr
        cestimaters[p - holdoff][1] = ycr

    if plot:
        plot.errorbar(
            cestimates[::, 0], cestimates[::, 1], cestimaters[::, 0], cestimaters[::, 1]
        )

    cestimate = np.sum(cestimates / (cestimaters**2), axis=0) / np.sum(
        1 / (cestimaters**2), axis=0
    )

    return (cestimate[0], cestimate[1]), np.sqrt(
        (cestimate[0] - i[mi]) ** 2 + (cestimate[1] - q[mi]) ** 2
    )


def circle_model(i, r, i0, q0):
    return np.sqrt(r**2 - (i - i0) ** 2) + q0


def find_truecenter(
    i,
    q,
    irms,
    qrms,
    center,
    radius,
    model=circle_model,
    plot=None,
):
    i_apcent = i - center[0]
    q_apcent = q - center[1]

    angles = np.arctan2(q_apcent, i_apcent)
    middle = (angles[0] + angles[-1]) / 2
    rotation = np.pi / 2 - middle

    i_apcentrot = i_apcent * np.cos(rotation) - q_apcent * np.sin(rotation)
    q_apcentrot = i_apcent * np.sin(rotation) + q_apcent * np.cos(rotation)

    if plot:
        plot.plot(i_apcentrot, q_apcentrot)
    popt, pcov = opt.curve_fit(
        model,
        i_apcentrot,
        q_apcentrot,
        sigma=np.sqrt(irms**2 + qrms**2),
        p0=(radius, 0, 0),
    )
    true_center = (
        center[0] + popt[1] * np.cos(-rotation) - popt[2] * np.sin(-rotation),
        center[1] + popt[1] * np.sin(-rotation) + popt[2] * np.cos(-rotation),
    )
    true_radius = popt[0]
    return true_center, true_radius


def quantize_rotation(rotation, bits=8):
    if rotation == np.pi:
        rotation = -np.pi
    return np.floor(rotation * (1 << (bits - 1)) / np.pi) * np.pi / (1 << (bits - 1))


def rotate_and_center(
    sweep: Sweep,
    targets: Optional[list[float] | nt.NDArray[np.float64]] = None,
    fit: Optional[Callable[..., nt.NDArray[np.float64]]] = None,
    program: Optional[ThreepartDDC] = None,
    plot: bool = False,
):
    if not targets:
        targets = np.zeros_like(sweep.iq.shape[0])

    phase_offsets = np.zeros(2048)
    ddc_centers = np.zeros(2048, dtype=np.complex64)

    iqs, iqrs = sweep.iq, sweep.iqsigma

    for t in range(iqs.shape[0]):
        i = iqs[t].real
        q = iqs[t].imag
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1)
            ax.set_aspect("equal")
            ax.set_title(str(sweep.config.waveform.default_ddc_config.tones[t]))
            ax.plot(i, q)

        ri = iqrs[t].real
        rq = iqrs[t].imag

        target = targets[t]

        approxc, approxr = approximate_center(i, q, ri, rq, plot=ax if plot else None)
        if fit:
            center, radius = find_truecenter(
                i, q, ri, rq, approxc, approxr, plot=ax if plot else None
            )
        else:
            center = approxc
            radius = approxr

        logging.getLogger(__name__).debug(
            "Tone {:.4f} @ I{:.3f}xQ{:.3f} R{:.3f}".format(
                sweep.config.waveform.default_ddc_config.tones[t],
                center[0],
                center[1],
                radius,
            )
        )

        if plot:
            ax.add_patch(plt.Circle(approxc, approxr, color="b", fill=False))
            ax.add_patch(plt.Circle(center, radius, color="r", fill=False))
            ax.scatter(*center)

        current_rotation = np.arctan2(
            q[q.size // 2] - center[1], i[i.size // 2] - center[0]
        )
        rotation = quantize_rotation(target - current_rotation)
        rotated_center = (
            np.cos(rotation) * center[0] - np.sin(rotation) * center[1],
            np.sin(rotation) * center[0] + np.cos(rotation) * center[1],
        )

        iq_offset = (rotated_center[0], rotated_center[1])

        phase_offsets[t] = rotation
        rotation = ((rotation + np.pi) % (2 * np.pi)) - np.pi
        ddc_centers[t] = rotated_center[0] + rotated_center[1] * 1j
        if plot:
            plt.show()

    ddc_centers /= 2**15

    ddc_config = DDCConfig(
        tones=sweep.config.waveform.default_ddc_config.tones,
        phase_offset=phase_offsets,
        loop_center=ddc_centers,
    )
    if program:
        program.configure(**ddc_config.settings_dict())
    return ddc_config
