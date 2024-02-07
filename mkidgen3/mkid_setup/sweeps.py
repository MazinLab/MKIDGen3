import time

import numpy as np
import numpy.typing as nt

from typing import Optional, Type
from dataclasses import dataclass

from mkidgen3.server.feedline_config import WaveformConfig

LO_QUANT = 0.596


@dataclass
class SweepConfig:
    steps: nt.NDArray[np.float64]
    waveform: WaveformConfig
    lo_center: float | np.float64 = 6000.0
    average: int = 1024
    output_atten: Optional[float] = None
    input_atten: Optional[float] = None
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
        output_atten: Optional[float] = None,
        input_atten: Optional[float] = None,
    ) -> "SweepConfig":
        spacing = LO_QUANT * np.floor(((bandwidth / points) / LO_QUANT))

        # These are not going to be actually quantized properly because FP but the LO algo on the IF Board is FP too.
        # We will probably want to fix both of these at some point in the near future as the IF Board is FP32
        steps = (np.arange(points) * spacing - spacing * points // 2) / 1e6

        return SweepConfig(
            steps, waveform, lo_center, average, output_atten, input_atten
        )

    def reset_ddc(self, ddccontrol) -> "SweepConfig":
        ddccontrol.configure(**self.waveform.default_ddc_config.settings_dict())
        return self

    def run_sweep(self, ifboard, capture) -> "Sweep":
        tones = self.waveform.waveform.freqs
        iq = np.empty((tones.size, self.steps.size), np.complex64)
        rms = np.empty((tones.size, self.steps.size), np.complex64)
        if self.input_atten:
            ifboard.set_attens(input_attens=self.input_atten)
        if self.output_atten:
            ifboard.set_attens(output_attens=self.output_atten)
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
class CombSweepConfig(SweepConfig):
    def __post_init__(self):
        tones = self.waveform.waveform.freqs
        if not np.allclose(
            tones, tones[0] + np.arange(tones.size) * (tones[1] - tones[0])
        ):
            raise ValueError(
                "Waveform required to have evenly spaced comb in incresing frequency order"
            )

    @classmethod
    def from_comb(
        cls,
        points: int,
        overlap: int,
        waveform: WaveformConfig,
        lo_center: float | np.float64 = 6000.0,
        average: int = 1024,
        output_atten: Optional[float] = None,
        input_atten: Optional[float] = None,
    ) -> "SweepConfig":
        if overlap > points:
            raise ValueError("overlap must be less than points")

        tones = waveform.waveform.freqs

        steps = np.linspace(0, tones[1] - tones[0], points, endpoint=False) / 1e6
        steps_overlap = steps[:overlap] + (tones[1] - tones[0]) / 1e6
        steps = np.concatenate((steps, steps_overlap))

        return CombSweepConfig(
            steps, waveform, lo_center, average, output_atten, input_atten
        )

    def run_sweep(self, ifboard, capture) -> "CombSweep":
        s = super().run_sweep(ifboard, capture)
        return CombSweep(s.iq, s.iqsigma, self)


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
            line = ax.semilogy(
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
                        color=line[0].get_color(),
                        linestyle="--",
                    )
                else:
                    ax.axvline(
                        newtones[i] / 1e6 + self.config.lo_center,
                        color=line[0].get_color(),
                        linestyle="--",
                    )
        if stacked:
            ax.axvline(0, color="black", lw=0.1, label="LO")
        ax.legend()
        ax.set_ylabel("S21 (Magnitude)")
        ax.set_xlabel("Frequency (MHz)")

    def plot_loops(self, ax):
        for i in range(self.iq.shape[0]):
            ax.plot(self.iq[i].real, self.iq[i].imag)
        ax.set_aspect("equal")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")


@dataclass
class CombSweep(Sweep):
    config: "CombSweepConfig"
