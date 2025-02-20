import time
import copy

import numpy as np
import numpy.typing as nt

from typing import Optional, Type, NewType
from types import NoneType
from dataclasses import dataclass
from abc import ABC

from mkidgen3.server.feedline_config import WaveformConfig

LO_QUANT = 0.596

# TODO: Common types with powersweep_helpers
InputAtten = NewType("InputAtten", float)
OutputAtten = NewType("OutputAtten", float)

# Power Gain
Gain = NewType("Gain", float)
# Total system gain normalized to the total system gain with
# both adc and dac attens set to 0
AttenRefered = NewType("AttenRefered", Gain)


def attens_to_refered(attens: tuple[OutputAtten, InputAtten]) -> AttenRefered:
    return -attens[0] - attens[1]


# Assumes data is field quantity
def apply_gain(data: nt.NDArray[np.float64 | np.complex64], gain: Type[Gain]):
    return data * np.sqrt(10 ** (gain / 10))


@dataclass
class SweepConfig:
    steps: nt.NDArray[np.float64]
    waveform: WaveformConfig
    lo_center: float | np.float64 = 6000.0
    average: int = 1024
    attens: Optional[tuple[OutputAtten, InputAtten]] = None
    tap: str = "ddciq"
    rmses: bool = True

    _idle = 0.0
    _flush = 0

    @classmethod
    def from_bandwidth(
        cls,
        bandwidth: float | np.float64,
        points: int,
        waveform: WaveformConfig,
        lo_center: float | np.float64 = 6000.0,
        average: int = 1024,
        attens: Optional[tuple[OutputAtten, InputAtten]] = None,
    ) -> "SweepConfig":
        spacing = LO_QUANT * np.floor(((bandwidth / points) / LO_QUANT))

        # These are not going to be actually quantized properly because FP but the LO algo on the IF Board is FP too.
        # We will probably want to fix both of these at some point in the near future as the IF Board is FP32
        steps = (np.arange(points) * spacing - spacing * points // 2) / 1e6

        return SweepConfig(steps, waveform, lo_center, average, attens)

    def reset_ddc(self, ddccontrol) -> "SweepConfig":
        ddccontrol.reset_phase_center()
        return self

    def run_sweep(self, ifboard, capture, progress = True) -> "Sweep":
        tones = self.waveform.waveform.quant_freqs
        iq = np.empty((tones.size, self.steps.size), np.complex64)
        if self.rmses:
            rms = np.empty((tones.size, self.steps.size), np.complex64)
        else:
            rms = None
        if self.attens:
            ifboard.set_attens(
                input_attens=self.attens[1], output_attens=self.attens[0]
            )

        it = enumerate(self.steps)
        if progress:
            import tqdm.notebook as tqdm

        it = tqdm.tqdm(it, total=len(self.steps), desc="FREQ")
        for i, step in it:
            ifboard.set_lo(step + self.lo_center)
            piq, prms = self._get_iq_point_rms(capture)
            iq[::, i] = piq[: tones.size]
            if self.rmses:
                rms[::, i] = prms[: tones.size]
        ifboard.set_lo(self.lo_center)
        return Sweep(iq, (rms / np.sqrt(self.average)) if self.rmses else rms, self)

    def frequencies(self) -> nt.NDArray[np.float64]:
        tones = self.waveform.waveform.quant_freqs
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
        tones = self.waveform.waveform.quant_freqs
        if self._idle:
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
        if self.rmses:
            ps_buf -= means.reshape((1, tones.size, 2))
            ps_buf *= ps_buf
            rmses = np.sqrt(np.sum(ps_buf, axis=0) / np.float32(self.average))

        if hasattr(self, "verify"):
            means_golden = np.empty((tones.size, 2), dtype=np.float64)
            ps_buf_old.mean(axis=0, out=means_golden)
            assert np.allclose(means_golden, means)
            if rmses:
                rmses_golden = np.empty((tones.size, 2), dtype=np.float64)
                ps_buf_old.std(axis=0, out=rmses_golden)
                assert np.allclose(rmses_golden, rmses)

        if self.rmses:
            return means[::, 0] + 1j * means[::, 1], rmses[::, 0] + 1j * rmses[::, 1]
        return means[::, 0] + 1j * means[::, 1], None


@dataclass
class SmoothSweepConfig(SweepConfig):
    def run_sweep(self, ifboard, capture, progress = True, cache=None) -> "SmoothSweep":
        tones = self.waveform.waveform.quant_freqs
        iq = np.empty((tones.size, self.steps.size), np.complex64)
        if self.rmses:
            rms = np.empty((tones.size, self.steps.size), np.complex64)
        else:
            rms = None
        if self.attens:
            ifboard.set_attens(
                input_attens=self.attens[1], output_attens=self.attens[0]
            )

        if cache is None:
            cache = []
        if cache == []:
            for step in self.steps:
                ifboard.set_lo(self.lo_center + step)
                cache.append(ifboard.trf_control.get_certificate())

        it = enumerate(self.steps)
        if progress:
            import tqdm.notebook as tqdm

        it = tqdm.tqdm(it, total=len(self.steps), desc="FREQ")
        for i, step in it:
            ifboard.set_lo(cache[i])
            piq, prms = self._get_iq_point_rms(capture)
            iq[::, i] = piq[: tones.size]
            if self.rmses:
                rms[::, i] = prms[: tones.size]
        ifboard.set_lo(self.lo_center)
        return SmoothSweep(iq, (rms / np.sqrt(self.average)) if self.rmses else rms, self, certificates=cache)


@dataclass
class CombSweepConfig(SweepConfig):
    overlap: int = 1

    def __post_init__(self):
        tones = self.waveform.waveform.quant_freqs
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
        attens: Optional[tuple[OutputAtten, InputAtten]] = None,
    ) -> "SweepConfig":
        if overlap > points:
            raise ValueError("overlap must be less than points")
        if overlap < 1:
            raise ValueError("Overlap should be >= 1 to enable stitching")

        tones = waveform.waveform.quant_freqs

        steps = np.linspace(0, tones[1] - tones[0], points, endpoint=False) / 1e6
        steps_overlap = steps[:overlap] + (tones[1] - tones[0]) / 1e6
        steps = np.concatenate((steps, steps_overlap))

        return CombSweepConfig(
            steps, waveform, lo_center, average, attens, overlap=overlap
        )

    def run_sweep(self, ifboard, capture, progress=True) -> "CombSweep":
        s = super().run_sweep(ifboard, capture, progress)
        return CombSweep(s.iq, s.iqsigma, self)


@dataclass
class AbstractSweep(ABC):
    iq: nt.NDArray[np.complex64]
    iqsigma: nt.NDArray[np.complex64]


@dataclass
class Sweep(AbstractSweep):
    iq: nt.NDArray[np.complex64]
    iqsigma: Optional[nt.NDArray[np.complex64]]
    config: "SweepConfig"
    equalized_gain: Optional[AttenRefered] = None

    @property
    def frequencies(self) -> nt.NDArray[np.float64]:
        return self.config.frequencies()

    def plot(
        self,
        ax,
        stacked: bool = False,
        newtones: Optional[list[float] | nt.NDArray[np.float64]] = None,
        channels: Optional[slice] = None,
        label_tones: bool = False,
        power: bool = True,
        **kwargs,
    ):
        if channels is None:
            channels = slice(0, self.iq.shape[0])
        for i in range(self.iq[channels, :].shape[0]):
            line = ax.plot(
                (self.frequencies[channels][i] / 1e6 if not stacked else self.config.steps),
                np.abs(self.iq[channels][i]) if not power else 20*np.log10(np.abs(self.iq[channels][i])),
                label=(
                    "Tone: {:.3f} MHz".format(
                        self.config.waveform.default_ddc_config.tones[channels][i] / 1e6
                    )
                    if label_tones
                    else None
                ),
                **kwargs,
            )
            if newtones:
                if stacked:
                    ax.axvline(
                        newtones[channels][i] / 1e6
                        - self.config.waveform.default_ddc_config.tones[channels][i] / 1e6,
                        color=line[0].get_color(),
                        linestyle="--",
                    )
                else:
                    ax.axvline(
                        newtones[channels][i] / 1e6 + self.config.lo_center,
                        color=line[0].get_color(),
                        linestyle="--",
                    )
        if stacked:
            ax.axvline(0, color="black", lw=0.1, label="LO")
        if label_tones:
            ax.legend()
        ax.set_ylabel("S21 (Magnitude)" if not power else "S21 (Power [dB])")
        ax.set_xlabel("Frequency (MHz)")

    def plot_loops(self, ax, channels: Optional[slice] = None, newtones: Optional[list[float] | nt.NDArray[np.float64]] = None, **kwargs):
        if channels is None:
            channels = slice(0, self.iq.shape[0])
        for i in range(self.iq[channels, :].shape[0]):
            ax.plot(self.iq[channels][i].real, self.iq[channels][i].imag, "-o", **kwargs)
        if newtones is not None:
            newtones = np.array(newtones)
            freqs = self.frequencies - self.config.lo_center * 1e6
            for i, t in enumerate(newtones[channels]):
                v = np.argmin(np.abs(freqs[channels][i] - t))
                ax.scatter(self.iq[channels][i][v].real, self.iq[channels][i][v].imag, marker="*", c='r', zorder=len(newtones) + 1, s = 24)
        ax.set_aspect("equal")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")

    def equalize(self, refered_gain: AttenRefered):
        if self.equalized_gain is not None:
            current_gain = self.equalized_gain
        elif self.config.attens is not None:
            current_gain = attens_to_refered(self.config.attens)
        else:
            raise ValueError(
                "Neither the current equalized gain, nor the original gain is known"
            )
        applied_gain = refered_gain - current_gain
        return self.__class__(
            iq=apply_gain(self.iq, applied_gain),
            iqsigma=apply_gain(self.iqsigma, applied_gain),
            config=self.config,
            equalized_gain=refered_gain,
        )
@dataclass
class SmoothSweep(Sweep):
    certificates: Optional[list["TRFCalibrationCertificate"]] = None

@dataclass
class CombSweep(Sweep):
    config: "CombSweepConfig"

    def stitch(self) -> "StitchedSweep":
        freqs = self.frequencies
        iq = self.iq
        iqs = self.iqsigma

        iq_new = np.empty(
            (
                1,
                (iq.shape[1] - self.config.overlap) * iq.shape[0] + self.config.overlap,
            ),
            np.complex64,
        )
        iqs_new = np.empty_like(iq_new)
        freqs_new = np.empty_like(iq_new, np.float64)

        iq_new[0][: iq.shape[1]] = iq[0]
        iqs_new[0][: iq.shape[1]] = iqs[0]
        freqs_new[0][: iq.shape[1]] = freqs[0]
        for i in range(1, self.frequencies.shape[0]):
            low = slice(
                (iq.shape[1] - self.config.overlap) * i,
                (iq.shape[1] - self.config.overlap) * i + self.config.overlap,
            )
            high = slice(self.config.overlap)
            high_new = slice(
                (iq.shape[1] - self.config.overlap) * i + self.config.overlap,
                (iq.shape[1] - self.config.overlap) * (i + 1) + self.config.overlap,
            )
            assert np.all(freqs_new[0][low] == freqs[i][high])
            ratio = iq_new[0][low] / iq[i][high]
            print(np.mean(ratio), np.abs(np.mean(ratio)))
            ratiosigma = (
                np.sqrt(
                    (iqs_new[0][low] / iq_new[0][low]) ** 2
                    + (iqs[i][high] / iq[i][high]) ** 2
                )
                * ratio
            )
            ratio, ratiosigma = np.sum(ratio / (ratiosigma**2)) / np.sum(
                1 / (ratiosigma**2)
            ), 1 / np.sqrt(np.sum(1 / (ratiosigma**2)))
            iq_new[0][high_new] = iq[i][self.config.overlap :] * ratio
            iqs_new[0][high_new] = iq_new[0][high_new] * np.sqrt(
                (iqs[i][self.config.overlap :] / iq[i][self.config.overlap :]) ** 2
                + (ratiosigma / ratio) ** 2
            )
            freqs_new[0][high_new] = freqs[i][self.config.overlap :]
        return StitchedSweep(
            iq=iq_new,
            iqsigma=iqs_new,
            original=self,
            equalized_gain=self.equalized_gain,
            frequencies=freqs_new,
        )


@dataclass
class StitchedSweep(AbstractSweep):
    original: "CombSweep"
    frequencies: nt.NDArray[np.float64]
    equalized_gain: Optional[AttenRefered] = None

    def equalize(self, refered_gain: AttenRefered):
        if self.equalized_gain is not None:
            current_gain = self.equalized_gain
        elif self.original.config.attens is not None:
            current_gain = attens_to_refered(self.original.config.attens)
        else:
            raise ValueError(
                "Neither the current equalized gain, nor the original gain is known"
            )
        applied_gain = current_gain - refered_gain
        return self.__class__(
            iq=apply_gain(self.iq, applied_gain),
            iqsigma=apply_gain(self.iqsigma, applied_gain),
            original=self.original,
            equalized_gain=refered_gain,
            frequencies=self.frequencies,
        )


@dataclass
class PowerSweepConfig:
    attens: dict[OutputAtten, InputAtten]
    sweep_config: Type[SweepConfig]

    def __post_init__(self):
        # TODO: Check attens valid
        if self.sweep_config.attens is not None:
            raise ValueError(
                "Input and Output attens must be unset in the template config"
            )

    @classmethod
    def from_matched(
        cls,
        starting_output: OutputAtten,
        starting_input: InputAtten,
        output_step: OutputAtten,
        steps: int,
        sweep_config: Type[SweepConfig],
    ):
        attens = {}
        for i in range(steps):
            attens[starting_output + output_step * i] = min(
                max(starting_input - output_step * i, 0), 31.75 * 2
            )
        return PowerSweepConfig(attens, sweep_config)

    def run_powersweep(self, ifboard, capture, progress=False) -> "PowerSweep":
        sweeps = {}
        iter = self.attens.items()
        kwargs = {}
        if isinstance(self.sweep_config, SmoothSweepConfig):
            kwargs["cache"] = []
        if progress:
            import tqdm.notebook as tqdm

            iter = tqdm.tqdm(iter, total=len(list(self.attens.keys())), desc="ATTN")
        for output_atten, input_atten in iter:
            this_sweepconfig = copy.copy(self.sweep_config)
            this_sweepconfig.attens = (output_atten, input_atten)
            sweeps[output_atten] = (
                input_atten,
                this_sweepconfig.run_sweep(ifboard, capture, progress, **kwargs),
            )
        return PowerSweep(self, sweeps)


@dataclass
class PowerSweep:
    config: "PowerSweepConfig"
    sweeps: dict[OutputAtten, tuple[InputAtten, Type[Sweep]]]

    def plot(
        self,
        fig,
        ax,
        output_range: Optional[tuple[OutputAtten, OutputAtten]] = None,
        equalize: Optional[AttenRefered] = 0,
        power: bool = True,
        channels: Optional[slice] = None,
        cmap=None,
    ):
        if cmap is None:
            import matplotlib.pyplot as plt

            cmap = plt.cm.viridis_r
        if output_range is not None and output_range[0] > output_range[1]:
            output_range = (output_range[1], output_range[0])
        sweeps = (
            self.sweeps
            if output_range is None
            else {
                o: (i, s)
                for o, (i, s) in self.sweeps.items()
                if o >= output_range[0] and o <= output_range[1]
            }
        )
        sweeps = (
            sweeps
            if equalize is None
            else {o: (i, s.equalize(equalize)) for o, (i, s) in sweeps.items()}
        )
        output_min = (
            min(list(sweeps.keys())) if output_range is None else output_range[0]
        )
        output_max = (
            max(list(sweeps.keys())) if output_range is None else output_range[1]
        )
        import matplotlib.colors as cl
        import matplotlib.cm as cm

        norm = cl.Normalize(output_min, output_max)
        smap = cm.ScalarMappable(norm, cmap)
        fig.colorbar(smap, ax=ax, label="Output Attenuation (dB)")
        for output_atten, (_, sweep) in sweeps.items():
            sweep.plot(ax, channels = channels, color=cmap(norm(output_atten)))
