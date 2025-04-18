import time
import pickle

import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Any


from mkidgen3.server.waveform import SimpleFreqlistWaveform
from mkidgen3.server.feedline_config import WaveformConfig
from mkidgen3.mkid_setup.sweeps import (
    SweepConfig,
    PowerSweepConfig,
    PowerSweep,
    Sweep,
)


def find_maxiq(freqs, iqs, start):
    i = start
    while np.abs(iqs[i - 1] - iqs[i]) > np.abs(iqs[i + 1] - iqs[i]) or np.abs(
        iqs[i - 1]
    ) < np.abs(iqs[i]):
        i -= 1
    return freqs[i]


def agc(ifb, capture, output_atten, target_fraction=0.2, step=1, input_atten=62):
    while input_atten > 0:
        ifb.set_attens(output_attens=output_atten, input_attens=input_atten)
        d = capture.capture_adc(2**19, complex=False)
        m = np.max(np.abs(d))
        del d
        if m / 32764.0 > target_fraction:
            ifb.set_attens(output_attens=output_atten, input_attens=input_atten + step)
            return input_atten + step
        input_atten -= step
    return 0.0


def fft_agc(overlay, target_fraction=0.2, caplen=4096):
    options = [
        0xFFF,
        0xF7F,
        0x77F,
        0x777,
        0x757,
        0x755,
        0x555,
        0x515,
        0x115,
        0x111,
        0x101,
        0x001,
        0x000,
    ]
    gpio = overlay.photon_pipe.opfb.fft.axi_gpio_0
    for i in range(1, len(options)):
        gpio.channel1.write(options[i], 0xFFF)
        light = overlay.capture.capture_iq(caplen, tap_location="ddciq")
        blight = np.copy(light).flatten()
        del light
        if np.max(np.abs(blight)) / (1 << 15) > target_fraction:
            gpio.channel1.write(options[i - 1], 0xFFF)
            return options[i - 1]
    gpio.channel1.write(options[-1], 0xFFF)
    return options[-1]


@dataclass
class BinConfig:
    fftbin: np.int16
    tone: np.float64
    phase_offset: np.float64
    loop_center: np.complex64


@dataclass
class IQCapture:
    filename: str
    savetime: float
    captime: tuple[int, int, int]
    channel_map: Optional[dict[int, int]]
    usermeta: Any

    def load_channel(self, base, channel, sl=None):
        if self.channel_map:
            channel = self.channel_map[channel]
        f = np.load(base + self.filename)
        if sl:
            return f[sl, channel, 0] + 1.0j * f[sl, channel, 1]
        return f[::, channel, 0] + 1.0j * f[::, channel, 1]


@dataclass
class MKIDSetup:
    name: str
    attenuation: tuple[float, float]
    waveform: SimpleFreqlistWaveform
    bin_settings: list[list[BinConfig]]
    darks: list[list[IQCapture]] = field(default_factory=list)
    lights: list[dict[float | str, list[IQCapture]]] = field(default_factory=list)
    powersweep: Optional[PowerSweep] = None
    biassweep: Optional[Sweep] = None
    centeringsweeps: list[Sweep] = field(default_factory=list)
    centeredsweeps: list[Sweep] = field(default_factory=list)
    lo: float = 6000.0
    fftscale: int = 0xFFF
    tied_tones: dict[int, list[int]] = field(default_factory=dict)
    basedir: str = "./"

    def load(self, overlay, ifboard, *, recenter: bool = False):
        ol, ifb = overlay, ifboard

        ifb.set_lo(self.lo)
        ifb.set_attens(*self.attenuation)
        gpio = overlay.photon_pipe.opfb.fft.axi_gpio_0
        gpio.channel1.write(self.fftscale, 0xFFF)

        waveform = WaveformConfig(waveform=self.waveform)
        if self.bin_settings and not recenter:
            bins = np.array([b.fftbin for b in self.bin_settings[-1]])
            btones = np.array([b.tone for b in self.bin_settings[-1]])
            bphase = np.array([b.phase_offset for b in self.bin_settings[-1]])
            bcent = np.array([b.loop_center for b in self.bin_settings[-1]])
        else:
            bins = waveform.default_channel_config.bins
            btones = waveform.default_ddc_config.settings_dict()["tones"]
            bphase = np.zeros_like(btones)
            bcent = np.zeros_like(btones, dtype=np.complex64)

        print(btones)

        ol.dac_table.configure(**waveform.settings_dict())
        ol.photon_pipe.reschan.bin_to_res.configure(**{"bins": bins})
        ol.photon_pipe.reschan.ddccontrol_0.configure(
            **{"tones": btones, "phase_offset": -bphase, "loop_center": bcent}
        )

        if recenter:
            from mkidgen3.mkid_setup.loop_locator import rotate_and_center

            print("Centering...")
            centeringsweep = SweepConfig.from_bandwidth(
                bandwidth=0.25e6,
                points=256,
                average=1024 * 64,
                waveform=waveform,
                lo_center=self.lo,
                attens=self.attenuation,
            ).run_sweep(ifboard=ifb, capture=ol.capture)

            config, _, _ = rotate_and_center(
                centeringsweep, targets=list(np.zeros(len(self.waveform.quant_freqs)))
            )
            bphase = config.settings_dict()["phase_offset"][: len(bphase)]
            bcent = config.settings_dict()["loop_center"][: len(bphase)]
            ol.photon_pipe.reschan.ddccontrol_0.configure(
                **{"tones": btones, "phase_offset": -bphase, "loop_center": bcent}
            )

            centeredsweep = SweepConfig.from_bandwidth(
                bandwidth=0.25e6,
                points=256,
                average=1024 * 64,
                waveform=waveform,
                lo_center=self.lo,
                attens=self.attenuation,
            ).run_sweep(ifboard=ifb, capture=ol.capture)

            bin_setting = [
                BinConfig(bins[i], btones[i], bphase[i], bcent[i])
                for i in range(len(bins))
            ]

            self.bin_settings.append(bin_setting)
            self.centeringsweeps.append(centeringsweep)
            self.centeredsweeps.append(centeredsweep)
            self.darks.append([])
            self.lights.append({})

    def take_snap(
        self, name, overlay, first_sixteen=False, metacallback=lambda: None, groups=None
    ):
        #           Mib  Kib  b
        memsize = 4 * 1024 * 1024 * 1024
        sample_size = 16 * 2 * 2 if first_sixteen or groups else 16 * 2 * 2048
        count = memsize // (sample_size * 2)

        if first_sixteen:
            groups = [0, 1]

        data = overlay.capture.capture_iq(count, groups=groups, tap_location="ddciq")
        captime = (
            overlay.capture.axis2mm.mmio.read(0b01 << 2),
            overlay.capture.axis2mm.mmio.read(0b10 << 2),
            overlay.capture.axis2mm.mmio.read(0b11 << 2),
        )
        savetime = time.time()
        fn = "{:s}-{:s}-{:f}.npy".format(self.name, name, savetime)
        if first_sixteen:
            np.save(self.basedir + fn, data[::, :16, ::])
        else:
            np.save(self.basedir + fn, data[::, : len(self.waveform.quant_freqs), ::])
        data.freebuffer()
        del data

        channel_map = None
        if groups:
            channel_map = {}
            for i in range(16):
                channel_map[groups[0] * 16 + i] = i

        return IQCapture(fn, savetime, captime, channel_map, metacallback()), count / 2e6

    def take_darks(
        self,
        overlay,
        *,
        count=None,
        time=None,
        first_sixteen=False,
        metacallback=lambda: None,
        groups=None,
    ):
        import tqdm.autonotebook as tqdm

        assert (count is None) or (time is None)
        assert (count is not None) or (time is not None)
        if count:
            for i in tqdm.tqdm(range(count), desc="Darks"):
                sn, t = self.take_snap(
                    "dark",
                    overlay,
                    first_sixteen=first_sixteen,
                    metacallback=metacallback,
                    groups=groups,
                )
                self.darks[-1].append(sn)
        else:
            total = 0
            with tqdm.tqdm(total=time, desc="Darks") as pbar:
                while total < time:
                    sn, t = self.take_snap(
                        "dark",
                        overlay,
                        first_sixteen=first_sixteen,
                        groups = groups,
                        metacallback=metacallback,
                    )
                    self.darks[-1].append(sn)
                    total += t
                    pbar.update(t)

    def take_lights(
        self,
        overlay,
        source,
        *,
        count=None,
        time=None,
        first_sixteen=False,
        metacallback=lambda: None,
        groups=None,
    ):
        import tqdm.autonotebook as tqdm

        assert (count is None) or (time is None)
        assert (count is not None) or (time is not None)
        if source not in self.lights[-1].keys():
            self.lights[-1][source] = []
        if count:
            for i in tqdm.tqdm(range(count), desc="Lights"):
                sn, t = self.take_snap(
                    "light",
                    overlay,
                    first_sixteen=first_sixteen,
                    metacallback=metacallback,
                    groups=groups,
                )
                self.lights[-1][source].append(sn)
        else:
            total = 0
            with tqdm.tqdm(total=time, desc="Lights") as pbar:
                while total < time:
                    sn, t = self.take_snap(
                        "light",
                        overlay,
                        first_sixteen=first_sixteen,
                        metacallback=metacallback,
                        groups=groups,
                    )
                    self.lights[-1][source].append(sn)
                    total += t
                    pbar.update(t)

    def save(self, *, savetime=True):
        pickle.dump(
            self,
            open(
                self.basedir
                + self.name
                + (".pkl" if not savetime else "{:f}.pkl".format(time.time())),
                "wb",
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    @classmethod
    def with_selected(cls, name, overlay, ifboard, starting_tones, starting_amps, output_atten, selected_attens, *, maxvel=False, lo=6000.0, target_dynrange=0.25, tied_tones={}):
        import tqdm.autonotebook as tqdm

        from mkidgen3.mkid_setup.loop_locator import rotate_and_center

        baseline = output_atten

        ol, ifb = overlay, ifboard
        spacing = np.fft.fftfreq(1 << 19, 1 / 4.096e9)[1]

        tones = np.array(starting_tones)
        amps = np.array(starting_amps)

        attens = np.array(selected_attens)
        amps = amps * (10 ** (-(attens - baseline) / 20))

        print("Configuring DAC...")

        waveform = WaveformConfig(
            waveform=SimpleFreqlistWaveform(frequencies=tones, amplitudes=amps)
        )
        ol.dac_table.configure(**waveform.settings_dict())
        chan = waveform.default_channel_config
        ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
        ddc = waveform.default_ddc_config
        ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

        print("Running AGC...")
        ifb.set_lo(lo + 5)
        atten = (baseline, agc(ifb, ol.capture, baseline, target_dynrange, 0.25))
        fftsetting = fft_agc(ol, target_dynrange)
        ifb.set_lo(lo)

        if maxvel:
            print("Finding MAX IQ Velocity...")
            sweepmaxvel = SweepConfig(
                np.arange(-128, 16) * spacing / 1e6,
                waveform,
                lo,
                attens=atten,
                average=1024 * 64,
            ).run_sweep(ifboard=ifb, capture=ol.capture)
            start = np.argmin(np.abs(sweepmaxvel.config.steps))
            newtones = np.array(
                [
                    find_maxiq(sweepmaxvel.frequencies[i, ::], sweepmaxvel.iq[i, ::], start)
                    - lo * 1e6
                    for i in range(sweepmaxvel.frequencies.shape[0])
                ]
            )

            for main, tieds in tied_tones.items():
                newtones[tieds] = tones[tieds] + (newtones[main] - tones[main])
            tones = newtones

            waveform = WaveformConfig(
                waveform=SimpleFreqlistWaveform(frequencies=tones, amplitudes=amps)
            )
            ol.dac_table.configure(**waveform.settings_dict())
            chan = waveform.default_channel_config
            ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
            ddc = waveform.default_ddc_config
            ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

            print("Rerunning AGC...")
            ifb.set_lo(lo + 5)
            atten = (baseline, agc(ifb, ol.capture, baseline, target_dynrange, 0.25))
            fftsetting = fft_agc(ol, target_dynrange)
            ifb.set_lo(lo)
        else:
            sweepmaxvel = None

        setup = MKIDSetup(
            name,
            atten,
            waveform.waveform,
            [],
            powersweep=None,
            biassweep=sweepmaxvel,
            fftscale=fftsetting,
            tied_tones=tied_tones,
            lo=lo,
        )
        setup.load(ol, ifb, recenter=True)
        return setup


    @classmethod
    def interactive_setup(
        cls,
        name,
        overlay,
        ifboard,
        starting_tones,
        starting_amps,
        output_attens,
        *,
        lo=6000.0,
        target_dynrange=0.25,
        tied_tones={},
    ):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        import tqdm.autonotebook as tqdm
        from IPython.display import display
        from jupyter_ui_poll import ui_events

        from mkidgen3.mkid_setup.loop_locator import rotate_and_center

        ol, ifb = overlay, ifboard
        spacing = np.fft.fftfreq(1 << 19, 1 / 4.096e9)[1]

        tones = np.array(starting_tones)
        amps = np.array(starting_amps)
        waveform = WaveformConfig(
            waveform=SimpleFreqlistWaveform(frequencies=tones, amplitudes=amps)
        )
        ol.dac_table.configure(**waveform.settings_dict())
        chan = waveform.default_channel_config
        ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
        ddc = waveform.default_ddc_config
        ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

        fftscales = {}
        attens = {}
        ifb.set_lo(lo + 5)
        for atten in tqdm.tqdm(output_attens, desc="Running AGC..."):
            attens[atten] = agc(ifb, ol.capture, atten, target_dynrange, 0.25)
            fftscales[atten] = fft_agc(ol, target_dynrange)
        ifb.set_lo(lo)

        gpio = overlay.photon_pipe.opfb.fft.axi_gpio_0
        gpio.channel1.write(max(fftscales.values()), 0xFFF)

        sweep_base = SweepConfig(
            np.arange(-128, 64) * spacing / 1e6,
            waveform,
            lo,
            average=1024 * 16,
            rmses=False,
        )
        powersweep = PowerSweepConfig(
            attens=attens, sweep_config=sweep_base
        ).run_powersweep(ifb, ol.capture, progress=True)

        current_channel = 0
        current_atten = 0
        atten_options = np.array(list(powersweep.sweeps.keys()))
        atten_settings = np.zeros(len(tones), dtype=np.int64) - 1

        prev_res = widgets.Button(description="Prev")
        up = widgets.Button(description="Up")
        down = widgets.Button(description="Down")
        next_res = widgets.Button(description="Next")

        done = widgets.Button(description="DONE")

        cont = False

        def on_up(b):
            nonlocal current_atten
            nonlocal current_channel
            current_atten += 1
            if current_atten < 0:
                current_atten = 0
            if current_atten >= len(atten_options):
                current_atten = len(atten_options) - 1
            atten_settings[current_channel] = current_atten
            update_plot()

        def on_down(b):
            nonlocal current_atten
            nonlocal current_channel
            current_atten -= 1
            if current_atten < 0:
                current_atten = 0
            if current_atten >= len(atten_options):
                current_atten = len(atten_options) - 1
            atten_settings[current_channel] = current_atten
            update_plot()

        def on_next(b):
            nonlocal current_atten
            nonlocal current_channel
            current_channel += 1
            if current_channel < 0:
                current_channel = 0
            if current_channel >= len(tones):
                current_channel = len(tones) - 1
            current_atten = (
                atten_settings[current_channel]
                if atten_settings[current_channel] != -1
                else current_atten
            )
            atten_settings[current_channel] = current_atten
            update_plot()

        def on_prev(b):
            nonlocal current_atten
            nonlocal current_channel
            current_channel -= 1
            if current_channel < 0:
                current_channel = 0
            if current_channel >= len(tones):
                current_channel = len(tones) - 1
            current_atten = (
                atten_settings[current_channel]
                if atten_settings[current_channel] != -1
                else current_atten
            )
            atten_settings[current_channel] = current_atten
            update_plot()

        def on_done(b):
            nonlocal cont
            cont = True

        prev_res.on_click(on_prev)
        up.on_click(on_up)
        down.on_click(on_down)
        next_res.on_click(on_next)
        done.on_click(on_done)

        display(widgets.HBox([prev_res, up, down, next_res, done]))

        fig, (ax1, ax2) = plt.subplots(1, 2)

        def update_plot():
            nonlocal current_atten
            nonlocal current_channel
            ax1.clear()
            ax2.clear()
            ax1.set_title(atten_options[current_atten])
            ax2.set_title(current_channel)
            powersweep.sweeps[atten_options[current_atten]][1].plot(
                ax1,
                channels=slice(current_channel, current_channel + 1),
                newtones=list(tones),
            )
            powersweep.sweeps[atten_options[current_atten]][1].plot_loops(
                ax2,
                channels=slice(current_channel, current_channel + 1),
                newtones=list(tones),
            )
            fig.canvas.draw()

        update_plot()

        with ui_events() as poll:
            while not cont:
                poll(10)
        for main, tieds in tied_tones.items():
            atten_settings[tieds] = atten_settings[main]
        print("Selected attens:", atten_options[atten_settings])
        basesel = widgets.FloatText(
            value=atten_options[atten_settings[0]],
            step=0.25,
            description="Select Baseline Atten:",
            disabled=False,
        )
        done = widgets.Button(description="DONE")
        done.on_click(on_done)
        cont = False
        display(widgets.HBox([basesel, done]))
        with ui_events() as poll:
            while not cont:
                poll(10)
        baseline = basesel.value

        attens = atten_options[atten_settings]
        amps = amps * (10 ** (-(attens - baseline) / 20))

        print("Configuring DAC...")

        waveform = WaveformConfig(
            waveform=SimpleFreqlistWaveform(frequencies=tones, amplitudes=amps)
        )
        ol.dac_table.configure(**waveform.settings_dict())
        chan = waveform.default_channel_config
        ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
        ddc = waveform.default_ddc_config
        ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

        print("Rerunning AGC...")
        ifb.set_lo(lo + 5)
        atten = (baseline, agc(ifb, ol.capture, baseline, target_dynrange, 0.25))
        fftsetting = fft_agc(ol, target_dynrange)
        ifb.set_lo(lo)

        print("Finding MAX IQ Velocity...")
        sweepmaxvel = SweepConfig(
            np.arange(-128, 16) * spacing / 1e6,
            waveform,
            lo,
            attens=atten,
            average=1024 * 64,
        ).run_sweep(ifboard=ifb, capture=ol.capture)
        start = np.argmin(np.abs(sweepmaxvel.config.steps))
        newtones = np.array(
            [
                find_maxiq(sweepmaxvel.frequencies[i, ::], sweepmaxvel.iq[i, ::], start)
                - lo * 1e6
                for i in range(sweepmaxvel.frequencies.shape[0])
            ]
        )

        for main, tieds in tied_tones.items():
            newtones[tieds] = tones[tieds] + (newtones[main] - tones[main])
        tones = newtones

        waveform = WaveformConfig(
            waveform=SimpleFreqlistWaveform(frequencies=tones, amplitudes=amps)
        )
        ol.dac_table.configure(**waveform.settings_dict())
        chan = waveform.default_channel_config
        ol.photon_pipe.reschan.bin_to_res.configure(**chan.settings_dict())
        ddc = waveform.default_ddc_config
        ol.photon_pipe.reschan.ddccontrol_0.configure(**ddc.settings_dict())

        print("Rerunning AGC...")
        ifb.set_lo(lo + 5)
        atten = (baseline, agc(ifb, ol.capture, baseline, target_dynrange, 0.25))
        fftsetting = fft_agc(ol, target_dynrange)
        ifb.set_lo(lo)

        setup = MKIDSetup(
            name,
            atten,
            waveform.waveform,
            [],
            powersweep=powersweep,
            biassweep=sweepmaxvel,
            fftscale=fftsetting,
            tied_tones=tied_tones,
            lo=lo,
        )
        setup.load(ol, ifb, recenter=True)
        return setup

    @property
    def dark(self):
        for ds in self.darks[::-1]:
            if ds:
                return ds[-1]
        return None

    @property
    def by_source(self):
        sources = {}
        for ls in self.lights:
            for k, v in ls.items():
                if k not in sources:
                    sources[k] = []
                sources[k].extend(v)
        return sources
