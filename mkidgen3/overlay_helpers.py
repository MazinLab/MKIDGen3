from logging import getLogger
import numpy as np
import time

import mkidgen3.clocking
import mkidgen3.opfb as dsp
from .drivers.ddc import tone_increments
from . import util

_gen3_overlay, _frequencies = [None] * 2


def set_waveform(freq, amplitudes=None, attenuations=None, simple=False, **kwarg):
    """
    Configure the DAC replay table to output a waveform containing the specified frequencies.

    frequencies are in Hz, amplitudes are from 0-1, attenuations are in dB

    Additional keyword arguments are passed on when calling the dac table drivers replay() method.

    Do this either simply, and directly within this function or via either dactable.daccomb or
    dactable.generate_dac_comb.

    Manual loading of a waveform can be attained by performing
    table = generate_dac_comb(frequencies=np.array([0.3e9]), n_samples=2**19, sample_rate=4.096e9,
                              amplitudes=np.array([1.0]))
    ol.dac_table.stop()
    ol.dac_table.replay(table['iq'], fpgen=lambda x: (x*2**15).astype(np.uint16))

    and looking at the contents of ol.dac_table._buffer which should match

    buf = np.zeros((2 ** 15, 2, 16), dtype=np.int16)
    buf[:, 0, :] = table['iq'].real.reshape((2 ** 15,16)) * 2**15
    buf[:, 1,: ] = table['iq'].imag.reshape((2 ** 15,16)) * 2**15

    in Hz amplitude 0-1"""
    from .daccomb import daccomb, generate_dac_comb
    n_samples = 2 ** 19
    sample_rate = 4.096e9
    n_res = 2048

    # my old ipython test channelizer code
    if simple:
        if amplitudes is None:
            amplitudes = np.ones_like(freq)
        t = 2 * np.pi * np.arange(n_samples) / sample_rate
        comb = np.zeros(n_samples, dtype=np.complex64)
        phases = np.zeros(n_res)
        for i in range(freq.size):
            comb += amplitudes[i] * np.exp(1j * (t * freq[i] + phases[i]))
        dactable = {'comb': comb}
    else:
        if attenuations is not None:
            dactable = daccomb(frequencies=freq, n_samples=n_samples, attenuations=attenuations,
                               sample_rate=sample_rate, return_full=True)
            comb = dactable['comb']
        else:
            if amplitudes is None:
                amplitudes = np.ones_like(freq)

            dactable = generate_dac_comb(frequencies=freq, n_samples=n_samples, sample_rate=sample_rate,
                                         amplitudes=amplitudes)
            comb = dactable['iq']

    getLogger(__name__).debug(f"Comb shape: {comb.shape}. \n"
                              f"Total Samples: {comb.size}. Memory: {comb.size * 4 / 1024 ** 2:.0f} MB\n")
    play_waveform(comb, **kwarg)
    return dactable


def play_waveform(iq, **kwargs):
    _gen3_overlay.dac_table.replay(iq, stop_if_needed=True, **kwargs)


def set_channels(freq):
    """
    Set each resonator channel to use the correct OPFB bin given the 2048 frequencies (in Hz) for each channel.

    Only the first 2048 frequencies will be used. Channels are assigned in the order frequencies are specified.
    If fewer than 2048 frequencies are specified no assumptions may be made about which OPFB bin number(s) drive the
    remaining channels, however they may be determined by inspecting the .bins property of bin_to_res.
    """
    freq = freq[:2048]
    bins = np.arange(2048, dtype=int)
    bins[:freq.size] = dsp.opfb_bin_number(freq)
    _gen3_overlay.photon_pipe.reschan.bin_to_res.bins = bins


def iq_find_phase(n_points=1024):
    """A helper function to capture data just after bin2res and after reschan. NEEDS WORK"""
    x = _gen3_overlay.capture.capture_iq(n_points, 'all', tap_location='iq')
    iq = np.array(x)
    x.freebuffer()
    iq = iq[..., 0] + iq[..., 1] * 1j

    phase = np.angle(iq)
    return -phase.mean(0) / (2 * np.pi), iq


def set_frequencies(freq, **kwargs):
    """ Set the DAC waveform, OPFB bins, and DDC tones so that frequencies being fed through the photon pipeline.

    The order of the frequencies is preserved (they are not sorted).

    kwargs is passed on to set_waveform.

    The device must be suitably configured.
    """
    global _frequencies
    _frequencies = freq
    getLogger(__name__).info('setting waveform')
    set_waveform(freq, **kwargs)
    getLogger(__name__).info('setting resonator channels')
    set_channels(freq)
    getLogger(__name__).info('setting ddc tones')
    configure_ddc(freq)


def enable_interrupts():
    _gen3_overlay.axi_intc_0.register_map.IER = -1
    _gen3_overlay.axi_intc_0.register_map.MER.HIE = True
    _gen3_overlay.axi_intc_0.register_map.MER.ME = True


def configure(bitstream, ignore_version=True, clocks=False, programming_key=False, download=True, mts=False):
    import pynq
    import mkidgen3

    if clocks:
        mkidgen3.clocking.start_clocks(programming_key=programming_key)
        time.sleep(0.5)  # allow clocks to stabilize before loading overlay

    global _gen3_overlay
    ol = _gen3_overlay = mkidgen3._gen3_overlay = pynq.Overlay(bitstream, ignore_version=ignore_version, download=download)
    getLogger(__name__).info(f"PL Bitfile: {pynq.PL.bitfile_name} ({ol.timestamp})  Loaded: {ol.is_loaded()}")

    mkidgen3.quirks.Overlay.post_configure()

    if mts:
        assert download, "Enable MTS after downloading the bitstream with ol.rfdc.enable_mts()"
        ol.rfdc.enable_mts()

    return _gen3_overlay

def capture_opfb(n=256, raw=False):
    """Capture the OPFB output, exercise caution with large n as the result is copied from PL to PS DDR4"""
    out = np.zeros((n, 4096, 2) if raw else (n, 4096), dtype=np.int16 if raw else np.complex64)
    _gen3_overlay.photon_pipe.reschan.bin_to_res.bins = range(0, 2048)
    x = _gen3_overlay.capture.capture_iq(n, 'all', tap_location='rawiq')
    if raw:
        out[:, :2048, :] = x
        x.freebuffer()
    else:
        out[:, :2048] = util.buf2complex(x, free=True)
    _gen3_overlay.photon_pipe.reschan.bin_to_res.bins = range(2048, 4096)

    x = _gen3_overlay.capture.capture_iq(n, 'all', tap_location='rawiq')
    if raw:
        out[:, 2048:, :] = x
        x.freebuffer()
    else:
        out[:, 2048:] = util.buf2complex(x, free=True)

    return out
