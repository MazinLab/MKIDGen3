from logging import getLogger
import numpy as np
import time
try:
    import pynq
    from .drivers import *
except ImportError:
    getLogger(__name__).info('pynq not available, functionality will be limited.')


_gen3_overlay, _mig_overlay, _frequencies = [None]*3


def set_lo_freq(lo_ghz):
    import requests
    r = requests.get(f'http://skynet.physics.ucsb.edu:51111/loset/{lo_ghz}')
    return r.json()


def iqcapture(n):
    """ Capture and return n samples for the currently configured frequencies"""
    if _frequencies is None:
        return None

    _gen3_overlay.capture.axis_switch_0.set_master(1, commit=True)  # After the FIR

    #4 bytes per IQ * n * np.ceil(_frequencies.size/8)
    buf = pynq.allocate((2*n*int(np.ceil(_frequencies.size/8)), 2), 'i2', target=_mig_overlay.MIG0)  # 2**30 is 4 GiB

    captime = _gen3_overlay.capture.iq_capture_0.capture(n, groups=np.arange(np.ceil(_frequencies.size/8), dtype=int),
                                                         device_addr=buf.device_address, start=False)
    #TODO we need to wait for the core to signal done
    time.sleep(captime*2)

    # buf now has IQ data in the 16_15 format IQ0_0 IQ1_0 ... IQnfreq*8-1_0  IQ0_1......  IQnfreq*8-1_n-1

    return list(buf)


def set_waveform(freq, amplitudes=None, attenuations=None, simple=False, **kwarg):
    """ freq in Hz amplitude 0-1"""
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
        dactable={'comb':comb}
    else:
        if attenuations is not None:
            dactable = daccomb(frequencies=freq, n_samples=n_samples, attenuations=attenuations,
                            sample_rate=sample_rate, return_full=True)
            comb=dactable['comb']
        else:
            if amplitudes is None:
                amplitudes = np.ones_like(freq)

            dactable = generate_dac_comb(frequencies=freq, n_samples=n_samples, sample_rate=sample_rate,
                                         amplitudes=amplitudes)
            comb = dactable['iq']

    print(f"Comb shape: {comb.shape}. \nTotal Samples: {comb.size}. Memory: {comb.size * 4 / 1024 ** 2:.0f} MB\n")
    _gen3_overlay.dac_table_axim_0.stop()
    _gen3_overlay.dac_table_axim_0.replay(comb, **kwarg)
    return dactable


def opfb_bin_number(freq):
    """Compute the OPFB bin number corresponding to each frequency, specified in Hz"""
    # if opfb bins werent shifted then it would be: strt_bins=np.round(freq/1e6).astype(int)+2048
    return ((np.round(freq / 1e6).astype(int) + 2048) + 2048) % 4096


def set_channels(freq):
    """
    Set each resonator channel to use the correct OPFB bin given the 2048 frequencies (in Hz) for each channel.
    """
    bins = np.arange(2048, dtype=int)
    bins[:freq.size] = opfb_bin_number(freq)
    _gen3_overlay.photon_pipe.reschan.bin_to_res.bins = bins


def tone_increments(freq):
    """Compute the DDS tone increment for each frequency (in Hz), assumes optimally selected OPFB bin when computing
    central frequency"""

    f_center = np.fft.fftfreq(4096, d=1 / 4.096e9)
    shft_bins = opfb_bin_number(freq)

    # This must be 2MHz NOT 2.048MHz, the sign matters! Use 1MHz as that corresponds to ±PI
    return (freq - f_center[shft_bins]) / 1e6


def set_tones(freq):
    tones = np.zeros((2, 2048))
    tones[0, :min(freq.size, 2048)] = tone_increments(freq)
    tones[1, :] = np.zeros(2048)
    print('Writing tones...')  # The core expects normalized increments
    _gen3_overlay.photon_pipe.reschan.resonator_ddc.tones = tones


def set_frequencies(freq, amplitudes=None):
    """ Set the bins and ddc values so that freq are in the associated resonator channels"""
    global _frequencies
    _frequencies = freq
    configure('/home/xilinx/jupyter_notebooks/Unit_Tests/Full_Channelizer/rst_rfdconly_axipc/gen3_512_iqsweep.bit',
              ignore_version=True)
    getLogger(__name__).info('setting waveform')
    set_waveform(freq, amplitudes=amplitudes)
    getLogger(__name__).info('setting resonator channels')
    set_channels(freq)
    getLogger(__name__).info('setting ddc tones')
    set_tones(freq)


def plstatus():
    from pynq import PL
    print(f"PL Bitfile: {PL.bitfile_name}\nPL Timestamp: {PL.timestamp}\n")


def configure(bitstream, mig=False, ignore_version=False, clocks=False, external_10mhz=False, download=True):
    import pynq

    global _gen3_overlay
    _gen3_overlay = pynq.Overlay(bitstream, ignore_version=ignore_version, download=download)

    if clocks:
        import mkidgen3.drivers.rfdc
        rfdc.patch_xrfclk_lmk()
        _gen3_overlay.rfdc.start_clocks(external_10mhz=external_10mhz)

    return _gen3_overlay
