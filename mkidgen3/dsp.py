import numpy as np


def opfb_bin_spectrum(data, bins=None, norm_max=True, shift=True, left_snip=1, db=True):
    """
    Inputs:
    - bin: OPFB bin number (allowed values 0 - 4095)
    - data: Raw data out of OPFB. Should be in the form N x 4096 where N is the number of samples from a single bin.
    - norm_max: boolean. Normalize the spectrum to the max value (easier to read spur, etc. height)
    - shift: boolean. Either apply to every OPFB bin or don't (this should always be applied).
    - left_snip: Cuts out the left most sample in a bin. This is an annoying crutch to handle the
        fact we place the N/2 bin on the far left.
        See https://www.gaussianwaves.com/2015/11/interpreting-fft-results-complex-dft-frequency-bins-and-fftshift/
        for explanation.

    Output:
    - Returns spectrum of a single FFT bin.
    """
    bins = np.asarray(bins).astype(int)
    if db:
        x = 20 * np.log10(np.abs(np.fft.fft(data, axis=0)))
    if shift:
        x = np.fft.fftshift(x, axes=0)
    if norm_max:
        x-=np.max(x)
    return np.flip(x[left_snip:, bins], axis=0)


def opfb_bin_freq(bins, resolution, Fs=4.096e9, M=4096, OS=2, left_snip=1):
    """
    Inputs:
    - bin: OPFB bin (0-4095). bin 0 contains -2048 to -2046 MHz, bin 4095 contains 2045 to 2047 MHz.
    - resolution: the number of samples from a single bin to take the FFT of. This dictates the frequency
        resolution in a single OPFB bin.
    - Fs: ADC Sampling Rate.
    - M: OPFB FFT Size.
    - OS: Oversample ratio.
    - left_snip: Cuts out the left most sample in a bin. This is an annoying crutch to handle the
        fact we place the N/2 bin on the far left.
        See https://www.gaussianwaves.com/2015/11/interpreting-fft-results-complex-dft-frequency-bins-and-fftshift/
        for explanation.

    Outputs:
    - Returns an array of frequency values in Hz for the given OPFB bin. """
    try:
        len(bins)
    except TypeError:
        bins = [bins]
    bins = np.asarray(bins).astype(int)
    bin_centers = (Fs / M) * np.linspace(-M / 2, M / 2 - 1, M)
    bin_width = (Fs / M) * OS
    base_freq = np.linspace(-bin_width / 2, bin_width / 2 - 1, resolution - left_snip)
    return base_freq[:, None] + bin_centers[bins]
