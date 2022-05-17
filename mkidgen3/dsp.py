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


"""
FFT bin ordering is confusing.

We find it more convenient to work with bins in order of increasing central frequency and call this
default or native order. Though "natural" may be tempting it is overloaded as the bit order of the FFT 
algorithm is often referred to as either "bit-reversed" or "natural" order for word outputs. Our FFT outputs 
words in the natural order. 

The SSR FFT used in the OPFB produces bins in a shifted order: 
    [-1, 1), ..., [2046, 2048), [2047, -2047), ..., [-2, -1)
this on-fpga order that is ingested by bin_to_res we refer to as the raw ssr order.
"""


def opfb_bin_number(freq, ssr_raw_order=True):
    """
    Compute the OPFB bin number corresponding to each frequency, specified in Hz.

    Frequencies are assumed to be in [-2.048, 2.048) GHz. Frequencies are placed in
    the closest bin by central frequency. Bins are 2 MHz wide and centered on 1 MHz
    increments (50% overlapping).

    bins are indexed 0 to 4095 from left to right across the 4 GHz IQ spectrum.
    bin 0 is centered at -2048 MHz and contains [2047, -2047) MHz.
    bin 1 contains [-2048, -2046) MHz.
    bin 2048 contains [-1, 1) MHz.
    bin 4095 contains [2046, 2048) MHz.

    ssr_raw_order: bool
        If true return the bin number in the order of the OPFB output.
        If false return
        the OPFB bins are in order received from the Xilinx SSR FFT with no shift applied
        bin 0 contains frequencies in the interval [-1, 1) MHz, bin 1 [0, 2) MHz...
        bin 2047 contains [2046, 2048) MHz.
        The nyquist frequency component in bin 2048 is common to both positive
        and negative frequencies. By convention, it is the highest negative frequency so
        bin 2048 contains the highest negative frequency [2047, -2047) MHz
        bins 2049 + contain negative frequencies progressing towards 0 MHz.

    """
    if ssr_raw_order:
        return (np.round(freq / 1e6).astype(int) + 4096) % 4096
    else:
        return np.round(freq / 1e6).astype(int) + 2048


def opfb_bin_frequencies(bins, resolution, Fs=4.096e9, M=4096, OS=2, left_snip=1):
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
    # TODO Don't compute all the centers only to subscript
    try:
        len(bins)
    except TypeError:
        bins = [bins]
    bins = np.asarray(bins).astype(int)
    bin_centers = (Fs / M) * np.linspace(-M / 2, M / 2 - 1, M)
    bin_width = (Fs / M) * OS
    base_freq = np.linspace(-bin_width / 2, bin_width / 2 - 1, resolution - left_snip)
    return base_freq[:, None] + bin_centers[bins]


def opfb_bin_center(bins, Fs=4.096e9, M=4096):
    try:
        len(bins)
    except TypeError:
        bins = [bins]
    bins = np.asarray(bins).astype(int)
    return (Fs / M) * np.linspace(-M / 2, M / 2 - 1, M)[bins]  # TODO Don't compute all the centers only to subscript
