import numpy as np


def quantize_frequencies(x, rate=4.096e9, n_samples=2 ** 19):
    x = np.asarray(x) if not isinstance(x, (int, float)) else x
    freq_res = rate / n_samples
    return np.round(x / freq_res) * freq_res


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
        x -= np.max(x)
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
    try:
        len(bins)
    except TypeError:
        bins = [bins]
    bins = np.asarray(bins).astype(int)
    bin_centers = (Fs / M) * np.linspace(-M / 2, M / 2 - 1, M)
    bin_width = (Fs / M) * OS
    base_freq = np.linspace(-bin_width / 2, bin_width / 2 - 1, resolution - left_snip)
    return base_freq[:, None] + bin_centers[bins]


def opfb_bin_center(bins, Fs=4.096e9, M=4096, ssr_order=True):
    try:
        len(bins)
    except TypeError:
        bins = [bins]
    bins = np.asarray(bins).astype(int)
    f=(Fs / M) * np.linspace(-M / 2, M / 2 - 1, M)
    return np.fft.fftshift(f)[bins] if ssr_order else f[bins]


def do_fixed_point_pfb(fpcomb, fpcoeff, n_convert=None, truncate=True):
    """Set truncate to false to preserve the full output bitwidth. Truncation is done with FpBinary defaults."""
    n_total_packets = fpcomb.size // 2048 // 2 - 16 if n_convert is None else n_convert
    fft_block = np.zeros((n_total_packets+1, 256, 16), dtype=np.complex64)
    for i in range(0, n_total_packets, 2):  # each packet of ADC samples, 128 new things to a lane 2 packets to feed all channels
        lane_out = np.zeros((2, 256, 16), dtype=np.complex64)
        for l in range(16):
            fresh = np.array([fpcoeff[l, :, 7 - c_i] * fpcomb[i + 2 * c_i:i + 2 * c_i + 2, l::16, :].reshape(256, 2).T
                              for c_i in range(8)]).sum(axis=0)
            delay = np.roll(np.array(
                [fpcoeff[l, :, 7 - c_i] * fpcomb[1 + i + 2 * c_i:1 + i + 2 * c_i + 2, l::16, :].reshape(256, 2).T
                 for c_i in range(8)]).sum(axis=0), 128, axis=1)
            # Sum the multiplies are roll the delayed samples
            if truncate:
                outformat = (-9, sum(fpcomb.flat[0].format) + 9)
                conv = lambda a: np.array(list(map(lambda x: float(x.resize(outformat)), a)))
                lane_out[0, :, l] = conv(fresh[0]) + conv(fresh[1]) * 1j
                lane_out[1, :, l] = conv(delay[0]) + conv(delay[1]) * 1j
            else:
                lane_out[0, :, l] = fresh[0].astype(float) + fresh[1].astype(float) * 1j
                lane_out[1, :, l] = delay[0].astype(float) + delay[1].astype(float) * 1j
        fft_block[i] = lane_out[0]
        fft_block[i + 1] = lane_out[1]
    return fft_block
