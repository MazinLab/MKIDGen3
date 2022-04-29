from pynq import allocate, DefaultIP
import numpy as np


def opfb_bin_number(freq, ssr_raw_order=False):
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

    ssr_raw_order: 
        the OPFB bins are in order receviced from the Xilinx SSR FFT with no shift applied
        bin 0 contains frequencies in the interval [-1, 1) MHz, bin 1 [0, 2) MHz...
        bin 2047 contains [2046, 2048) MHz.
        The nyquist frequency component in bin 2048 is common to both positive
        and negative frequencies. By convention it is the highest negative frequency so
        bin 2048 contains the highest negative frequency [2047, -2047) MHz
        bins 2049 + contain negative frequencies progressing towards 0 MHz.

    """
    if ssr_raw_order:
        return (np.round(freq / 1e6).astype(int) + 4096) % 4096
    else:
        return np.round(freq / 1e6).astype(int) + 2048


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


class BinToResIP(DefaultIP):
    resmap_addr = 0x1000
    bindto = ['MazinLab:mkidgen3:bin_to_res:0.6', 'mazinlab:mkidgen3:bin_to_res:1.33']

    def __init__(self, description):
        """
        The core uses an array of 256 values, each consisting of 8 12 bit numbers packed into 96 bit word that
        specifies the OPFB bin number (0-4095, hence 12bit) used by each of the resonator channels in a group of 8.
        So to drive resonator channel 1035 from OPFB bin 3043 you need to set bits 35:24 (output 3) of group 129 to
        3043.

        control
        0x1000 ~
        0x1fff : Memory 'rid_to_bin' (256 * 128b)
                 Word 4n   : bit [31:0] - rid_to_bin[n][31: 0]
                 Word 4n+1 : bit [31:0] - rid_to_bin[n][63:32]
                 Word 4n+2 : bit [31:0] - rid_to_bin[n][95:64]
                 Word 4n+3 : bit [31:0] - rid_to_bin[n][127:96]  (unused)
        """
        super().__init__(description=description)


    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def _read_group(self, group_ndx):
        self._checkgroup(group_ndx)
        g = 0
        vals = [self.read(self.resmap_addr + 16 * group_ndx + 4 * i) for i in range(3)]
        for i, v in enumerate(vals):
            # print(format(v,'032b'))
            g |= v << (32 * i)
        # print('H-'+format(g,'096b')+'-L')
        return [((g >> (12 * j)) & 0xfff) for j in range(8)]

    def _write_group(self, group_ndx, group):
        self._checkgroup(group_ndx)
        if len(group) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        for i, g in enumerate(group):
            bits |= (int(g) & 0xfff) << (12 * i)
        data = bits.to_bytes(12, 'little', signed=False)
        self.write(self.resmap_addr + 16 * group_ndx, data)

    def bin(self, res):
        """
        Retrieve the OPFB bin assigned to resonator channel res, 0-2047
        The mapping for resonator i is 12 bits and will require reading 1 or 2 32 bit word
        n=i//8 j=(i%8)*12//32
        """
        return self._read_group(res // 8)[res % 8]

    @property
    def bins(self):
        """ Return a tuple of the OPFB bins assigned to each of the 2048 resonator channels"""
        return tuple([v for g in range(256) for v in self._read_group(g)])

    @bins.setter
    def bins(self, bins):
        """ Set which OPFB bins to use for each resonator channel. Set to list of 2048 numbers in [0, 4095]"""
        if len(bins) != 2048:
            raise ValueError('len(bins)!=2048')
        if min(bins) < 0 or max(bins) > 4095:
            raise ValueError('Bin values must be in [0,4095]')
        for i in range(256):
            self._write_group(i, bins[i * 8:i * 8 + 8])
