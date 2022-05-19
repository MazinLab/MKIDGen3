import numpy as np
from fpbinary import FpBinary
from pynq import DefaultIP
from mkidgen3.mkidpynq import fp_factory
from mkidgen3.dsp import opfb_bin_number, opfb_bin_center, quantize_frequencies


def tone_increments(freq, quantize=True, **kwargs):
    """
    Compute the DDS tone increment for each frequency (in Hz),
    assumes channel will use OPFB bin returned by mkidgen3.drivers.bintores.opfb_bin_number
    when computing central frequency

    If quantize, the tone increment frequencies will be quantized via dsp.quantize_frequencies
    """
    centers = opfb_bin_center(opfb_bin_number(freq, ssr_raw_order=True), ssr_order=True)
    # This must be 2MHz NOT 2.048MHz, the sign matters! Use 1MHz as that corresponds to ±Pi
    x = (freq - centers)
    if quantize:
        x = quantize_frequencies(x, **kwargs)
    return x / 1e6


class DDC(DefaultIP):
    offset_tones = 0x2000
    TONE_FORMAT = (1, 10, 'signed')  # ap_fixed<11,1>
    PHASE0_FORMAT = (1, 20, 'signed')  # ap_fixed<21,1>

    def __init__(self, description):
        """
        The core uses an array of 256 values, each consisting of 8 32 bit numbers packed into 256 bit word that
        specifies the phase offset and phase increment used to digitally down-convert the corresponding resonator
        channel. Each 32 bit number is itself a packed fixed point number with 1 integer bit, 21 bits for the phase and
        11 for the tone. The high bits are for the phase offset.

        0x2000 ~
        0x3fff : Memory 'tones' (256 * 256b)  inc0-8 p0 0-8
                 Word 8n   : bit [31:0] - tones[n][31: 0]
                 Word 8n+1 : bit [31:0] - tones[n][63:32]
                 Word 8n+2 : bit [31:0] - tones[n][95:64]
                 Word 8n+3 : bit [31:0] - tones[n][127:96]
                 Word 8n+4 : bit [31:0] - tones[n][159:128]
                 Word 8n+5 : bit [31:0] - tones[n][191:160]
                 Word 8n+6 : bit [31:0] - tones[n][223:192]
                 Word 8n+7 : bit [31:0] - tones[n][255:224]


        """
        super().__init__(description=description)

    bindto = ['mazinlab:mkidgen3:resonator_dds:1.33']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx, raw=False):
        """Read the numbers in the group from the core and convert them from binary data to python numbers"""
        self._checkgroup(group_ndx)
        vals = [self.read(self.offset_tones + 32 * group_ndx + 4 * i) for i in range(8)]  # 32 bit packed word
        tone_mask = 2 ** 12 - 1
        phase_mask = 2 ** 22 - 1
        tone_fmt = fp_factory(*self.TONE_FORMAT, frombits=True)
        phase_fmt = fp_factory(*self.PHASE0_FORMAT, frombits=True)

        x = 0
        for i, v in enumerate(vals):
            x |= v << (32 * i)
        tones = x
        offsets = x >> (11 * 8)

        tones = [int((tones >> (11 * i)) & tone_mask) for i in range(8)]
        offsets = [int((offsets >> (21 * i)) & phase_mask) for i in range(8)]
        if not raw:
            tones = [float(tone_fmt(v)) for v in tones]
            offsets = [float(phase_fmt(v)) for v in offsets]
        return tones, offsets

    def write_group(self, group_ndx, increments, phases):
        """ Convert the numbers in the group from python data to binary data and load it into the core """
        self._checkgroup(group_ndx)
        if len(increments) != 8 or len(phases) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        tone_fmt = fp_factory(*self.TONE_FORMAT, include_index=True)
        phase_fmt = fp_factory(*self.PHASE0_FORMAT, include_index=True)
        for i, (g0, g1) in enumerate(zip(map(tone_fmt, increments), map(phase_fmt, phases))):
            bits |= ((g1 << (11 * 8)) | g0) << (11 * i)
        data = bits.to_bytes(32, 'little', signed=False)
        self.write(self.offset_tones + 32 * group_ndx, data)

    @property
    def tones(self):
        return np.hstack([self.read_group(g) for g in range(256)])

    @tones.setter
    def tones(self, tones):
        """tones is a [2,2048] array of tone increments and phase offsets """
        if tones.shape != (2, 2048):
            raise ValueError('tones.shape !=(2,2048)')
        if tones.min() < -1 or tones.max() > 1:
            raise ValueError('Tones must be in [-1,1)')
        for i in range(256):
            self.write_group(i, *tones[:, i * 8:i * 8 + 8])


class OldDDC(DefaultIP):
    offset_tones = 0x2000

    def __init__(self, description):
        """
        Note the axilite memory space is
        0x2000 ~
        0x3fff : Memory 'tones' (256 * 256b)  inc0-8 p0 0-8
                 Word 8n   : bit [31:0] - tones[n][31: 0]
                 Word 8n+1 : bit [31:0] - tones[n][63:32]
                 Word 8n+2 : bit [31:0] - tones[n][95:64]
                 Word 8n+3 : bit [31:0] - tones[n][127:96]
                 Word 8n+4 : bit [31:0] - tones[n][159:128]
                 Word 8n+5 : bit [31:0] - tones[n][191:160]
                 Word 8n+6 : bit [31:0] - tones[n][223:192]
                 Word 8n+7 : bit [31:0] - tones[n][255:224]
        """
        super().__init__(description=description)

    bindto = ['MazinLab:mkidgen3:resonator_dds:0.13', 'MazinLab:mkidgen3:resonator_dds:1.0']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx, offset, fmt=(1, 15), consecutive=True, signed=True):
        """Read the numbers in group from the core and convert them from binary data to python numbers"""
        self._checkgroup(group_ndx)
        if fmt is None:
            fmt = lambda x: np.int16(x) if signed else np.uint16(x)
        else:
            fmt = lambda x: float(FpBinary(int_bits=fmt[0], frac_bits=fmt[1], signed=signed, bit_field=x))
        vals = [self.read(offset + 32 * group_ndx + 4 * i) for i in range(8)]  # 2 16bit values each
        if consecutive:
            a = [fmt((v >> (16 * i)) & 0xffff) for v in vals[:4] for i in (0, 1)]
            b = [fmt((v >> (16 * i)) & 0xffff) for v in vals[4:] for i in (0, 1)]
        else:
            a = [fmt((v >> (16 * i)) & 0xffff) for v in vals[::2] for i in (0, 1)]
            b = [fmt((v >> (16 * i)) & 0xffff) for v in vals[::2] for i in (0, 1)]
        return a, b

    def write_group(self, group_ndx, increments, phases):
        """Convert the numbers in the group from python data to binary data and load it into the core"""
        self._checkgroup(group_ndx)
        if len(increments) != 8 or len(phases) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        fixedgroup = list(map(FP16_15, increments)) + list(map(FP16_15, phases))
        for i, (g0, g1) in enumerate(zip(*[iter(fixedgroup)] * 2)):  # take them by twos
            bits |= ((g1 << 16) | g0) << (32 * i)
        data = bits.to_bytes(32, 'little', signed=False)
        bits.to_bytes(32, 'little', signed=False)
        self.write(self.offset_tones + 32 * group_ndx, data)

    @property
    def tones(self):
        return np.hstack([self.read_group(g, self.offset_tones) for g in range(256)])

    @tones.setter
    def tones(self, tones):
        """tones[2,2048]"""
        if tones.shape != (2, 2048):
            raise ValueError('tones.shape !=(2,2048)')
        if tones.min() < -1 or tones.max() > 1:
            raise ValueError('Tones must be in [-1,1)')
        for i in range(256):
            self.write_group(i, *tones[:, i * 8:i * 8 + 8])


class OldOldDDC(DefaultIP):
    """
    Note the axilite memory space is
    0x1000 ~
    0x1fff : Memory 'toneinc_V' (256 * 128b)
             Word 4n   : bit [31:0] - toneinc_V[n][31: 0]
             Word 4n+1 : bit [31:0] - toneinc_V[n][63:32]
             Word 4n+2 : bit [31:0] - toneinc_V[n][95:64]
             Word 4n+3 : bit [31:0] - toneinc_V[n][127:96]
    0x2000 ~
    0x2fff : Memory 'phase0_V' (256 * 128b)
             Word 4n   : bit [31:0] - phase0_V[n][31: 0]
             Word 4n+1 : bit [31:0] - phase0_V[n][63:32]
             Word 4n+2 : bit [31:0] - phase0_V[n][95:64]
             Word 4n+3 : bit [31:0] - phase0_V[n][127:96]
    """
    toneinc_offset = 0x1000
    phase0_offset = 0x2000

    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['MazinLab:mkidgen3:resonator_dds:0.5']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, offset, group_ndx):
        """Read the numbers in group from the core and convert them from binary data to python numbers"""
        self._checkgroup(group_ndx)
        signed = offset == self.toneinc_offset
        vals = [self.read(offset + 16 * group_ndx + 4 * i) for i in range(4)]  # 2 16bit values each
        ret = [float(FpBinary(1, 15, signed=signed, bit_field=(v >> (16 * i)) & 0xffff))
               for v in vals for i in (0, 1)]
        # print(f"Read {bin(vals[0]&0xffff)} from the first address.")
        return ret

    def write_group(self, offset, group_ndx, group):
        """Convert the numbers in the group from python data to binary data and load it into the core"""
        self._checkgroup(group_ndx)
        if len(group) != 8:
            raise ValueError('len(group)!=8')
        signed = offset == self.toneinc_offset
        bits = 0
        fixedgroup = [FpBinary(int_bits=1, frac_bits=15, signed=signed, value=g) for g in group]
        for i, (g0, g1) in enumerate(zip(*[iter(fixedgroup)] * 2)):  # take them by twos
            bits |= ((g1.__index__() << 16) | g0.__index__()) << (32 * i)
        data = bits.to_bytes(16, 'little', signed=False)
        # print(f"Writing {bin(bits&0xffff)} to the first address.")
        self.write(offset + 16 * group_ndx, data)

    def toneinc(self, res):
        """ Retrieve the tone increment for a particular resonator """
        return self.read_group(self.toneinc_offset, res // 8)[res % 8]

    def phase0(self, res):
        """ Retrieve the phase offset for a particular resonator """
        return self.read_group(self.phase0_offset, res // 8)[res % 8]

    @property
    def toneincs(self):
        return [v for g in range(256) for v in self.read_group(self.toneinc_offset, g)]

    @toneincs.setter
    def toneincs(self, toneincs):
        if len(toneincs) != 2048:
            raise ValueError('len(toneincs)!=2048')
        if min(toneincs) < -1 or max(toneincs) >= 1:
            raise ValueError('Tone increments must be in [-1,1)')
        for i in range(256):
            self.write_group(self.toneinc_offset, i, toneincs[i * 8:i * 8 + 8])

    @property
    def phase0s(self):
        return [v for g in range(256) for v in self.read_group(self.phase0_offset, g)]

    @phase0s.setter
    def phase0s(self, phase0s):
        if len(phase0s) != 2048:
            raise ValueError('len(phase0s)!=2048')
        if min(phase0s) < 0 or max(phase0s) > 1:
            raise ValueError('Phase offsets must be in [0,1]')
        for i in range(256):
            self.write_group(self.phase0_offset, i, phase0s[i * 8:i * 8 + 8])
