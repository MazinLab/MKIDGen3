from pynq import DefaultIP
import numpy as np
from fpbinary import FpBinary, OverflowEnum, RoundingEnum


def dma_status(dma):
    # dma.recvchannel.idle,dma.sendchannel.idle
    msg = ("DMA:\n"
           f" Buffer Length: {dma.buffer_max_size} bytes\n"
           " MM2s\n"
           f" Idle:{dma.sendchannel.idle}\n"
           f" MM2S_DMASR (status):{hex(dma.mmio.read(4))}\n"
           f" MM2S_SA (ptr) :{hex(dma.mmio.read(24))}\n"
           f" MM2S_LENGTH (len):{dma.mmio.read(40)}\n"
           " S2MM\n"
           f" Idle:{dma.recvchannel.idle}\n"
           f" S2MM_DMASR (status):{hex(dma.mmio.read(52))}\n"
           f" S2MM_DA (ptr) :{hex(dma.mmio.read(72))}\n"
           f" S2MM_LENGTH (len):{dma.mmio.read(88)}")
    print(msg)


class AxisSwitch(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['xilinx.com:ip:axis_switch:1.1']

    def set_master(self, master, slave=0, disable=False):
        """Set the slave for the master"""
        cfg = 0x80000000 if disable else (slave & 0b111)
        self.write(0x0040 + master * 4, cfg)

    def status(self):
        self.read(0x0040)

    def commit(self):
        """Commit config, triggers a soft 16 cycle reset"""
        self.write(0x0000, 0x2)


class BinToResIP(DefaultIP):
    resmap_addr = 0x1000

    def __init__(self, description):
        """
        0x1fff : Memory 'data_V' (256 * 96b)
        Word 4n   : bit [31:0] - data_V[n][31: 0]
        Word 4n+1 : bit [31:0] - data_V[n][63:32]
        Word 4n+2 : bit [31:0] - data_V[n][95:64]
        Word 4n+3 : bit [31:0] - reserved
        """
        super().__init__(description=description)

    bindto = ['MazinLab:mkidgen3:bin_to_res:0.6']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx):
        self._checkgroup(group_ndx)
        g = 0
        vals = [self.read(self.resmap_addr + 16 * group_ndx + 4 * i) for i in range(3)]
        for i, v in enumerate(vals):
            # print(format(v,'032b'))
            g |= v << (32 * i)
        # print('H-'+format(g,'096b')+'-L')
        return [((g >> (12 * j)) & 0xfff) for j in range(8)]

    def write_group(self, group_ndx, group):
        self._checkgroup(group_ndx)
        if len(group) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        for i, g in enumerate(group):
            bits |= (int(g) & 0xfff) << (12 * i)
        data = bits.to_bytes(12, 'little', signed=False)
        self.write(self.resmap_addr + 16 * group_ndx, data)

    def bin(self, res):
        """ The mapping for resonator i is 12 bits and will require reading 1 or 2 32 bit word
        n=i//8 j=(i%8)*12//32
        """
        return self.read_group(res // 8)[res % 8]

    @property
    def bins(self):
        return [v for g in range(256) for v in self.read_group(g)]

    @bins.setter
    def bins(self, bins):
        if len(bins) != 2048:
            raise ValueError('len(bins)!=2048')
        if min(bins) < 0 or max(bins) > 4095:
            raise ValueError('Bin values must be in [0,4095]')
        for i in range(256):
            self.write_group(i, bins[i * 8:i * 8 + 8])


class ResonatorDDSV2IP(DefaultIP):
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

    bindto = ['MazinLab:mkidgen3:resonator_dds:0.13']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx, offset, fmt=(1, 15), consecutive=True, signed=True):
        """Read the numbers in group from the core and convert them from binary data to python numbers"""
        self._checkgroup(group_ndx)
        if fmt is None:
            FMT = lambda x: np.int16(x) if signed else np.uint16(x)
        else:
            FMT = lambda x: float(FpBinary(int_bits=fmt[0], frac_bits=fmt[1], signed=signed, bit_field=x))
        vals = [self.read(offset + 32 * group_ndx + 4 * i) for i in range(8)]  # 2 16bit values each
        if consecutive:
            a = [FMT((v >> (16 * i)) & 0xffff) for v in vals[:4] for i in (0, 1)]
            b = [FMT((v >> (16 * i)) & 0xffff) for v in vals[4:] for i in (0, 1)]
        else:
            a = [FMT((v >> (16 * i)) & 0xffff) for v in vals[::2] for i in (0, 1)]
            b = [FMT((v >> (16 * i)) & 0xffff) for v in vals[::2] for i in (0, 1)]
        return a, b

    def write_group(self, group_ndx, increments, phases):
        """Convert the numbers in the group from python data to binary data and load it into the core"""
        self._checkgroup(group_ndx)
        if len(increments) != 8 or len(phases) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        FP16_15 = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x).__index__()
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


class ResonatorDDSIP(DefaultIP):
    toneinc_offset = 0x1000
    phase0_offset = 0x2000

    def __init__(self, description):
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
        for i, (g0, g1) in enumerate(zip(*[iter(fixedgroup)] * 2)):  #take them by twos
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

#
#
# class ReschanIP(DefaultIP):
#     toneinc_offset = 0x1000
#     phase0_offset = 0x2000
#
#     def __init__(self, description):
#         """
#
#         Note the axilite memory space is
#         0x1000 ~
#         0x1fff : Memory 'toneinc_V' (256 * 128b)
#                  Word 4n   : bit [31:0] - toneinc_V[n][31: 0]
#                  Word 4n+1 : bit [31:0] - toneinc_V[n][63:32]
#                  Word 4n+2 : bit [31:0] - toneinc_V[n][95:64]
#                  Word 4n+3 : bit [31:0] - toneinc_V[n][127:96]
#         0x2000 ~
#         0x2fff : Memory 'phase0_V' (256 * 128b)
#                  Word 4n   : bit [31:0] - phase0_V[n][31: 0]
#                  Word 4n+1 : bit [31:0] - phase0_V[n][63:32]
#                  Word 4n+2 : bit [31:0] - phase0_V[n][95:64]
#                  Word 4n+3 : bit [31:0] - phase0_V[n][127:96]
#         """
#         super().__init__(description=description)
#
#     bindto = ['MazinLab:mkidgen3:gen3_reschan:1.12']
#
#     @staticmethod
#
#
#     def toneinc(self, res):
#         """ Retrieve the tone increment for a particular resonator """
#         return self.read_group(self.toneinc_offset, res // 8)[res % 8]
#
#     def phase0(self, res):
#         """ Retrieve the phase offset for a particular resonator """
#         return self.read_group(self.phase0_offset, res // 8)[res % 8]
#
#     @property
#     def toneincs(self):
#         return [v for g in range(256) for v in self.read_group(self.toneinc_offset, g)]
#
#     @toneincs.setter
#     def toneincs(self, toneincs):
#         if len(toneincs) != 2048:
#             raise ValueError('len(toneincs)!=2048')
#         if min(toneincs) < -1 or max(toneincs) >= 1:
#             raise ValueError('Tone increments must be in [-1,1)')
#         for i in range(256):
#             self.write_group(self.toneinc_offset, i, toneincs[i * 8:i * 8 + 8])
#
#     @property
#     def phase0s(self):
#         return [v for g in range(256) for v in self.read_group(self.phase0_offset, g)]
#
#     @phase0s.setter
#     def phase0s(self, phase0s):
#         if len(phase0s) != 2048:
#             raise ValueError('len(phase0s)!=2048')
#         if min(phase0s) < 0 or max(phase0s) > 1:
#             raise ValueError('Phase offsets must be in [0,1]')
#         for i in range(256):
#             self.write_group(self.phase0_offset, i, phase0s[i * 8:i * 8 + 8])
