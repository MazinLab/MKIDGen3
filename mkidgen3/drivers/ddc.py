import logging

import numpy as np
from fpbinary import FpBinary
from pynq import DefaultIP
from pynq.mmio import MMIO
from mkidgen3.fixedpoint import fp_factory
from mkidgen3.opfb import opfb_bin_number, opfb_bin_center, quantize_frequencies
import time
import mkidgen3.opfb as dsp
from logging import getLogger
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


# The DDC tone table registers are arranged with least significant bits and bytes at
# lower addresses in 256bit words of p0_7 ... p0_0 i_7 ... i_0
# the increments i are 11 bits and the phase offsets 21 bits.
#
# Addrresses are written by read and write which rea and write 32 bit words.
# so the first 32bit word of the tone table consists of 10 bits of i2, i1 and i0:  i2_9 ... i2_0 i1 i0
# .to_bytes(4, 'big', signed=False) will create a string that prints like this is written, but do not match how
# mimo.write writes the data.
#
# bitstruct.compile('>u11'*8).pack(*[1021]*8) will also printlike this is written, but it too is wrong
# bitstruct.compile('>u11'*8+'<').pack(*[1021]*8)
#
# mimo.write(addr, b'\x??\x??\x??\x??...')  writes to the core as
# B0 B1 B2 .... The key here is that \x## is read as the number 0x00, so 0xF0 is 240 and 0x0F is 15. So we are writing the numbers least significant byte in first (and least significant bit as well) and will read from the core in the same manner.
#
# bitstruct does not seem to be able to properly render this
# bitstruct.pack('>u32<', x) will properly pack a u32 into bytes but fails with u11


class DDC(DefaultIP):
    offset_tones = 0x2000
    TONE_FORMAT = (1, 10, 'signed')  # ap_fixed<11,1>
    PHASE0_FORMAT = (1, 20, 'signed')  # ap_fixed<21,1>

    bindto = ['mazinlab:mkidgen3:resonator_dds:1.33']

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

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx, raw=False):
        """Read the numbers in the group from the core and convert them from binary data to python numbers"""
        self._checkgroup(group_ndx)

        tone_fmt = fp_factory(*self.TONE_FORMAT, frombits=True)
        phase_fmt = fp_factory(*self.PHASE0_FORMAT, frombits=True)

        t_bits = sum(self.TONE_FORMAT[:2])
        p_bits = sum(self.PHASE0_FORMAT[:2])

        t_mask = 2 ** t_bits - 1
        p_mask = 2 ** p_bits - 1

        x = 0
        for i in range(8):
            v = self.read(self.offset_tones + 32 * group_ndx + i * 4)
            x |= v << (32 * i)

        tones = [(x >> (t_bits * i)) & t_mask for i in range(8)]
        x >>= 88
        phases = [(x >> (p_bits * i)) & p_mask for i in range(8)]

        if not raw:
            tones = [float(tone_fmt(v)) for v in tones]
            phases = [float(phase_fmt(v)) for v in phases]

        return tones, phases

    def write_group(self, group_ndx, increments, phases, raw=False):
        """ Convert the numbers in the group from python data to binary data and load it into the core """
        self._checkgroup(group_ndx)
        if len(increments) != 8 or len(phases) != 8:
            raise ValueError('len(group)!=8')

        tone_fmt = (lambda x: x) if raw else fp_factory(*self.TONE_FORMAT, include_index=True)
        phase_fmt = (lambda x: x) if raw else fp_factory(*self.PHASE0_FORMAT, include_index=True)

        t_bits = sum(self.TONE_FORMAT[:2])
        p_bits = sum(self.PHASE0_FORMAT[:2])

        inc = 0
        for i, v in enumerate(map(tone_fmt, increments)):
            inc |= v << (t_bits * i)

        pha = 0
        for i, v in enumerate(map(phase_fmt, phases)):
            pha |= v << (p_bits * i)

        d = (pha << 88) | inc
        data = d.to_bytes(32, 'little', signed=False)
        self.write(self.offset_tones + 32 * group_ndx, data)

    def pack_group(self, increments, phases, raw=False):
        """ Convert the numbers in the group from python data to binary data and load it into the core """

        if len(increments) != 8 or len(phases) != 8:
            raise ValueError('len(group)!=8')

        tone_fmt = (lambda x: x) if raw else fp_factory(*self.TONE_FORMAT, include_index=True)
        phase_fmt = (lambda x: x) if raw else fp_factory(*self.PHASE0_FORMAT, include_index=True)

        t_bits = sum(self.TONE_FORMAT[:2])
        p_bits = sum(self.PHASE0_FORMAT[:2])

        inc = 0
        for i, v in enumerate(map(tone_fmt, increments)):
            inc |= v << (t_bits * i)

        pha = 0
        for i, v in enumerate(map(phase_fmt, phases)):
            pha |= v << (p_bits * i)

        d = (pha << 88) | inc
        data = d.to_bytes(32, 'little', signed=False)
        data = np.frombuffer(data, dtype=np.uint32)
        return data

    @property
    def tones(self):
        return np.hstack([self.read_group(g) for g in range(256)])

    @tones.setter
    def tones(self, tones):
        """tones is a [2,2048] array of tone increments and phase offsets """
        if tones.shape != (2, 2048):
            raise ValueError('tones.shape !=(2,2048)')
        if tones.min() < -1 or tones.max() >= 1:
            raise ValueError('Tones must be in [-1,1)')
        for i in range(256):
            self.write_group(i, *tones[:, i * 8:i * 8 + 8])


class CenteringDDC(DDC):
    CENTER_FORMAT = (1, 15, 'signed')  # ap_fixed<16,15>
    offset_centers = 0x4000
    bindto = ['mazinlab:mkidgen3:resonator_ddc_control:1.0']

    def reset_accumulator(self):
        """ Reset any accumulated phase """
        self.register_map.clear_accumulator = True
        time.sleep(256*4/512e6)
        self.register_map.clear_accumulator = False

    @property
    def centers(self):
        """ Returns an array of 2048 complex loop centers [1,1] """
        mmio = MMIO(self.offset_centers, length=4 * 2048)
        u32d = np.array(mmio.array, dtype=np.uint32)
        u16 = np.frombuffer(u32d, dtype=np.uint16).reshape((2048, 2))
        center_fmt = fp_factory(*self.CENTER_FORMAT, frombits=True, include_index=True)
        data = np.zeros(2048, dtype=np.complex64)
        data.real = [float(center_fmt(int(x))) for x in u16[:, 0]]
        data.imag = [float(center_fmt(int(x))) for x in u16[:, 1]]
        return data

    @centers.setter
    def centers(self, centers):
        """ Centers is an array of 2048 complex loop centers [1,1] """
        if centers.shape != (2048,):
            raise ValueError('centers.shape != (2048,)')
        if np.abs(centers.real).max() > 1 or np.abs(centers.imag).max() > 1:
            raise ValueError('Centers must be in [-1,1)')
        if np.abs(centers).max() > 1:
            logging.getLogger(__name__).warning('Centers contains magnitudes outside of the unit circle')

        center_fmt = fp_factory(*self.CENTER_FORMAT, frombits=False, include_index=True)

        data = np.zeros((2048, 2), dtype=np.uint16)
        data[:, 0] = [center_fmt(x.real) for x in centers]
        data[:, 1] = [center_fmt(x.imag) for x in centers]
        u32d = np.frombuffer(data, dtype=np.uint32)
        mmio = MMIO(self.offset_centers, length=4 * 2048)
        mmio.array[:] = u32d


class ThreepartDDC(CenteringDDC):
    def __init__(self, bram_controller_mmio):
        #backup: init with bram_controller_mmio = MMIO(offset, length=8 * 2048)
        self.mmio=bram_controller_mmio

    def configure(self, tones=None, phase_offset=None, loop_center=None, center_relative=False, quantize=True):
        """
        Configure the DDC to down-convert resonator channels containing the specified tones.

        Optionally phase offsets may be specified for each channel. Phase offsets are in [-pi,pi] radians. Values outside
        are clipped.

        Optionally loop centers may be specified (complex nominally [-1,1) though values will be clipped when converted
        from floating to fix point).

        Center relative means specify the frequency relative to the bin center.

        Only the first 2048 frequencies/offsets will be used. DDC tones are assigned in the order frequencies are
        specified. If fewer than 2048 frequencies are specified no assumptions may be made about the DDC settings for the
        remaining channels, however they may be determined by inspecting the .tones property of resonator_ddc.
        """
        data = np.zeros((2, 2048))
        freq = np.asarray(tones)

        if freq.size > 2048:
            getLogger(__name__).warning(f'Using first 2048 of {freq.size} provided frequencies')

        if center_relative:
            freq = dsp.quantize_frequencies(freq) if quantize else freq
            data[0, :min(freq.size, 2048)] = freq[:2048] / 1e6
        else:
            data[0, :min(freq.size, 2048)] = tone_increments(freq[:2048], quantize=quantize)

        if phase_offset is not None:
            phase_offset = np.asarray(phase_offset)
            if freq.size != phase_offset.size:
                raise ValueError('If provided, phase_offsets must match frequencies')
            phase_offset = (phase_offset / np.pi).clip(-1, 1)
            data[1, :min(freq.size, 2048)] = phase_offset[:2048]

        centers = np.zeros(2048, dtype=np.complex64)
        if loop_center is not None:
            loop_center = np.asarray(loop_center)
            if freq.size != loop_center.size:
                raise ValueError('If provided, loop_center must match frequencies')
            centers[:loop_center[:2048].size] = loop_center[:2048]
            if (np.abs(centers) > 1).any():
                getLogger(__name__).warning(f'Loop centers exist outside of the unit circle')

        self._configure(data, centers)

    def _configure(self, tones, centers):
        """ Centers is an array of 2048 complex loop centers [1,1] """
        if centers.shape != (2048,):
            raise ValueError('centers.shape != (2048,)')
        if np.abs(centers.real).max() > 1 or np.abs(centers.imag).max() > 1:
            raise ValueError('Centers must be in [-1,1)')
        if np.abs(centers).max() > 1:
            logging.getLogger(__name__).warning('Centers contains magnitudes outside of the unit circle')

        if tones.shape != (2, 2048):
            raise ValueError('tones.shape !=(2,2048)')
        if tones.min() < -1 or tones.max() >= 1:
            raise ValueError('Tones must be in [-1,1)')

        center_fmt = fp_factory(*self.CENTER_FORMAT, frombits=False, include_index=True)

        data2 = np.zeros((2048, 2), dtype=np.uint16)
        data2[:, 0] = [center_fmt(x.real) for x in centers]
        data2[:, 1] = [center_fmt(x.imag) for x in centers]

        data1 = np.zeros((256, 8), dtype=np.uint32)
        for i in range(256):
            data1[i, :] = self.pack_group(*(-tones[:, i * 8:i * 8 + 8]))

        data = np.zeros((256, 2, 8), dtype=np.uint32)
        data[:, 0, :] = data1
        data[:, 1, :] = np.frombuffer(data2, dtype=np.uint32).reshape((256, 8))
        u32d = np.frombuffer(data, dtype=np.uint32)
        # mmio = MMIO(self.offsoffsetet, length=8 * 2048)
        self.mmio.array[:u32d.size] = u32d
