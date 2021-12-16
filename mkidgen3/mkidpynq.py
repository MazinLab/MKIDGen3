import subprocess
subprocess.run(['ls', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8')

from pynq import DefaultHierarchy
import numpy as np
from fpbinary import FpBinary
from logging import getLogger

MAX_CAP_RAM_BYTES = 2**32
PL_DDR4_ADDR = 0x500000000
N_IQ_GROUPS = 256
FP16_15 = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x)
FP32_8 = lambda x: FpBinary(int_bits=32 - 9, frac_bits=8, signed=True, value=x)


def fp_factory(int, frac, signed, frombits=False):
    if isinstance(signed, str):
        signed = True if 'signed' == signed.lower() else False
    else:
        signed = bool(signed)
    if frombits:
        return lambda x: FpBinary(int_bits=int, frac_bits=frac, signed=signed, bit_field=x)
    else:
        return lambda x: FpBinary(int_bits=int, frac_bits=frac, signed=signed, value=x)


def get_pldram_addr(hwhpath):
    """Return PL DRAM start address as specified in hwh"""

    pldramstr = '<MEMRANGE ADDRESSBLOCK="C0_DDR4_ADDRESS_BLOCK" BASENAME="C_BASEADDR" BASEVALUE="'
    hwh = open(hwhpath, "r")

    for line in hwh:
        if pldramstr in line:
            break
    try:
        pldram_addr = hex(int(line[88:99], 16))
    except LookupError:
        print('PL DRAM not found')
    hwh.close()
    return(pldram_addr)




def _which_one_bit_set(x, nbits):
    """
    Given the number x that only has a single bit set return the index of that bit.
    Return None if no bits < nbits bit is set (e.g. nbits=16 will check bits 0-15)
    """
    for i in range(nbits):
        if x & (1 << i):
            return i
    return None


def pack16_to_32(data):
    it = iter(data)
    vals = [x | (y << 16) for x, y in zip(it, it)]
    if data.size % 2:
        vals.append(data[-1])
    return np.array(vals, dtype=np.uint32)


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


class CaptureHierarchy(DefaultHierarchy):
    def __init__(self, description):
        super().__init__(description)
        # self.raw_phase_cap =
        # self.filt_phase_cap =
        self.adc_cap = self.adc_capture_0
        self.raw_iq_cap = self.iq_capture_0
        self.dds_iq_cap = self.iq_capture_1
        self.lp_iq_cap = None
        self.mcdma = self.axi_mcdma_0

    @staticmethod
    def hierarchy(description):
        for k in ('iq_capture_0','iq_capture_1','adc_capture_0', 'axi_mcdma_0'):
            if k not in description['ip']:
                return False
        return True

    def capture(self, n, res=8019, buffer_size=8192):

        raw_groups=(0, 1, 3)
        dds_groups=(128, 200, 255)

        IQ_GROUP_BYTES = 64

        raw_bytes_required = IQ_GROUP_BYTES*len(raw_groups)*n
        dds_bytes_required = IQ_GROUP_BYTES*len(dds_groups)*n

        total_bytes = raw_bytes_required +dds_bytes_required
        getLogger(__name__).info('Capture will require {total_bytes/1024/1024} MiB of DDR')
        if total_bytes > (2**32-1):
            raise MemoryError('Insufficient capture space')

        #TODO make this deal with all the different lanes properly
        # TODO to support different numbers of groups we need to ensure there are enough buffers for channel that
        #  will et the most data. That means some buffers oversized or per channel buffersize
        buffers_needed = np.ceil(raw_bytes_required/buffer_size)

        self.reset_capture()
        self.mcdma.config_recieve(n_buffers=buffers_needed, buffer_size_bytes=buffer_size, channels=(1, 2))
        self.raw_iq_cap.capture(n, groups=raw_groups, start=False)
        self.dds_iq_cap.capture(n, groups=dds_groups, start=False)
        self.raw_iq_cap.start_capture()
        self.dds_iq_cap.start_capture()

    def reset_capture(self):
        """ Terminate any capture and prep blocks so that capture can happen"""
        self.raw_iq_cap.halt_capture()
        self.dds_iq_cap.halt_capture()

    def capture_adc(self, n, res=8019):
        self.mcdma.config_recieve(n_buffers=2, buffer_size_bytes=8096, channels=(1, 2))
        self.adc_cap.capture(n, start=True)



# LUT of property addresses for our data-driven properties
_qpsk_props = [("transfer_symbol", 0), ("transfer_fft", 4),
               ("transfer_time", 60), ("reset_symbol", 8), ("reset_fft", 12),
               ("reset_time", 48), ("packetsize_symbol", 16),
               ("packetsize_rf", 20), ("packetsize_fft", 24),
               ("packetsize_time", 52), ("autorestart_symbol", 36),
               ("autorestart_fft", 40), ("autorestart_time", 56),
               ("lfsr_rst", 28), ("enable", 32), ("output_gain", 44)]


# Func to return a MMIO getter and setter based on a relative addr


def _mimo_attacher(class_def, mimo_regs):
    # Generate getters and setters based on mimo_regs
    def _create_mmio_property(addr):
        def _get(self):
            return self.read(addr)

        def _set(self, value):
            self.write(addr, value)

        return property(_get, _set)

    for (name, addr) in mimo_regs:
        setattr(class_def, name, _create_mmio_property(addr))
