import subprocess
import numpy as np

MAX_CAP_RAM_BYTES = 2**32
PL_DDR4_ADDR = 0x500000000
N_IQ_GROUPS = 256

PHOTON_DTYPE = np.dtype([('time', np.uint64), ('phase', np.int16), ('id', np.uint16)])

def get_board_name():
    x = subprocess.run(['cat', '/proc/device-tree/chosen/pynq_board'], capture_output=True, text=True).stdout
    return x.strip().strip('\x00')


def enable_axi_timeout():
    """ See https://discuss.pynq.io/t/help-debuging-chronic-pynq-system-hang/970"""
    import pynq
    #LPD
    mmio = pynq.MMIO(0xFF416000, 64)
    mmio.write(0x18, 3)  # Return slave errors when timeouts occur
    mmio.write(0x20, 0x1020)  # Set and enable prescale of 32 which should be about 10 ms
    mmio.write(0x10, 0x3)  # Enable transactions tracking
    mmio.write(0x14, 0x3)  # Enable timeouts

    #FPD
    mmio = pynq.MMIO(0xFD610000, 64)
    mmio.write(0x18, 7)  # Return slave errors when timeouts occur
    mmio.write(0x20, 0x1020)  # Set and enable prescale of 32 which should be about 10 ms
    mmio.write(0x10, 0x7)  # Enable transactions tracking
    mmio.write(0x14, 0x7)  # Enable timeouts


def get_pldram_addr(hwhpath):
    """Return PL DRAM start address as specified in hwh"""
    pldram_addr = None
    pldramstr = '<MEMRANGE ADDRESSBLOCK="C0_DDR4_ADDRESS_BLOCK" BASENAME="C_BASEADDR" BASEVALUE="'
    with open(hwhpath, "r") as hwh:
        for line in hwh:
            if pldramstr in line:
                break
        try:
            pldram_addr = hex(int(line[88:99], 16))
        except LookupError:
            print('PL DRAM not found')
    return pldram_addr


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


def check_description_for(description, kinds, check_version=False, force_dict=False):
    if isinstance(kinds, str):
        kinds = (kinds,)
    ret = {k: [] for k in kinds}
    for k in description['ip']:
        kind = description['ip'][k].get('type', '')
        if not check_version:
            kind, _, version = kind.rpartition(':')
        if kind in kinds:
            ret[kind].append(k)

    return ret if force_dict or len(kinds)>1 else ret[kinds[0]]
