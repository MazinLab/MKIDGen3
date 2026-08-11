import subprocess
import time
from logging import getLogger

import numpy as np

MAX_CAP_RAM_BYTES = 2**32
PL_DDR4_ADDR = 0x500000000

# The two PS-DDR windows the PL sees through HP0 on NODDR builds, where
# axis2mm's M_AXI joins the trigger's HP0 interconnect and capture results
# land in PS DDR instead of PL DDR4. Half-open (start, end).
HP0_WINDOWS = ((0x0000_0000, 0x8000_0000),
               (0x0008_0000_0000, 0x0009_0000_0000))


def hp0_reachable(addr, nbytes=1):
    """True if [addr, addr+nbytes) lies inside one HP0 window.

    An address in neither window DECERRs, and the daemon sees nothing at all
    -- no error, no data -- so check before arming a DMA. A buffer may not
    straddle the gap between the windows.
    """
    a = int(addr)
    n = max(1, int(nbytes))
    if a < 0:
        return False
    return any(lo <= a and a + n <= hi for lo, hi in HP0_WINDOWS)


# axis2mm moves whole 64-byte beats; its length register is in bytes while
# capture buffers are typed arrays, and mixing the two units is how a buffer
# ends up a fraction of the size the DMA was told to write.
CAPTURE_BEAT_BYTES = 64


def flush_transfer(n):
    """(u64 words to allocate, bytes axis2mm will write) to flush n beats.

    A flush of n beats is n*64 bytes, which is eight u64 words per beat --
    allocating n words covers an eighth of what the DMA is programmed with,
    and the other seven eighths land past the end of the buffer.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f'flush needs at least one beat, got {n}')
    nbytes = CAPTURE_BEAT_BYTES * n
    return nbytes // 8, nbytes


def arm_fault(addr, nbytes, buffer_nbytes=None, hp0=True):
    """Why axis2mm must not be armed for [addr, addr+nbytes), or None if it may.

    Checks the interval the DMA is actually programmed with rather than the
    one that was asked for: it must be whole beats, it must fit inside the
    buffer backing it, and on an HP0 path it must fit inside a single window.
    Returns a reason string so the caller can raise, and stays pure so it is
    testable off-board.

    Pass hp0=False for PL DDR4 targets, which do not reach memory through HP0
    and whose addresses lie outside both windows by construction.
    """
    addr = int(addr)
    nbytes = int(nbytes)
    if nbytes <= 0:
        return f'refusing to arm axis2mm for a {nbytes} byte transfer'
    if nbytes % CAPTURE_BEAT_BYTES:
        return (f'transfer of {nbytes} B is not a whole number of '
                f'{CAPTURE_BEAT_BYTES} B beats')
    if buffer_nbytes is not None and nbytes > int(buffer_nbytes):
        return (f'transfer of {nbytes} B at {addr:#x} overruns the '
                f'{int(buffer_nbytes)} B buffer behind it')
    if hp0 and not hp0_reachable(addr, nbytes):
        return (f'{nbytes} B at {addr:#x} is outside the HP0 windows '
                f'{[(hex(a), hex(b)) for a, b in HP0_WINDOWS]}; arming axis2mm '
                'with it would DECERR with no visible symptom')
    return None


# The status bits that say axis2mm has stopped touching memory. Deliberately
# not the driver's full `ready`, which also demands r_err clear: an abort sets
# r_err, and clearing it is what you do *after* the core has gone quiet.
AXIS2MM_QUIESCENT_BITS = ('r_busy', 'aborting')

# A hierarchy is replaced whenever an overlay is rebound, including the
# ``download=False`` path. It therefore cannot own the last reference to a
# buffer that a stuck axis2mm may still be writing. Keep those references at
# module (process) lifetime and discard them only after a caller confirms that
# the PL was really reconfigured, which resets every DMA master.
_STUCK_CAPTURE_BUFFERS = []


def release_stuck_buffers_after_reconfigure():
    """Release buffers retained for a DMA made harmless by PL reconfiguration.

    Call this only after a successful, real overlay download. Merely binding
    a new hierarchy with ``download=False`` does not stop the old axis2mm.
    Returns the number of retained references released, for logging/tests.
    """
    count = len(_STUCK_CAPTURE_BUFFERS)
    _STUCK_CAPTURE_BUFFERS.clear()
    return count


def retained_stuck_buffer_count():
    """Number of process-lifetime capture buffers awaiting reconfiguration."""
    return len(_STUCK_CAPTURE_BUFFERS)


def axis2mm_quiesced(status):
    """True if axis2mm has stopped writing, given a decoded cmd_ctrl_reg.

    abort() is a bare register write that returns long before the core is
    idle, so a buffer freed on the strength of that write alone can still be
    written to. Both r_busy and aborting must read back clear.
    """
    return not any(bool(status[k]) for k in AXIS2MM_QUIESCENT_BITS)


class CaptureBufferRelease:
    """PYNQ-free axis2mm settle/release logic for capture hierarchies.

    The hardware-facing capture module cannot be imported off-board. Keeping
    this small mixin here lets tests exercise the actual register ordering and
    failure behavior used by ``CaptureHierarchy`` with a scripted register
    file, without pretending an AST spelling check proves runtime behavior.
    """

    def _settle_axis2mm(self, timeout=0.5):
        """Abort, wait for quiescence, then clear the latched error."""
        self.axis2mm.abort()
        deadline = time.time() + timeout
        while True:
            if axis2mm_quiesced(self.axis2mm.cmd_ctrl_reg):
                self.axis2mm.clear_error()
                return True
            if time.time() > deadline:
                return False
            time.sleep(0.0005)

    def _retain_stuck_buffer(self, capture_buffer):
        """Keep exactly one process-lifetime reference to an unsafe buffer."""
        if not any(candidate is capture_buffer
                   for candidate in _STUCK_CAPTURE_BUFFERS):
            _STUCK_CAPTURE_BUFFERS.append(capture_buffer)

    def _release(self, capture_buffer, writing, what, abort_timeout=0.5):
        """Release a safe buffer; retain it if cleanup cannot prove safety.

        This method is called from ``finally`` blocks. Consequently every
        ordinary cleanup failure is caught here so it cannot replace the
        capture exception already in flight. If the DMA state is uncertain,
        a real reference is retained until process exit; otherwise
        ``PynqBuffer.__del__`` would silently free the CMA pages anyway.

        Returns True only when ``freebuffer`` completed successfully.
        """
        logger = None
        try:
            logger = getLogger(self.__class__.__module__)
            if writing and not self._settle_axis2mm(abort_timeout):
                self._retain_stuck_buffer(capture_buffer)
                logger.error(
                    f'axis2mm is still not quiet {abort_timeout:.2f} s after aborting the '
                    f'{what} (status {self.axis2mm.cmd_ctrl_reg}); leaking the '
                    f'{capture_buffer.nbytes} B buffer at '
                    f'{capture_buffer.device_address:#x} rather than returning memory a '
                    f'live DMA can still write to. Restart to reclaim it.')
                return False
            capture_buffer.freebuffer()
            return True
        except Exception as cleanup_failure:
            # Retain before returning even if abort/status/clear/freebuffer
            # failed. Logging is best-effort too: cleanup must not replace the
            # exception whose finally block brought us here.
            try:
                self._retain_stuck_buffer(capture_buffer)
            except Exception:
                pass
            try:
                if logger is None:
                    logger = getLogger(self.__class__.__module__)
                logger.error(
                    f'Failed to release the {what} buffer; retaining it because '
                    f'the DMA state is uncertain. Restart to reclaim it. Original '
                    f'cleanup failure: {cleanup_failure!r}',
                    exc_info=(type(cleanup_failure), cleanup_failure,
                              cleanup_failure.__traceback__))
            except Exception:
                pass
            return False
        except BaseException:
            # KeyboardInterrupt/SystemExit must retain the buffer but must not
            # be converted into an ordinary cleanup failure. A caller may be
            # interrupted between abort and confirmed quiescence, when freeing
            # CMA is still unsafe.
            try:
                self._retain_stuck_buffer(capture_buffer)
            except Exception:
                pass
            raise


N_IQ_GROUPS = 256

PHOTON_DTYPE = np.dtype([('time', np.uint64), ('phase', np.int16), ('id', np.uint16)])

def unpack_photons(x, out=None, n=0):
    """
    Unpack packed photons, optionally accumulating them into an existing output array

    Args:
        x: an array of packed photons
        out: optional, an array of type PhotonMAXI.PHOTON_DTYPE with the shape of x, but see n
        n: optional, an index into out to insert unpacked photons, the first axis is used if >1d an
            IndexError is raised if there is insufficient space.

    Returns: the unpacked photon array

    """
    if out is None:
        n = 0
    elif x.shape[0] + n > out.shape[0]:
        raise IndexError('Output array is too small')

    ret = np.zeros(x.shape, dtype=PHOTON_DTYPE) if out is None else out
    sl = slice(n, n + x.shape[0])
    ret['phase'][sl] = x & 0xffff
    ret['time'][sl] = x >> 28
    ret['id'][sl] = (x >> 16) & 0xfff
    return ret


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


def print_plstatus():
    from pynq import PL
    print(f"PL Bitfile: {PL.bitfile_name}\nPL Timestamp: {PL.timestamp}\n")


class DummyOverlay:
    class DummyBuffer(np.ndarray):
        def freebuffer(self):
            pass

    class DummyCap:
        @staticmethod
        def capture(csize, *args, **kwargs):
            n = csize * 2
            return np.random.uniform(low=-10000, high=10000, size=n).astype(np.int16).view(DummyOverlay.DummyBuffer)

        @staticmethod
        def ready():
            return True

    def __init__(self, bitstream, *args, **kwargs):
        self.capture = DummyOverlay.DummyCap()
