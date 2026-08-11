import time
from logging import getLogger
import numpy as np

try:
    from pynq import DefaultIP
except ImportError:  # off-board: importable for unit tests, unusable
    DefaultIP = object


class AxisFIFO(DefaultIP):
    """Support for an AXI FIFO without cut-through support, could be enhanced"""
    bindto = ['xilinx.com:ip:axi_fifo_mm_s:4.2']

    def __init__(self, description):
        super().__init__(description=description)
        self.length = 512

    def reset_tx_fifo(self):
        self.register_map.TDFR = 0x000000A5
        while not self.register_map.ISR.TRC:
            print('Waiting on tx reset complete...')
            time.sleep(1)

    def tx(self, data, destination=0, last_bytes=4, wait=True, check_vacancy=True):
        """
        Data must be an array of uint32

        The AXI FIFO writes the samples as written, so you can't pack pairs of 16 into 32 unless you have a stream data
        width converter and enable TKEEP.
        """
        if check_vacancy and data.size > self.tx_vacancy:
            raise ValueError('Insufficient room in fifo for data')

        getLogger(__name__).debug(f'ISR at TX start: {repr(self.register_map.ISR)}')
        self.register_map.ISR = 0xFFFFFFFF  # Write to clear reset done interrupt bits
        self.register_map.IER.TPOE = 1  # Interrupt if we try to load too much data (should not be possible)
        self.register_map.IER.TSE = 1  # Interrupt on transmit size errors
        self.register_map.IER.TCE = 1  # Enable transmit complete interrupt
        self.register_map.TDR.TDEST = destination  # Transmit Destination address

        for x in data:
            self.mmio.write(self.register_map.TDFD.address, int(x))  # Write value

        self.register_map.TLR.TXL = (data.size - 1) * 4 + last_bytes
        if wait:
            from ..interrupts import ThreadedPLInterruptManager
            _, event = ThreadedPLInterruptManager.get_monitor(self, id=repr(self)+'tx')
            event.wait()
            event.clear()

    def tx_chunked(self, data, destination=0, last_bytes=4, chunk_words=256,
                   timeout_s=2.0):
        """Send one logical packet as multiple FIFO frames, polling for room.

        For packets larger than the FIFO's usable vacancy (depth - 4 words,
        PG080) a single ``tx`` can never succeed: the 512-deep TX FIFO tops
        out at 508 words of vacancy, and the v3 FIR commit vector is 512
        words. Each slice goes out as its own frame (its own TLR write, so
        its own TLAST), which requires the consumer to treat back-to-back
        frames on the TDEST as one vector. Vacancy is polled between frames
        instead of using the transmit-complete interrupt so this does not
        depend on the interrupt manager being live.
        """
        data = np.asarray(data)
        for start in range(0, data.size, chunk_words):
            piece = data[start:start + chunk_words]
            deadline = time.time() + timeout_s
            while self.tx_vacancy < piece.size:
                if time.time() > deadline:
                    raise TimeoutError(
                        f'TX FIFO did not drain to {piece.size} words within '
                        f'{timeout_s} s; is the stream consumer clocked?')
                time.sleep(0.0005)
            piece_last = last_bytes if start + chunk_words >= data.size else 4
            self.tx(piece, destination=destination, last_bytes=piece_last,
                    wait=False)

    def rx(self):
        """Pull all the data out of the FIFO"""
        if not self.register_map.ISR.RC:  # receive is complete
            return None
        self.register_map.ISR = 0xFFFFFFFF  # Write to clear reset done interrupt bits
        getLogger(__name__).debug(f'ISR at RX start: {repr(self.register_map.ISR)}')

        addr = self.register_map.RDFD.address
        occ = self.rx_occupancy
        data = []
        for _ in range(occ):
            data.append(self.mmio.read(addr))
        occ = self.rx_occupancy
        for _ in range(occ):
            data.append(self.mmio.read(addr))
        if self.rx_occupancy:
            getLogger()
        return np.array(data)

    def powerup(self):
        assert self.register_map.ISR == 0x01D00000  # Read interrupt status register (indicates transmit reset complete
        # and receive reset complete)
        self.register_map.ISR = 0xFFFFFFFF  # Write to clear reset done interrupt bits

    @property
    def tx_vacancy(self):
        return self.register_map.TDFV.Vacancy

    @property
    def rx_occupancy(self):
        return self.register_map.RDFO.Occupancy
