import time
from logging import getLogger
from pynq import DefaultIP
import numpy as np


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
