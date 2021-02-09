from pynq import DefaultIP, DefaultHierarchy, allocate
import numpy as np
from fpbinary import FpBinary, OverflowEnum, RoundingEnum
from logging import getLogger
import time
import bitstruct


PL_DDR4_ADDR = 0x0500000000
N_IQ_GROUPS = 256
FP16_15 = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x).__index__()
FP32_8 = lambda x: FpBinary(int_bits=32 - 9, frac_bits=8, signed=True, value=x)


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


class PhasematchDriver(DefaultHierarchy):
    N_TEMPLATE_TAPS = 30
    N_RES = 2048
    N_RES_P_LANE = 512
    N_LANES = 4
    MAX_COEFF_VALUE = 127  # 16 bits, 1 sign, 8 fractional

    def __init__(self, description):
        super().__init__(description)
        self.fifo = self.axi_fifo_mm_s_0

    @staticmethod
    def hierarchy(description):
        for k in ('axi_fifo_mm_s_0',):
            if k not in description['ip']:
                return False
        return True

    @staticmethod
    def vet_coeffs(coeffs):
        if coeffs.size != PhasematchDriver.N_TEMPLATE_TAPS:
            raise ValueError('Incorrect number of taps')
        if max(abs(coeffs)) > PhasematchDriver.MAX_COEFF_VALUE:
            raise ValueError(f'Coefficients must be <= {PhasematchDriver.MAX_COEFF_VALUE}')

    @staticmethod
    def vet_res_id(res_id):
        if 0 > res_id or res_id >= PhasematchDriver.N_RES:
            raise ValueError(f'resID must be in [0-{PhasematchDriver.N_RES}]')

    @staticmethod
    def reorder_coeffs(coeffs):
        """convert taps to order needed by a reload packet"""
        PhasematchDriver.vet_coeffs(coeffs)
        return coeffs[::-1]  # see coefficient reload tab for order in block design

    def load_coeff(self, res_id, coeffs):
        """

        A reload packet consists of the coefficients and the coefficient set number

        See block diagram for layout. Resonators assigned to lanes 0-3 in consecutive sets of 512.

        FIRs have one reload slot and are in "on vector" update mode.

        See pg149 pg 18
        """
        self.vet_res_id(res_id)
        lane = res_id // PhasematchDriver.N_RES_P_LANE
        reload_packet = np.zeros(coeffs.size + 1, dtype=np.uint16)
        reload_packet[0] = res_id % PhasematchDriver.N_RES_P_LANE
        reload_packet[1:] = [FP32_8(c).__index__() for c in PhasematchDriver.reorder_coeffs(coeffs)]

        cfg_packet = np.arange(PhasematchDriver.N_RES_P_LANE, dtype=np.uint16)

        reload_packet = pack16_to_32(reload_packet)
        cfg_packet = pack16_to_32(cfg_packet)

        self.fifo.tx(reload_packet, destination=lane * 2, last_bytes=2)  # reload channels are 0,2,4,6
        self.fifo.tx(cfg_packet, destination=2 * lane + 1)  # Send a config packet to trigger the reload

    def load_coeff_sets(self, coeff_sets):
        for res in range(PhasematchDriver.N_RES):
            self.load_coeff(res, coeff_sets[res])


class MCMM2SBufferDescriptor:
    """
    next pointer 32bits
        0:5 Must be 0, descriptors must be 16-word aligned that is, 0x00, 0x40, 0x80, and so forth. Any other
            alignment has undefined results.
        6:31 value must be 16word aligned (e.g multiple of 0x40)
    next pointer upper 32
    buffer addr 32bits
        - must be data width aligned if core lacks Buffer Realignment Engine (ours do)
    buffer addr upper 32
    reserved 32bits set to 0
    control 32bits
        0:25 Buffer Length Indicates the size in bytes of the transfer buffer. This value indicates the amount of
            bytes to transmit out on the MM2S stream. The usable width of buffer length is specified by the
            parameter Width of Buffer Length Register in the block design. A maximum of 67,108,863 bytes can be
            described by this field. This value should be an integer multiple of the AXI4-Stream data width;
            however it can have any value if the TX EOF bit is set.
        26:29 Reserved
        30 TX EoF End of Frame. Flag indicating the last buffer to be processed. Set this flag to indicate to AXI
            MCDMA that this descriptor describes the end of the packet. The buffer associated with this descriptor is
            transmitted last and results in TLAST assertion.
        31 TX SoF Start of Frame. Flag indicating the first buffer to be processed. Set this flag to indicate to the
            AXI MCDMA that this descriptor describes the start of the packet. The buffer associated with this descriptor
            is transmitted first.
    control sideband 32bits
        0:15 TUSER This field contains the value of TUSER to be presented on the last data beat of the packet that is,
            the beat that has TLAST.
        16:23 Reserved.
        24:31 TID This field contains the value of TID to be presented on the AXI4-Stream interface. The value of TID
            remains the same throughout the packet length. Care should be taken to ensure that this value remains
            constant in the descriptor chain for a packet (e.g. between TLAST).
    status 32bits (set by core)
        0:25 Transferred Bytes Indicates the size in bytes of the actual data transferred for this descriptor. This
            value indicates the amount of bytes to transmit out on MM2S stream. This value should match the Control
            Buffer Length field. The usable width of Transferred Bytes is specified by the parameter Width of Buffer.
        26:27 Reserved
        28 DMAIntErr MCDMA Internal Error. Internal Error detected by AXI DataMover. This error can occur if a 0
            length bytes to transfer is fed to the AXI DataMover. This only happens if the Buffer Length specified in
            the fetched descriptor is set to 0. This error can also be caused if there is an under-run or
            over-run condition. This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is
            set to 0, and when the engine has completely shut down, the MCDMASR.Halted bit is set to 1.
        29 DMASlvErr DMA Slave Error. Slave Error detected by primary AXI DataMover. This error occurs if the slave
            read from the Memory Map interface issues a Slave Error. This error condition causes the AXI MCDMA to
            halt gracefully. The MCDMACR.RS bit is set to 0, and when the engine has completely shut down, the
            MCDMASR.Halted bit is set to 1.
        30 DMADecErr DMA Decode Error. Decode Error detected by primary AXI DataMover. This error occurs if the
            Descriptor Buffer Address points to an invalid address. This error condition causes the AXI MCDMA to halt
            gracefully. The MCDMACR.RS bit is set to 0, and when the engine has completely shut down,
            the MCDMASR.Halted bit is set to 1.
        31 Complete This indicates that the MCDMA engine has completed the transfer as described by the
            associated descriptor. The MCDMA Engine sets this bit to 1 when the transfer is completed. The software can
            manipulate any descriptor with the Completed bit set to 1. If a descriptor is fetched with this bit set to
            1, the descriptor is considered a stale descriptor. A SGIntErr is flagged and the AXI MCDMA engine halts.
    User0 - User4 32bits each
        User application fields 0 to 4. Specifies user-specific application data. When Status Control Stream is
        enabled, the Application (APP) fields of the Start of Frame (SOF) Descriptor are transmitted to the AXI
        Control Stream. For other MM2S descriptors with SOF = 0, the APP fields are fetched but ignored. See PG288 for
        more information.
    """
    pass


class MCS2MMBufferDescriptor:
    """
    S2MM BD:
    next pointer 32bits
        0:5 Must be 0, descriptors must be 16-word aligned that is, 0x00, 0x40, 0x80, and so forth. Any other
            alignment has undefined results.
        6:31 value must be 16word aligned (e.g multiple of 0x40)
    next pointer upper 32
    buffer addr 32bits
        - must be data width aligned if core lacks Buffer Realignment Engine (ours do)
    buffer addr upper 32
    reserved 32bits set to 0
    control 32bits
        0-25 This value indicates the amount of space in bytes available for receiving data in an S2MM stream.
            The usable width of buffer length is specified by the parameter Width of Buffer Length Register
            (c_sg_length_width). A maximum of 67,108,863 bytes of transfer can be described by this field. This
            value should be an integer multiple of AXI4-Stream data width. Note: The total buffer space in the S2MM
            descriptor chain (that is, the sum of buffer length values for each descriptor in a chain) must be, at a
            minimum, capable of holding the maximum receive packet size. Undefined results occur if a packet larger
            than the defined buffer space is received. Setting the Buffer Length Register Width smaller than 26
            reduces FPGA resource utilization.
        26:31 Reserved
    status 32bits (set by core)
        0:25 Transferred Bytes This value indicates the amount of data received and stored in the buffer described by
            this descriptor. This might or might not match the buffer length. For example, if this descriptor
            indicates a buffer length of 1,024 bytes but only 50 bytes were received and stored in the buffer,
            then the Transferred Bytes field indicates 0x32. The entire receive packet length can be determined by
            adding the Transferred Byte values from each descriptor from the RXSOF descriptor to the Receive End of
            Frame (RXEOF) descriptor. The usable width of Transferred Bytes is specified by the parameter
            Width of Buffer Length Register (c_sg_length_width) in the block design. A maximum of 67,108,863 bytes of
            transfer can be described by this field.
        26 RXEOF End of Frame. Flag indicating buffer holds the last part of packet. This bit is set by AXI MCDMA to
            indicate that the buffer associated with this descriptor contains the end of the packet. User Application
            data sent through the status stream input is stored in APP0 to APP4 of the RXEOF descriptor when the
            Control/Status Stream is enabled.
        27 RXSOF Start of Frame. Flag indicating buffer holds first part of packet. This bit is set by AXI MCDMA to
            indicate that the buffer associated with this descriptor contains the start of the packet.
        28 DMAIntErr MCDMA Internal Error. Internal Error detected by AXI DataMover. This error can occur if a 0
            length bytes to transfer is fed to the AXI DataMover. This only happens if the Buffer Length specified in
            the fetched descriptor is set to 0. This error can also be caused if there is an under-run or
            over-run condition. This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is
            set to 0, and when the engine has completely shut down, the MCDMASR.Halted bit is set to 1.
        29 DMASlvErr DMA Slave Error. Slave Error detected by primary AXI DataMover. This error occurs if the slave
            read from the Memory Map interface issues a Slave Error. This error condition causes the AXI MCDMA to
            halt gracefully. The MCDMACR.RS bit is set to 0, and when the engine has completely shut down, the
            MCDMASR.Halted bit is set to 1.
        30 DMADecErr DMA Decode Error. Decode Error detected by primary AXI DataMover. This error occurs if the
            Descriptor Buffer Address points to an invalid address. This error condition causes the AXI MCDMA to halt
            gracefully. The MCDMACR.RS bit is set to 0, and when the engine has completely shut down,
            the MCDMASR.Halted bit is set to 1.
        31 Complete This indicates that the MCDMA engine has completed the transfer as described by the
            associated descriptor. The MCDMA Engine sets this bit to 1 when the transfer is completed. The software can
            manipulate any descriptor with the Completed bit set to 1. If a descriptor is fetched with this bit set to
            1, the descriptor is considered a stale descriptor. A SGIntErr is flagged and the AXI MCDMA engine halts.
    sideband 32bits (set by core)
        0:15 TUSER This field contains the value of TUSER present on the last data beat of the packet that is,
            the beat that has TLAST.
        16:19 TDEST This field contains the value of TDEST.
        20:23 Reserved.
        24:31 TID This field contains the value of TID present on the AXI4-Stream input. It is expected that this
            value should remain constant throughout the packet length.
    User0 - User3 32bits each
        When Status/Control Stream is enabled, the status data received on the AXI Status Stream is stored into the
        APP fields of the End of Frame (EOF) Descriptor. For other S2MM descriptors with EOF = 0, the APP fields are
        set to zero by the Scatter Gather Engine.
    User4 32bits
        User Application field 4 and Receive Byte Length. If Use RxLength In Status Stream is not enabled, this field
        functions identically to APP0 to APP3 in that the status data received on the AXI Status Stream is stored into
        the APP4 field of the End of Frame (EOF) Descriptor. This field has a dual purpose when Use RxLength in
        Status Stream is enabled. The first least significant bits specified in the Buffer Length Register Width
        (up to a maximum of 16) specify the total number of receive bytes for a packet that were received on the
        S2MM primary data stream. Second, the remaining most significant bits are User Application data.
    """
    bd = bitstruct.compile('>u64 u64 p32'
                           'p6 u26'
                           'b1b1 b1b1 b1b1 u26'
                           'u8 p4 u4 u16'
                           'u32 u32 u32 u32 u32', names=['next', 'buff', 'control.buff_len',
                                                         'status.complete', 'status.dma_decode_err',
                                                         'status.dma_slave_err', 'status.dma_internal_err',
                                                         'status.rx_sof', 'status.rx_eof',
                                                         'status.xfer_len',
                                                         'sideband.tid', 'sideband.tdest', 'sideband.tuser', 'app0',
                                                         'app1', 'app2', 'app3', 'app4'])

    def __init__(self, next_addr, buf_addr, length):
        """
        Note that this may generate an illegal BD:
            next must be in multiples of 0x40
            Buf_addr mod stream_width_bytes must be 0. E.g. for a 64 byte wide stream it must be in multiples of 0x40
            length must be representable by the width of the core's Buffer Length Register
        """
        # NB Changing the hard coded defaults will set non-zero initial values for BD properties that are set by the
        # MCDMA core, while this probably has no effect it may make it harder to debug what comes back.
        # setting status.complete=True would initialize the BD as stale.
        byt = self.bd.pack({'next': next_addr, 'buff': buf_addr, 'control.buff_len': length,
                            'status.xfer_len': 0, 'status.rx_eof': False, 'status.rx_sof': False,
                            'status.dma_internal_err': False, 'status.dma_slave_err': False,
                            'status.dma_decode_err': False, 'status.complete': False, 'sideband.tuser': 0,
                            'sideband.tdest': 0, 'sideband.tid': 0, 'app0': 0, 'app1': 0, 'app2': 0, 'app3': 0,
                            'app4': 0})
        self._bytes = byt

    @staticmethod
    def from_dict(d):
        ret = MCS2MMBufferDescriptor(0, 0, 0)
        ret._bytes = ret.bd.pack(d)
        return ret

    @staticmethod
    def array_to_dict(array):
        return {'next': np.uint64(array[0]) | (np.uint64(array[1]) << np.uint64(32)),
                'buff': np.uint64(array[2]) | (np.uint64(array[3]) << np.uint64(32)),
                'control.buff_len': array[5] & ((1 << 26) - 1),
                'status.xfer_len': array[6] & ((1 << 26) - 1),
                'status.rx_eof': bool(array[6] & (1 << 26)), 'status.rx_sof': bool(array[6] & (1 << 27)),
                'status.dma_internal_err': bool(array[6] & (1 << 28)),
                'status.dma_slave_err': bool(array[6] & (1 << 29)),
                'status.dma_decode_err': bool(array[6] & (1 << 30)), 'status.complete': bool(array[6] & (1 << 31)),
                'sideband.tuser': array[7] & 0xffff, 'sideband.tdest': (array[7] >> 16) & 0xf,
                'sideband.tid': (array[7] >> 24), 'app0': array[8],
                'app1': array[9], 'app2': array[10], 'app3': array[11], 'app4': array[12]}

    @staticmethod
    def from_array(array):
        if array.size < 13:
            raise ValueError('Insufficient data, 13 words required')
        ret = MCS2MMBufferDescriptor(0, 0, 0)
        ret._bytes = ret.bd.pack(ret.array_to_dict(array))
        return ret

    @property
    def array(self):
        """Return 16 words with the contents of the buffer descriptor."""
        words = [int(self._bytes[i * 4:(i + 1) * 4].hex(), 16) for i in range(13)] + [0] * 3
        words[:2] = words[:2][::-1]  # gotta swap so that the high word goes into the right register
        words[2:4] = words[2:4][::-1]
        return np.array(words, dtype=np.uint32)


class MCS2MMBufferChain:
    def __init__(self, n, buffer_size=8192, contiguous=True, zero_buffers=True, mmio=None):
        if buffer_size % 0x40:
            raise ValueError('Buffer size must be a multiple of the stream width (0x40)')
        self._chain = allocate(16 * n, dtype=np.uint32)  # TODO ensure starting address is on a 0x40 boundary
        if self._chain.device_address % 0x40:
            raise ValueError('Chain does not start on a 0x40 address boundary')
        # if contiguous:
        #     self.contiguous = True
        #     if mmio is None:
        #         # TODO ensure starting address is on a stream width (here 0x40) boundary as DRE is disabled
        #         # TODO allocate in the PL DDR4
        #         raise NotImplementedError
        #         buff = allocate(buffer_size * n, np.uint8)
        #         self._buffers = [buff]
        #         self.bd_addr = [buff.device_address + buffer_size * i for i in range(n)]
        #     else:
        self.contiguous = True
        self._buffers = [mmio.array]
        self.bd_addr = [mmio.base_addr + buffer_size * i for i in range(n)]
        self.buffer_sizes = [buffer_size]*n
        # else:
        #     # TODO ensure starting address is on a stream width (here 0x40) boundary as DRE is disabled
        #     # TODO allocate in the PL DDR4
        #     raise NotImplementedError
        #     self._buffers = [allocate(buffer_size, np.uint8) for i in range(n)]
        #     self.bd_addr = [b.device_address for b in self._buffers]
        #     self.contiguous = False

        # Zero the buffers
        if zero_buffers:
            for b in self._buffers:
                b[:] = 0

        # Build chain
        for i, buf_addr in zip(range(n), self.bd_addr):
            descriptor = MCS2MMBufferDescriptor(self._chain.device_address + 0x40 * (i + 1), buf_addr, buffer_size)
            self._chain[i * 16:(i + 1) * 16] = descriptor.array

    @property
    def chain_length(self):
        """How many links are in the chain"""
        return len(self.bd_addr)

    @property
    def head_addr(self):
        """ Return the memory address of the first descriptor in the chain"""
        return self._chain.device_address

    @property
    def tail_addr(self):
        """ Return the memory address of the last descriptor in the chain"""
        return self._chain.device_address+16*(self.chain_length-1)*4

    def descriptor_dict(self, i):
        """Return a dict of link i of the chain"""
        if not 0 <= i < len(self.bd_addr):
            raise ValueError(f'Chain is of length {len(self.bd_addr)}')
        return MCS2MMBufferDescriptor.array_to_dict(self._chain[16*i:16*(i+1)])

    def descriptor_data(self, i):
        """Retrieve the data from the descriptor"""
        if not 0 <= i < len(self.bd_addr):
            raise ValueError(f'Chain is of length {len(self.bd_addr)}')
        d=self.descriptor_dict(i)[''] # TODO access directly for speed?
        if not d['complete']:
            raise ValueError('Descriptor not complete')
        n = d['status.xfer_len']
        return [self._buffers[0][i] for i in range(n // 4)]

    def __del__(self):
        for b in self._buffers:
            b.freebuffer()


class MCDMA(DefaultIP):
    bindto = ['xilinx.com:ip:axi_mcdma:1.1']
    ADDR_S2MM_CONTROL = 0x500
    ADDR_S2MM_COMMON_STATUS = 0x504
    ADDR_S2MM_CHAN_ENABLE = 0x508
    ADDR_S2MM_INPROGRESS = 0x50C
    ADDR_S2MM_ERROR = 0x510
    ADDR_S2MM_ALLPACKETDROP = 0x514
    ADDR_S2MM_CHAN_COMPLETED = 0x518
    ADDR_S2MM_INTMON = 0x520

    ADDR_S2MM_CHAN_CONTROL = {i: 0x540 + i * 0x40 for i in range(16)}
    ADDR_S2MM_CHAN_STATUS = {i: 0x544 + i * 0x40 for i in range(16)}
    ADDR_S2MM_CURDESC = {i: 0x548 + i * 0x40 for i in range(16)}
    ADDR_S2MM_TAILDESC = {i: 0x550 + i * 0x40 for i in range(16)}
    ADDR_S2MM_PACKETDROP = {i: 0x558 + i * 0x40 for i in range(16)}
    ADDR_S2MM_PACKETPROC = {i: 0x55C + i * 0x40 for i in range(16)}

    S2MM_ERROR_REG = bitstruct.compile('>p25 b1b1b1 p1 b1b1b1',
                                       names=['sg_decode', 'sg_slave', 'sg_internal', 'dma_decode', 'dma_slave',
                                              'dma_internal'])
    S2MM_CHAN_CTRL_REG = bitstruct.compile('>u8 u8 u8 b1b1 b1b1b1 p2 b1',
                                           names=['delay_timeout', 'bdcomplete_threshold', 'drop_threshold', 'error',
                                                  'timeout', 'bd_complete', 'packet_drop', 'otherchan_error',
                                                  'runstop'])
    S2MM_CHAN_STAT_REG = bitstruct.compile('>u8 u8 u8 b1b1b1b1 b1p1b1b1',
                                           names=['delay_timeout', 'bdcomplete_threshold', 'drop_threshold',
                                                  'error', 'timeout', 'bd_complete', 'packet_drop',
                                                  'otherchan_error', 'bd_shortfall', 'idle'])
    S2MM_CHAN_STAT_REG = bitstruct.compile('>u8 u8 u8 b1b1b1b1 b1p1b1b1',
                                           names=['delay_timeout', 'bdcomplete_threshold', 'drop_threshold',
                                                  'error', 'timeout', 'bd_complete', 'packet_drop',
                                                  'otherchan_error', 'bd_shortfall', 'idle'])

    S2MM_INT_SOURCE = bitstruct.compile('>p16'+'b1'*16, names=[f'ch{i}' for i in range(16)])
    S2MM_CHAN_COMPLETE = bitstruct.compile('>p16'+'b1'*16, names=[f'ch{i}' for i in range(16)])
    S2MM_STAT_REG = bitstruct.compile('>p30b1b1', names=['idle', 'halted'])

    def s2mm_current_descriptor(self, chan):
        return self.read(MCDMA.ADDR_S2MM_CURDESC[chan]) | (self.read(MCDMA.ADDR_S2MM_CURDESC[chan]+4)<<32)

    def s2mm_status(self):
        """
        32:2 Reserved
        1 Idle  MCDMA S2MM Idle. Indicates the state of AXI MCDMA operations. When IDLE, indicates the SG Engine has
        reached the tail pointer for all the channels and all queued descriptors have been processed.
        Writing to the tail pointer register to any channel automatically restarts MCDMA operations.
        0 Halted (MCDMA.Halted)  MCDMA Halted. Indicates the run/stop state of the MCDMA.
        There can be a lag of time between when MCDMACR.RS = 0 and when MCDMASR.Halted = 1.
        """
        return MCDMA.S2MM_STAT_REG.unpack(self.read(MCDMA.ADDR_S2MM_COMMON_STATUS).to_bytes(4,'big'))

    def s2mm_channel_status(self, chan):
        """
        31:24 IRQ DELAY Status Interrupt delay time Status. Indicates current interrupt delay time value.
        23:16 IRQ Threshold Status Interrupt Threshold Status. Indicates current interrupt threshold value.
        15:8 IRQ Packet Drop Status Interrupt Packet Drop Threshold. Indicates current interrupt threshold value.
        7 Err Irq Interrupt on Error. When set to 1, indicates an interrupt event was generated on an error.
        If the corresponding bit in Control register is enabled (Err_IrqEn = 1), an interrupt out is
        generated from the AXI MCDMA. Writing a 1 to this bit clears it.
        6 DlyIrq Interrupt on Delay. When set to 1, indicates an interrupt event was generated on delay timer
        timeout. If the corresponding bit in Control register is enabled (Dly_IrqEn = 1), an interrupt out is
        generated from the AXI MCDMA. Writing a 1 to this bit clears it.
        5 IOC_Irq Interrupt on Complete. When set to 1 an interrupt event was generated on completion of a
        descriptor. This occurs for descriptors with the EOF bit set. If the corresponding bit in the Control
        register is enabled (IOC_IrqEn = 1) and if the interrupt threshold has been met, causes an interrupt out to
        be generated from the AXI MCDMA. Writing a 1 to this bit clears it.
        4 Pktdrop_irq Interrupt on packet drop. When set to 1 indicates an interrupt event was generated on packet
        drop. If the corresponding bit in the Control register is enabled (PktDrp_IrqEn = 1) and if the interrupt
        threshold has been met, causes an interrupt out to be generated from the AXI MCDMA. Writing a 1 to this bit
        clears it.
        3 Err_on_other_ch_irq Interrupt on Error on other channels. When set to 1, indicates an interrupt event was
        generated on an error on other channels. If the corresponding bit in Control register is enabled (
        Err_on_other En = 1), an interrupt out is generated from the AXI MCDMA. Writing a 1 to this bit clears it.
        2 Reserved
        1 BD ShortFall This bit is set when a packet is being processed and the BD queue becomes empty. This means
        that the packet that is being serviced is too large to be accommodated in the BD queue. This scenario leads
        to MCDMA waiting forever for the packet to get completed. To get over this, extend the BD chain and program
        the TD to fetch more BDs. This does not result in a packet drop. This bit can also get set momentarily when
        the MCDMA is servicing the last BD for a channel and accommodates the packet. After the TLAST is
        accommodated in the last BD, this bit is unset.
        0 Idle (Queue Empty) MCDMA Channel Idle. Indicates the SG Engine has reached the tail pointer for the
        associated channel and all queued descriptors have been processed. This means that the BD queue is empty and
        there are no more BDs to process. Writing to the tail pointer register automatically restarts the BD fetch
        operations. If the packet arrives while this bit is set, that packet is dropped.
        """
        return MCDMA.S2MM_CHAN_STAT_REG.unpack(self.read(MCDMA.ADDR_S2MM_CHAN_STATUS[chan]).to_bytes(4, 'big'))

    def s2mm_packets_processed(self, chan):
        """
        31:16 RSVD
        15:0 Packet processed count Reports the number of packets processed for the channel. This counter rolls over \
                to 0 after reaching maximum value.
        """
        return self.read(MCDMA.ADDR_S2MM_PACKETPROC[chan]) & 0xffff

    def s2mm_packets_dropped(self, chan):
        """
        31:16 RSVD
        15:0 Packet drop count Reports the number of packets dropped for the channel. This value increments by 1
        every time a packet is dropped. The counter wraps around after it has reached maximum value. This register is cleared when read.
        """
        return self.read(MCDMA.ADDR_S2MM_PACKETDROP[chan]) & 0xffff

    def s2mm_total_packets_dropped(self):
        """
        Reports th number of packets dropped across all channels. This value increments by 1 every time a packet is
        dropped on any channel. The counter wraps around after it has reached maximum value.
        """
        return self.read(MCDMA.ADDR_S2MM_ALLPACKETDROP)

    def s2mm_inprogress(self):
        """
        31:16 Reserved
        15:0 This is the channel ID that was last serviced. This register has a one-hot value to identify the channel
        that caused the error. A value of 1 corresponds to TDEST = 0, a value of 2 corresponds to TDEST = 1,
        a value of 4 corresponds to TDEST = 2 and so on.
        """
        return _which_one_bit_set(self.read(MCDMA.ADDR_S2MM_INPROGRESS) & 0xffff, 16)

    def s2mm_interrupt_source(self):
        """
        This gives info about the channel(s) that generated the interrupt. This register has a one-hot value to
        identify the channel(s) that generated the interrupt. A value of 1 corresponds to Channel 0, a value of 2
        corresponds to Channel 1, a value of 4 corresponds to Channel 2 and so on. Bits are automatically cleared
        when the corresponding interrupt is cleared in the channel status register.
        """
        #return _which_one_bit_set(self.read(MCDMA.ADDR_S2MM_INTMON) & 0xffff, 16)
        return MCDMA.S2MM_CHAN_COMPLETE.unpack(self.read(MCDMA.ADDR_S2MM_INTMON).to_bytes(4, 'big'))

    def s2mm_channels_completed(self):
        """
        This is the channel that was serviced. This register has a one-hot value to identify the channel that was
        processed. A value of 1 corresponds to TDEST = 0, a value of 2 corresponds to TDEST = 1, a value of 4
        corresponds to TDEST = 2 and so on. This register is cleared on read.
        """
        # return _which_one_bit_set(self.read(MCDMA.ADDR_S2MM_CHAN_COMPLETED) & 0xffff, 16)
        return MCDMA.S2MM_CHAN_COMPLETE.unpack(self.read(MCDMA.ADDR_S2MM_CHAN_COMPLETED).to_bytes(4,'big'))

    def s2mm_error(self):
        """
        31:7 Reserved
        6 SGDecErr Scatter Gather Decode Error. This error occurs if CURDESC_PTR and/or NXTDESC_PTR point to an
        invalid address. This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is set to
        0 and when the engine has completely shut down, the MCDMASR.Halted bit is set to 1. MCDMA Engine halts. This
        error cannot be logged into the descriptor.
        5 SGSlvErr Scatter Gather Slave Error. This error occurs if the slave read from on the Memory Map interface
        issues a Slave Error. This error condition causes the AXI MCDMA to gracefully halt. The MCDMACR.RS bit is
        set to 0, and when the engine has completely shut down, the MCDMASR.Halted bit is set to 1. MCDMA Engine
        halts.This error cannot be logged into the descriptor.
        4 SGIntErr Scatter Gather Internal Error. This error occurs if a descriptor with the Complete bit already
        set is fetched. This indicates to the SG Engine that the descriptor is a tail descriptor. This error
        condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is set to 0, and when the engine has
        completely shut down, the MCDMASR.Halted bit is set to 1. This error cannot be logged into the descriptor.
        3 Reserved
        2 DMA Dec Err MCDMA Decode Error. This error occurs if the address request points to an invalid address.
        This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is set to 0, and when the
        engine has completely shut down, the MCDMASR.Halted bit is set to 1. This bit can be set when such an event
        occurs on any of the channels.
        1 DMA SLv Err MCDMA Slave Error. This error occurs if the slave read from the Memory Map interface issues a
        Slave Error. This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit is set to 0
        and when the engine has completely shut down the MCDMASR.Halted bit is set to 1. This bit can be set when
        such an event occurs on any of the channels.
        0 DMA Intr Err MCDMA Internal Error. This error occurs if the buffer length specified in the fetched
        descriptor is set to 0. Also, when in Scatter Gather Mode and using the status app length field, this error
        occurs when the Status AXI4-Stream packet RxLength field does not match the S2MM packet being received by
        the S_AXIS_S2MM interface. This error condition causes the AXI MCDMA to halt gracefully. The MCDMACR.RS bit
        is set to 0, and when the engine has completely shut down, the MCDMASR.Halted bit is set to 1. This bit is
        set when such an event occurs on any of the channels.
        """
        return MCDMA.S2MM_ERROR_REG.unpack(self.read(MCDMA.ADDR_S2MM_ERROR).to_bytes(4, 'big'))

    @staticmethod
    def gen_s2mm_control_reg(runstop=False, other_chan_errors=False, packet_drop=False, bd_complete=True, timeout=False,
                             error=True, drop_threshold=1, bdcomplete_threshold=1, delay_timeout_us=100,
                             MCDMA_SGCLOCK_MHZ=128):
        """
        other_chan_errors - IRQ on other channel errors
        packet_drop = IRQ on packet drops
        bd_complete - IRQ on completion of buffer descriptors
        timeout - IRQ on no inbound packet timeout (i.e. time from tlast of one to first of next)
        error - IRQ on errors

        0 CHANNEL.RunStop (CH.RS)
        1:2 Reserved
        3 Err_on_other En
        4 Pktdrop_IrqEn
        5 IOC_IrqEn
        6 DlyIrqEn
        7 Err Irq En
        15:8 IRQ Packet Drop Threshold
        23:16 IRQ Threshold
        31:24  IRQ DELAY
        """
        delay_timeout = (delay_timeout_us * MCDMA_SGCLOCK_MHZ) // 125
        if delay_timeout == 0 and timeout:
            getLogger(__name__).warning('Channel delay timeout is quantized in units of 1.25 us (assuming a 100MHz '
                                        'SG clock. Setting to minimum. If disable intended set irq_delay=False')
        if delay_timeout > 255 and timeout:
            getLogger(__name__).warning('Maximum channel delay timeout is 318.75 us (assuming a 100MHz '
                                        'SG clock). Setting to maximum.')
            delay_timeout = 255

        drop_threshold = min(drop_threshold, 255)
        bdcomplete_threshold = min(bdcomplete_threshold, 255)

        reg = MCDMA.S2MM_CHAN_CTRL_REG.pack({'delay_timeout': delay_timeout,
                                             'bdcomplete_threshold': bdcomplete_threshold,
                                             'drop_threshold': drop_threshold, 'error': error, 'timeout': timeout,
                                             'bd_complete': bd_complete, 'packet_drop': packet_drop,
                                             'otherchan_error': other_chan_errors, 'runstop': runstop})
        return reg

    def __init__(self, description):
        super().__init__(description=description)
        self.bd_chains = None

    def config_recieve(self, n_buffers=2, buffer_size_bytes=8192, channels=tuple(range(16))):
        """
        1. Enable the required channels. (can be also done after step 6).
        2. Program the CD registers of the channels. If the IP is configured for address_width > 32 (c_addr_width > 32),
            then program the corresponding MSB registers.
        3. Program the CHANNEL.RS bit of channel control registers. Note: At this point the CDs cannot be re-programmed.
        4. Start the MCDMA by programming MCDMA.RS bit (0x500).
        5. Program the interrupt thresholds, Enable Interrupts.
        6. Program the TD register of channels. If the IP is configured for address_width > 32 (c_addr_width > 32), then
           program the corresponding MSB registers. Programming the TDs of a particular channel triggers the fetching
            of the BDs for the respective channels.
        """
        # build buffer descriptor chains
        self.bd_chains = {c: MCS2MMBufferChain(n=n_buffers, buffer_size=buffer_size_bytes, contiguous=True)
                          for c in channels}

        # Enable channels
        # If the MCDMA receives a packet on a disabled channel, the entire packet is dropped by the MCDMA engine.
        # This register does not stop the BD fetch of the channel; it only stops a particular channel from being
        # serviced. This register can be programmed at any time, but it will come into effect only when there is no
        # data present on the S2MM AXI4-Stream interface or on arrival of the next packet.
        # Note: Disabling a channel does not disable its interrupt behavior. Setting a bit to 1 enables the channel
        # for service. Each bit corresponds to a channel. Bit[0] corresponds to channel with TDEST = 0 and so on.

        # 1. Enable the required channels. (can be also done after step 6). If done after step 6 BD fetching will start
        # and the core will be operating however it will drop any inbound packets until the channels are enabled.
        enable = 0
        for c in channels:
            enable |= 1 << c
        self.write(MCDMA.ADDR_S2MM_CHAN_ENABLE, enable)

        # 2. Program the CD registers of the channels.
        for c in channels:
            self.write(MCDMA.ADDR_S2MM_CURDESC[c], self.bd_chains[c].head_addr & 0xffffffff)
            self.write(MCDMA.ADDR_S2MM_CURDESC[c]+4, self.bd_chains[c].head_addr >> 32)

        # 3. Program the CHANNEL.RS bit of channel control registers. After this point the CDs cannot be reprogrammed.
        for c in range(16):
            x = self.read(MCDMA.ADDR_S2MM_CHAN_CONTROL[c])
            x = x | 1 if c in channels else (x | 1) ^ 1  # clear run bits on other channels
            self.write(MCDMA.ADDR_S2MM_CHAN_CONTROL[c], x)

        # 4. Start the MCDMA by programming MCDMA.RS bit.
        self.write(MCDMA.ADDR_S2MM_CONTROL, 1)

        # 5. Program the interrupt thresholds and enable interrupts.
        for c in channels:
            reg = self.gen_s2mm_control_reg(True, other_chan_errors=False, packet_drop=False, bd_complete=True,
                                            timeout=False, error=True, bdcomplete_threshold=2)
            self.write(MCDMA.ADDR_S2MM_CHAN_CONTROL[c], int(reg.hex(), 16))  # writing as bytes leads to corruption

        # 6. Program the TD register of channels. Programming the TDs of a particular channel triggers the fetching of
        # the BDs for the respective channels. The channel pauses when it completes processing for the tail descriptor.
        # it resumes on writing the taildescriptor. If the AXI MCDMA Channel MCDMACR.RS bit is set to 0, a
        # write by the CPU to the TAILDESC_PTR register has no effect except to reposition the pause point.
        for c in channels:
            self.write(MCDMA.ADDR_S2MM_TAILDESC[c], self.bd_chains[c].tail_addr & 0xffffffff)
            self.write(MCDMA.ADDR_S2MM_TAILDESC[c]+4, self.bd_chains[c].tail_addr >> 32)


class AxisFIFO(DefaultIP):
    bindto = ['xilinx.com:ip:axi_fifo_mm_s:4.2']

    def __init__(self, description):
        super().__init__(description=description)
        self.length = 512

    def reset_tx_fifo(self):
        self.register_map.TDFR = 0x000000A5
        while not self.register_map.ISR.TRC:
            print('Waiting on tx reset complete...')
            time.sleep(1)

    def tx(self, data, destination=0, last_bytes=4):
        """Data must be an array of uint32"""
        if data.size > self.tx_vacancy:
            raise ValueError('Insufficient room in fifo for data')

        getLogger(__name__).debug(f'ISR at TX start: {repr(self.register_map.ISR)}')
        self.register_map.ISR = 0xFFFFFFFF  # Write to clear reset done interrupt bits
        self.register_map.IER.TPOE = 1  # Interrupt if we try to load too much data (should not be possible)
        self.register_map.IER.TSE = 1  # Interrupt on transmit size errors
        self.register_map.IER.TC = 1  # Enable transmit complete interrupt
        self.register_map.TDR.TDEST = destination  # Transmit Destination address

        for x in data:
            self.mmio.write(self.register_map.TDFD.address, int(x))  # Write value

        self.register_map.TLR.TXL = ( data.size - 1) * 4 + last_bytes  # Transmit length in bytes, this starts transmission
        self.interrupt.wait()  # wait for the transmit to complete

        getLogger(__name__).debug(f'ISR at TX end: {repr(self.register_map.ISR)}')

    def rx(self):
        """Pull all the data out of the FIFO"""
        if not self.register_map.ISR.RS:  # a recieve is complete
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


class ResonatorDDSIP(DefaultIP):
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


class DACTableAXIM(DefaultIP):
    bindto = ['mazinlab:mkidgen3:dac_table_axim:0.6']

    def __init__(self, description):
        super().__init__(description=description)
        self._buffer=None

    def replay(self, data, tlast=True, tlast_every=256, length=None, start=True, fpgen=lambda x: FP16_15(x).__index()):
        """
        data - an array of complex numbers, nominally 2^18 samples
        tlast - Set to true to emit a tlast pulse every tlast_every transactions
        length - Number of groups of 16 (DAC takes 16/clock) in the data to use.
            e.g. if data[N].reshape(N//16,16). Note that the user channel will count to length-1
        fpgen - set to a function to convert floating point numbers to integers. if None data will be truncated.
        start - start the replay immediately
        """
        # Data has right shape
        if data.size < 2 ** 18:
            getLogger(__name__).warning('Insufficient data, padding with zeros')
            x = np.zeros(2 ** 18, dtype=np.complex64)
            x[:data.size] = data
            data = x

        if data.size > 2 ** 18:
            getLogger(__name__).warning('Data will be truncated to 2^18 samples')
            data = data[:2 ** 18]

        # Process Length
        if length is None:
            length = data.size // 8
        length = int(length)
        if length > 2 ** 15:  # NB 15 as length is in sets of 8
            getLogger(__name__).warning('Clipping replay length to 2^15 sets of 8')
        if length < 1:
            raise ValueError('Replay length must be at least 1')

        # Process tlast_every
        tlast_every = int(tlast_every)

        if tlast and not 1 <= tlast_every <= length:
            raise ValueError('tlast_every must be in [1-length] when tlast is set')
        if not tlast:
            tlast_every = 256  # just assign some value to ensure we didn't get handed garbage

        self._buffer = allocate(2 ** 19, dtype=np.uint16)
        self._buffer[:data.size * 2:2] = [fpgen(x) for x in data.real] if fpgen is not None else data.real
        self._buffer[1:data.size * 2:2] = [fpgen(x) for x in data.imag] if fpgen is not None else data.imag

        self.register_map.a = self._buffer.device_address
        self.register_map.length_V = length
        self.register_map.tlast = bool(tlast)
        self.register_map.replay_length_V = tlast_every
        self.register_map.run = True
        if start:
            self.register_map.CTRL.AP_START = 1

    def stop(self):
        self.register_map.run = False

    def start(self):
        #TODO probably need to check for the case where not idle and run = False (axis stall to completion)
        if self._buffer is None:
            raise RuntimeError('Must call replay first to configure the core')
        self.register_map.CTRL.AP_START = 1


class IQCapture(DefaultIP):
    """
    // control
    // 0x00 : reserved
    // 0x04 : reserved
    // 0x08 : reserved
    // 0x0c : reserved
    // 0x10 : Data signal of keep_V
    //        bit 31~0 - keep_V[31:0] (Read/Write)
    // 0x14 : Data signal of keep_V
    //        bit 31~0 - keep_V[63:32] (Read/Write)
    // 0x18 : Data signal of keep_V
    //        bit 31~0 - keep_V[95:64] (Read/Write)
    // 0x1c : Data signal of keep_V
    //        bit 31~0 - keep_V[127:96] (Read/Write)
    // 0x20 : Data signal of keep_V
    //        bit 31~0 - keep_V[159:128] (Read/Write)
    // 0x24 : Data signal of keep_V
    //        bit 31~0 - keep_V[191:160] (Read/Write)
    // 0x28 : Data signal of keep_V
    //        bit 31~0 - keep_V[223:192] (Read/Write)
    // 0x2c : Data signal of keep_V
    //        bit 31~0 - keep_V[255:224] (Read/Write)
    // 0x30 : reserved
    // 0x34 : Data signal of capturesize_V
    //        bit 31~0 - capturesize_V[31:0] (Read/Write)
    // 0x38 : reserved
    // 0x3c : Data signal of configure
    //        bit 0  - configure[0] (Read/Write)
    //        others - reserved
    // 0x40 : reserved
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
    """
    bindto = ['MazinLab:mkidgen3:iq_capture:0.5']
    ADDR_KEEP = 0x10  # 8 words
    ADDR_CAPTURESIZE = 0x34
    ADDR_CONFIGURE = 0x3c

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, n, groups=tuple(range(N_IQ_GROUPS)), start=True):
        """
        Tell the capture block to forward along n IQ samples of the selected resonator groups.

        n*len(groups) groups will be forwarded for capture

        groups - an iterable of IQ group numbers or 'all'. IQs are processed in 256 groups of 8, so to capture the IQ
        values of IQs 0, 37, and 2047 groups must be set to either 'all' or include (0, 4, 255). Note that this will
        cause IQs 0-7, 32-39, and 2040-2047 to be captured.
        """
        if not 0 < n <= 2**24-1:
            raise ValueError('Invalid capture length')
        keep = np.zeros(8, dtype=np.uint32)
        if isinstance(groups, str):
            n_groups = 256
            if groups.lower() == 'all':
                keep += 0xFFFFFFFF
            else:
                raise ValueError("The only legal string option for groups is 'all'")
        else:
            n_groups = 0
            for g in groups:
                if not 0 <= g < 256:
                    raise ValueError(f'Groups {g} not in range 0-255')
                n_groups += 1
                i = g//32  # (0-7)
                keep[i] |= (1 << (g % 32))  #set the right bit

        self.write(self.ADDR_CAPTURESIZE, int(n*n_groups))
        for i, k in enumerate(keep):
            self.write(self.ADDR_KEEP+0x4*i, int(k))
        if start:
            self.start_capture()

    def halt_capture(self):
        self.write(self.ADDR_CONFIGURE, 1)

    def start_capture(self):
        self.write(self.ADDR_CONFIGURE, 1)
        self.write(self.ADDR_CONFIGURE, 0)


class ADCCapture(DefaultIP):
    """
    // control
    // 0x00 : reserved
    // 0x04 : reserved
    // 0x08 : reserved
    // 0x0c : reserved

    // 0x38 : reserved
    // 0x3c : Data signal of configure
    //        bit 0  - configure[0] (Read/Write)
    //        others - reserved
    // 0x40 : reserved
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
    """
    ADDR_CAPTURESIZE = 0x10
    ADDR_CONFIGURE = 0x18
    bindto = ['MazinLab:mikdgen3:adc_capture:0.5']

    def __init__(self, description):
        super().__init__(description=description)

    def capture(self, n, start=True):
        """
        Tell the capture block to forward along n IQ samples of the selected resonator groups.

        groups - an iterable of IQ group numbers or 'all'. IQs are processed in 256 groups of 8, so to capture the IQ
        values of IQs 0, 37, and 2047 groups must be set to either 'all' or include (0, 4, 255). Note that this will
        cause IQs 0-7, 32-39, and 2040-2047 to be captured.
        """
        self.write(self.ADDR_CAPTURESIZE, n)
        if start:
            self.start_capture()

    def halt_capture(self):
        self.write(self.ADDR_CONFIGURE, 1)

    def start_capture(self):
        self.write(self.ADDR_CONFIGURE, 1)
        self.write(self.ADDR_CONFIGURE, 0)


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
