import asyncio
from logging import getLogger
import numpy as np
import pynq

from mkidgen3.util import _which_one_bit_set
from pynq import allocate, DefaultIP
import bitstruct
from pynq.lib.dma import _SGDMAChannel

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

    def config_receive(self, n_buffers=2, buffer_size_bytes=8192, channels=tuple(range(16))):
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


class S2MMBufferDescriptor:
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
    reserved 32bits
    reserved 32bits
    control 32bits
        0-25 This value indicates the amount of space in bytes available for receiving data in an S2MM stream.
            The usable width of buffer length is specified by the parameter Width of Buffer Length Register
            (c_sg_length_width). A maximum of 67,108,863 bytes of transfer can be described by this field. This
            value should be an integer multiple of AXI4-Stream data width. Note: The total buffer space in the S2MM
            descriptor chain (that is, the sum of buffer length values for each descriptor in a chain) must be, at a
            minimum, capable of holding the maximum receive packet size. Undefined results occur if a packet larger
            than the defined buffer space is received. Setting the Buffer Length Register Width smaller than 26
            reduces FPGA resource utilization.
        26  End of Frame. Flag indicating the last buffer to be processed. This flag is set by the sw/user to indicate
            to AXI DMA that this descriptor describes the end of the packet. The buffer associated with this
            descriptor is received last.
            0 = Not End of Frame.
            1 = End of Frame.
            This is applicable only when AXI_DMA is configured in Micro mode.
        27 RXSOF  Start of Frame. Flag indicating the first buffer to be processed. This flag is set by the
            sw/user to indicate to AXI DMA that this descriptor describes the start of the packet. The buffer
            associated with this descriptor is received first.
            0 = Not Start of Frame.
            1 = Start of Frame.
            This is applicable only when AXI_DMA is configured in Micro mode.
        28-31 Reserved, should be set to 0
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
    S2MM_STATUS_BD = bitstruct.compile('b1b1 b1b1 b1b1 u26', names=['complete', 'dma_decode_err',
                                                        'dma_slave_err', 'dma_internal_err',
                                                        'rx_sof', 'rx_eof', 'xfer_len'])
    BD = bitstruct.compile('>u64 u64 '
                           'p64'
                           'p4 b1b1 u26'
                           'b1b1 b1b1 b1b1 u26'
                           'u32 u32 u32 u32 u32', names=['next', 'buff',
                                                         'control.rxsof', 'control.rxeof', 'control.buff_len',
                                                         'status.complete', 'status.dma_decode_err',
                                                         'status.dma_slave_err', 'status.dma_internal_err',
                                                         'status.rx_sof', 'status.rx_eof', 'status.xfer_len',
                                                         'app0', 'app1', 'app2', 'app3', 'app4'])

    def __init__(self, next_addr, buf_addr, length, _dict=None):
        """
        Note that this may generate an illegal BD:
            next must be in multiples of 0x40
            Buf_addr mod stream_width_bytes must be 0. E.g. for a 64 byte wide stream it must be in multiples of 0x40
            length must be representable by the width of the core's Buffer Length Register
        """
        # NB Changing the hard coded defaults will set non-zero initial values for BD properties that are set by the
        # DMA core, while this probably has no effect it may make it harder to debug what comes back.
        # setting status.complete=True would initialize the BD as stale.
        defaults = {'next': next_addr, 'buff': buf_addr,
                     'control.rxsof':0, 'control.rxeof':0, 'control.buff_len': length,
                     'status.xfer_len': 0, 'status.rx_eof': False, 'status.rx_sof': False,
                     'status.dma_internal_err': False, 'status.dma_slave_err': False,
                     'status.dma_decode_err': False, 'status.complete': False,
                     'app0': 0, 'app1': 0, 'app2': 0, 'app3': 0, 'app4': 0}
        if _dict:
            defaults.update(_dict)
        self._bytes = self.BD.pack(defaults)

    @staticmethod
    def from_dict(d):
        return S2MMBufferDescriptor(0, 0, 0, _dict=d)

    @staticmethod
    def array_to_dict(array):
        d = {'next': np.uint64(array[0]) | (np.uint64(array[1]) << np.uint64(32)),
             'buff': np.uint64(array[2]) | (np.uint64(array[3]) << np.uint64(32)),
             'control.buff_len': array[6] & ((1 << 26) - 1),
             'status.xfer_len': array[7] & ((1 << 26) - 1),
             'status.rx_eof': bool(array[7] & (1 << 26)), 'status.rx_sof': bool(array[7] & (1 << 27)),
             'status.dma_internal_err': bool(array[7] & (1 << 28)),
             'status.dma_slave_err': bool(array[7] & (1 << 29)),
             'status.dma_decode_err': bool(array[7] & (1 << 30)),
             'status.complete': bool(array[7] & (1 << 31))}
        if array.size>7:
            d.update({'app0': array[8], 'app1': array[9], 'app2': array[10],
                      'app3': array[11], 'app4': array[12]})
        return d

    @staticmethod
    def from_array(array):
        if array.size not in (8, 12, 16):
            raise ValueError('Improper data, 8, 12, or 16 words required')
        return S2MMBufferDescriptor(0, 0, 0,
                                    _dict=S2MMBufferDescriptor.array_to_dict(array[:12]))

    @property
    def array(self):
        """ Return 8 or 12 words with the contents of the buffer descriptor. """
        n=len(self._bytes)//4
        words = [int(self._bytes[i * 4:(i + 1) * 4].hex(), 16) for i in range(n)]
        words[:2] = words[:2][::-1]  # gotta swap so that the high word goes into the right register
        words[2:4] = words[2:4][::-1]
        words.extend([0]*(16-len(words)))
        return np.array(words, dtype=np.uint32)


class S2MMBufferChain:
    def __init__(self, n_buffers, buffer_size, dtype='u8', cyclic=True):
        # if buffer_size % dtype().itemsize:
        #     raise ValueError('Buffer size must be a multiple of the itemsize width (0x40)')
        self._chain = allocate((n_buffers, 16), dtype='u4')
        if self._chain.device_address % 0x40:
            raise ValueError('Chain does not start on a 0x40 address boundary')

        self._buffers = pynq.allocate((n_buffers, buffer_size), dtype=dtype)
        self.chain_length = n_buffers
        item_nbytes= int(dtype[1:])

        self.buf_addr = [self._buffers.device_address + i*buffer_size*item_nbytes for i in range(n_buffers)]
        self.buffer_size = buffer_size
        self._buffers[:] = 0
        self.cyclic=cyclic

        self.chain_addr = [self._chain.device_address + 0x40 * i for i in range(n_buffers)]

        # Build chain
        for i, buf_addr in zip(range(n_buffers), self.buf_addr):
            if i==n_buffers-1:
                next_addr = self.chain_addr[0] if self.cyclic else 0
            else:
                next_addr = self.chain_addr[i+1]
            descriptor = S2MMBufferDescriptor(next_addr, buf_addr, self.buffer_size*item_nbytes)
            self._chain[i, :] = descriptor.array

    @property
    def head_addr(self):
        """ Return the memory address of the first descriptor in the chain"""
        return self._chain[0].device_address

    @property
    def tail_addr(self):
        """ Return the memory address of the last descriptor in the chain"""
        return  self._chain[-1].device_address

    def cyclify_index(self, i):
        if not self.cyclic and i<0 or i>=self.chain_length:
            raise IndexError(f'Index {i} out of bounds for a non-cyclic chain of length {self.chain_length}')
        i%=self.chain_length
        return i

    def descriptor(self, i):
        """Return a dict of link i of the chain"""
        return self._chain[self.cyclify_index(i)]

    def descriptor_dict(self, i):
        """Return a dict of link i of the chain"""
        return S2MMBufferDescriptor.array_to_dict(self._chain[self.cyclify_index(i)])

    def descriptor_data(self, i):
        """Retrieve the data from the descriptor"""
        i = self.cyclify_index(i)

        cd = self._chain[i]
        cd.flush()
        desc=S2MMBufferDescriptor.from_array(cd)

        # sd = unpack_s2mm_status(self._descr[fetch, 7])
        if desc.DMAIntErr:
            assert desc.size != 0, 'a descriptor has null size'
            raise RuntimeError('DMA under/overrun, dropping data??')
        if desc.DMADecErr:
            raise RuntimeError('Invalid buffer descriptor address')
        if desc.DMASlvErr:
            raise RuntimeError('Error on slave read of the Memory Map interface')
        if not desc.complete:
            raise RuntimeError('IOC but descriptor not completed')
        if not desc.RXSOF or not desc.RXEOF:
            raise NotImplementedError('Packet spans multiple descriptors')
        # if not cd[7]>>31:
        #     raise RuntimeError('Descriptor not complete')
        self._buffers[i].flush()
        d = np.array(self._buffers[i, :desc.n_transferred])
        cd[7]&=0x7FFF_FFFF
        return d

    def __del__(self):
        self._buffers.freebuffer()
        self._chain.freebuffer()


class _CyclicS2MMDMAChannel(_SGDMAChannel):
    def __init__(self, existing_rx):
        super().__init__(existing_rx._mmio, existing_rx._max_size, existing_rx._align,
                         existing_rx._tx_rx, existing_rx._dre, interrupt=existing_rx._interrupt)
        self.chain = None

    def dma_info(self, s2mmchain=True):
        S2MM_DMACR = MM2S_DMACR = (  # (name, offset, length)
            ('irq_delay', 24, 8),
            ('irq_threshold', 16, 8),
            ('error_irq_enable', 14, 1),
            ('delay_irq_enable', 13, 1),
            ('complete_irq_enable', 12, 1),
            ('cyclic_bd_enable', 4, 1), ('keyhole', 3, 1), ('reset', 2, 1), ('runstop', 0, 1))

        S2MM_DMASR = MM2S_DMASR = (  # (name, offset, length)
            ('delay_status', 24, 8),
            ('threshold_status', 16, 8),
            ('error_irq', 14, 1),
            ('delay_irq', 13, 1),
            ('complete_irq', 12, 1),
            ('sg_decode_error', 10, 1), ('sg_slave_error', 9, 1), ('sg_internal_error', 8, 1),
            ('dma_decode_error', 6, 1), ('dma_slave_error', 5, 1), ('dma_internal_error', 4, 1),
            ('sg_included', 3, 1), ('idle', 1, 1), ('halted', 0, 1))

        S2MM_CURDESC = MM2S_CURDESC = (('current_descriptor', 0, 64),)
        S2MM_TAILDESC = MM2S_TAILDESC = (('tail_descriptor', 0, 64),)
        S2MM_DA = MM2S_SA = (('address', 0, 64),)
        S2MM_LENGTH = MM2S_LENGTH = (('length', 0, 26),)
        SG_CTL = (('sg_user', 8, 4),
                  ('sg_cache', 0, 4))

        def pack(description, d):
            result = 0
            for n, o, l in description:
                v = d[n]
                result |= (v & (2 ** l - 1)) << o
            return result

        def unpack(description, x):
            result = {}
            for n, o, l in description:
                result[n] = (x >> o) & (2 ** l - 1)
            return result

        regs = (('MM2S_DMACR', 0, 1, MM2S_DMACR),
                ('MM2S_DMASR', 0x4, 1, MM2S_DMASR),
                ('MM2S_CURDESC', 0x8, 2, MM2S_CURDESC),
                ('MM2S_TAILDESC', 0x10, 2, MM2S_TAILDESC),
                ('MM2S_SA', 0x18, 2, MM2S_SA),
                ('MM2S_LENGTH', 0x28, 1, MM2S_LENGTH),
                ('SG_CTL', 0x2c, 1, SG_CTL),
                ('S2MM_DMACR', 0x30, 1, S2MM_DMACR),
                ('S2MM_DMASR', 0x34, 1, S2MM_DMASR),
                ('S2MM_CURDESC', 0x38, 2, S2MM_CURDESC),
                ('S2MM_TAILDESC', 0x40, 2, S2MM_TAILDESC),
                ('S2MM_DA', 0x4c, 2, S2MM_DA),
                ('S2MM_LENGTH', 0x58, 1, S2MM_LENGTH))

        for k, a, l, r in regs:
            x = self._mmio.read(a, l * 4)
            print(k, x, unpack(r, x), '\n')
        if s2mmchain:
            for i in range(self.chain.chain_length):
                print(self.chain.descriptor(i).physical_address, self.chain.descriptor_dict(i))


    @property
    def s2mm_currdesc(self):
        return self._mmio.read(self._offset+8, length=8, word_order='little')

    def s2mm_dmasr(self, unpack=False):
        x= self._mmio.read(self._offset + 4)
        if unpack:
            x = S2MMBufferDescriptor.S2MM_STATUS_BD.unpack(np.array([x]))
        return x

    @property
    def error_str(self):
        error = self.s2mm_dmasr
        if not error & 0x770:  # error bits
            return None
        if error & 0x10:
            return "DMA Internal Error (transfer length 0?)"
        if error & 0x20:
            return "DMA Slave Error (cannot access memory map interface)"
        if error & 0x40:
            return "DMA Decode Error (invalid address)"
        if error & 0x100:
            return "Scatter-Gather Internal Error (re-used completed descriptor)"
        if error & 0x200:
            return "Scatter-Gather Slave Error (cannot access memory map interface)"
        if error & 0x400:
            return "Scatter-Gather Decode Error (invalid descriptor address)"

    def transfer(self):
        """
        Transfer memory with the DMA

        Transfer must only be called when the channel is halted
        For `nbytes`, 0 means everything after the starting point.

        If the AXI DMA is not configured for data re-alignment then a
        valid address must be aligned or undefined results occur.

        For S2MM (recv), if Data Realignment Engine is not included,
        the destination address must be S2MM Memory Map data width aligned.

        For example, if memory map data width = 32, data is aligned if it is
        located at word offsets (32-bit offset), that is, 0x0, 0x4, 0x8, 0xC,
        and so forth.

        Cyclic Buffer Descriptor (BD) mode allows the DMA to loop through the
        buffer descriptors without user intervention. As the DMA cycles through
        the BDs indefinitely, the wait() function is not valid in this mode.
        Instead, use the stop() function to terminate DMA operation. This mode
        is only valid for the sendchannel.

        Parameters
        ----------
        array : ContiguousArray
            An contiguously allocated array to be transferred
        start : int
             Offset into array to start. Default is 0.
        nbytes : int
             Number of bytes to transfer. Default is 0.
        cyclic : bool
             Enable cyclic BD mode. Default is False.

        """
        if not self.halted:
            raise RuntimeError("DMA channel not halted")

        self._cyclic = False

        # Idle DMA engine
        self.stop()

        # Figure out largest possible block size

        blk_size = self._max_size - (self._max_size % self._align)
        n = blk_size//np.uint64().itemsize
        if blk_size%np.uint64().itemsize:
            raise RuntimeError('buffer size is not an appropriate multiple')

        self.chain = S2MMBufferChain(4, dtype='u8', buffer_size=n, cyclic=True)

        start = 0
        if not self._dre and ((self.chain._buffers.physical_address + start) % self._align) != 0:
            raise RuntimeError(
                "DMA does not support unaligned transfers; "
                "Starting address must be aligned to "
                "{} bytes.".format(self._align)
            )

        self.chain._buffers[:]=0
        if self._flush_before:
            self.chain._buffers.flush()

        # Flush DMA descriptors
        self.chain._chain.flush()
        self._descr = self.chain._chain


        desc0 = self.chain.descriptor(0)
        descN = self.chain.descriptor(self.chain.chain_length-1)

        # Write first desc
        self._mmio.write(self._offset + 0x08, desc0.physical_address & 0xFFFFFFFF)
        self._mmio.write(self._offset + 0x0C, (desc0.physical_address>> 32) & 0xFFFFFFFF)

        self._active_buffer = self.chain._buffers

        # Let's go!
        self.transferred = 0
        self._mmio.write(self._offset, 0x1001)  # Interrupt on complete non-cyclic (cyclic: 0x1011
        while not self.running:
            pass

        # Writing last desc triggers the descriptor fetches
        self._mmio.write(self._offset + 0x10, descN.physical_address & 0xFFFFFFFF)
        self._mmio.write(self._offset + 0x14, (descN.physical_address>> 32) & 0xFFFFFFFF)

    def wait(self, use_asyncio=False):
        if use_asyncio:
            try:
                loop = asyncio.get_running_loop()
            except:
                loop = asyncio.new_event_loop()
            task = loop.create_task(self._interrupt.wait())
            loop.run_until_complete(task)
        else:
            while not self.s2mm_dmasr() & 0x7000:  # an interrupt
                pass

    def yield_buf(self, use_asyncio=False):
        self.transfer()

        self.packets = []
        max_packets = 40
        while True:
            self.wait(use_asyncio=use_asyncio)
            status = self.s2mm_dmasr(unpack=True)
            if status['dma_decode_err'] or status['dma_slave_err'] or status['dma_internal_err']:
                error_str = 'Errors: '
                if status['dma_decode_err']:
                    error_str+='dma_decode_err '
                if status['dma_slave_err']:
                    error_str += 'dma_slave_err '
                if status['dma_internal_err']:
                    error_str += 'dma_internal_err '
                raise RuntimeError(error_str)
            elif status['idle'] or status['halted']:
                break
            elif status['complete']:
                i = self.chain.chain_addr.index(self.s2mm_currdesc)
                i = self.chain.cyclify_index(i-1)
                try:
                    d = self.chain.descriptor_data(i)
                except RuntimeError as e:
                    break
                self.s2mm_taildesc = self.chain.chain_addr[i]
                self.packets.append(d)
                if len(self.packets) > max_packets:
                    break
            else:
                raise RuntimeError('Not possible')

    @property
    def s2mm_taildesc(self):
        return self.mmio.read(self._offset+0x10, 8, 'little')

    @s2mm_taildesc.setter
    def s2mm_taildesc(self, x):
        self.mmio.write(self._offset+0x10, x & 0xffffffff)
        self.mmio.write(self._offset+0x14, (x  >> 32) & 0xffffffff)
