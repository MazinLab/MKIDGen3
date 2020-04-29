# 0.75/np.abs(comb.real).max() = 382.....
from pynq import allocate
import numpy as np
from fpbinary import FpBinary
from mkidgen3.fixedpoint import FP16_15, FP16_26

SCALE_IN = 382.65305668618328 * 2 ** 15
SCALE_OUT = 1 / (.75 * 2 ** 9)


n_packets_sent, pptx, input_buffer = None, None, None


def prep_buffers(ntx=16, n_res=2048, n_bin=4096, latency_shift=3*16):
    """
    ntx: How many packets do we send per DMA transfer.
         Must be <=16 to move the stream smoothly through the core
    """
    global pptx, input_buffer, output_buffer
    pptx = ntx

    try:
        # Close buffers if they are open
        input_buffer.close()
        output_buffer.close()
    except (NameError,AttributeError):
        pass

    # Create the buffers
    n_in_buff = n_res * 2 * pptx if n_packets_sent else n_res * 2 * pptx + latency_shift * 2
    input_buffer = allocate(shape=(n_in_buff,), dtype=np.uint16)  # 2048 I & Q
    output_buffer = allocate(shape=(n_bin * 2,), dtype=np.uint16)


def init_pipe(dma, n_res=2048):
    """Send one packet's worth of 0s down the line."""
    foo = allocate(shape=n_res*2, dtype=np.uint16)
    foo[:] = 0
    dma.sendchannel.transfer(foo)
    dma.sendchannel.wait()
    foo.close()


def packet_to_buffer(packet, zero_i=tuple(), zero_q=tuple(), fp=True, scale=SCALE_IN):
    """
    Converts a packet of complex data into a n_res*2 uint16 array

    Packet should be an array of dtype=np.complex64 or 128. length must be a multiple of 16

    zero_i & zero_q may be set to a tuple to selectively zero out values going into specific lanes

    Setting fp = False will result in the real and imaginaty values being multiplied by scale
    and then cast to uint16. Otherwise FpBinary will be used to convert the data from float to signed 16_15.
    """
    if fp:
        ibits = [FP16_15(x).__index__() for x in packet.real]
        qbits = [FP16_15(x).__index__() for x in packet.imag]
    else:
        ibits = (packet.real * scale).astype(np.uint16)
        qbits = (packet.imag * scale).astype(np.uint16)
    data = np.zeros(2 * packet.size, dtype=np.uint16)
    for i in range(8):
        data[i::16] = 0 if i in zero_i else ibits[i::8]
        data[i + 8::16] = 0 if i in zero_q else qbits[i::8]
    return data


def packet_from_buffer(buffer, fp=True, scale=SCALE_OUT):
    """
    Converts a buffer of uint16 fixed point data to a np.complex128.

    Buffer must be in multiples of 2*n_bin.
    """
    ibits = np.zeros(buffer.size // 2, dtype=np.uint16)
    qbits = np.zeros(buffer.size // 2, dtype=np.uint16)
    for i in range(16):
        ibits[i::16] = buffer[2 * i::32]
        qbits[i::16] = buffer[2 * i + 1::32]

    packet = np.zeros(buffer.size // 2, dtype=np.complex128)
    if fp:
        packet.real = [float(FpBinary(int_bits=-9, frac_bits=25, signed=True, bit_field=int(x))) for x in ibits]
        packet.imag = [float(FpBinary(int_bits=-9, frac_bits=25, signed=True, bit_field=int(x))) for x in qbits]
    else:
        packet.real = ibits * scale
        packet.imag = qbits * scale

    return packet


def txpackets(dma, packets, **kwargs):
    """
    Send packets to the FPGA and increment the number of packets sent. The number of packets
    must correspond to the number specified in prep_buffers.
    """
    global n_packets_sent, pptx, input_buffer
    input_buffer[:] = np.array([packet_to_buffer(p, **kwargs) for p in packets]).ravel()
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    n_packets_sent += pptx


def txcomb(dma, comb, n_res=2048, latency_shift=3*16, **kwargs):
    global n_packets_sent, pptx, input_buffer, next_sample_send
    # 92/2 empirically determined to overcome reorder core latency

    if n_packets_sent:
        n_to_send = pptx * n_res
        data = comb.ravel()[next_sample_send:next_sample_send + n_to_send]
        input_buffer[:] = packet_to_buffer(data, **kwargs)
    else:
        n_to_send = (pptx - 1) * n_res + latency_shift
        data = comb.ravel()[next_sample_send:next_sample_send + n_to_send]
        input_buffer[:n_res * 2] = 0
        input_buffer[n_res * 2:] = packet_to_buffer(data, **kwargs)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    next_sample_send += n_to_send
    n_packets_sent = 1 + next_sample_send // n_res


def rxpackets(dma, packets_out, n=None, status=False, **kwargs):
    """Attempts to receive packets. If no number is specified then n_outstanding-1 are received."""
    global n_packets_sent, n_packets_rcvd
    if n is None:
        n = n_packets_sent - n_packets_rcvd
    for i in range(n - 1):
        if status:
            print(f"Receiving packet {n_packets_rcvd}")
        dma.recvchannel.transfer(output_buffer)
        dma.recvchannel.wait()
        converted = packet_from_buffer(output_buffer, **kwargs)
        packets_out[n_packets_rcvd] = converted
        n_packets_rcvd += 1
    return converted


def txrx(dma, comb, packets_out, n_total_packets=None):
    if n_total_packets is None:
        n_total_packets=comb.size//256//8
    n_loop=(n_total_packets - n_packets_sent) // pptx
    for i in range(n_loop):
        txcomb(dma, comb)
        rxpackets(dma, packets_out)
        print(f"Sent: {n_packets_sent} Received: {n_packets_rcvd}. Pending: {n_packets_sent - n_packets_rcvd}")


def fir(comb, coeffs, packets_out, matlab_sim_out=None,
        lane=0, fftclk=None, show=True, rollfpga_new=False, rollfpga_old=False, fpga_i=0, fpga_p=0,
        first_good_packet=17):
    """
    Compute the PFB FIR in floating and fixed point for the specified lane and fft input clock.

    Also extract and return the corresponding data from the matlab input (if passed) and the FPGA

    Outputs: mlab, pyfloat, pyfixed, fpga

    Assumes comb.shape(N, 2048) matlab_sim_out.shape=(N,256,16)"""
    mlab_block = fftclk // 256
    mlab_chan = fftclk % 256  # fft order

    reorder = mlab_block % 2
    packet_0 = mlab_block  # + (1 if reorder else 0) #fir order
    coeff_chan = mlab_chan if not reorder else (mlab_chan + 128) % 256  # input order
    if coeff_chan < 128:
        comb_chan = coeff_chan
    else:
        comb_chan = coeff_chan - 128
        packet_0 += 1

    pak_out = mlab_block + first_good_packet + fpga_p  # + #(-1 if swapfpga and reorder else 1)

    fpga_chan = mlab_chan + fpga_i
    if (rollfpga_old and reorder) or (rollfpga_new and not reorder):
        fpga_chan = (fpga_chan + 128)
    fpga_chan %= 256

    if show:
        print(f'Matlab block {mlab_block}, channel {mlab_chan}, lane {lane}')
        print(f'FIR Coeff channel {coeff_chan}')
        print(f'Delayed cycle needing reorder: {reorder}')
        print(f'Input samples {packet_0}-{packet_0 + 14} (by twos), channel {comb_chan} (i={16 * comb_chan + lane})')
        print(f'FPGA Output packet {pak_out}, sample {fpga_chan} (i={fpga_chan * 16 + lane})')

    pysamp = comb[packet_0:packet_0 + 15:2, 16 * comb_chan + lane]

    taps = coeffs[lane, coeff_chan, ::-1]
    fpdat = [(FP16_15(s.real), FP16_15(s.imag), FP16_26(c)) for s, c in zip(pysamp, taps)]
    fpreal, fpimag = sum(r * c for r, i, c in fpdat), sum(i * c for r, i, c in fpdat)
    py = (pysamp * taps).sum()
    pyfp = float(fpreal.resize((-9, 25))) + float(fpimag.resize((-9, 25))) * 1j

    fpga = packets_out.reshape(packets_out.shape[0], 256, 16)[pak_out, fpga_chan, lane]

    mlab = matlab_sim_out[mlab_block, mlab_chan, lane] if matlab_sim_out is not None else np.nan
    if show:
        print(f" Matlab:  {mlab:.3g}")  # 256 clocks per packet we are looking for 384 so 2nd packet chan. 128
        print(f" Python:  {py:.3g}  (matlab diff {mlab - py:.0g})")
        print(f" FP16_25: {pyfp:.3g}  (min rep. diff {100 * (py - pyfp) / 2 ** -25:.0f}%)")
        print(f" FPGA[{pak_out},{fpga_chan * 16 + lane}]: {fpga:.3g}")
        print(f" Diff: {pyfp - fpga:.4g}")
    return mlab, py, pyfp, fpga


def do_fft(datain, do_pfb_fft=False, sl=None, roll_pfb_fft=0, n_bin=4096):
    """ Do the fft necessary to look at the tone(s) in each PFB bin.

    Optionally do the PFB FFT (e.g. if the FFT core isn't present.

    sl may be used to run on a slice of the data.
    roll_pfb_fft is intended to roll the FIR output along channels e.g. to investigte reorder issues
    """
    data = datain.reshape(datain.shape[0], n_bin) if datain.ndim == 3 else datain
    if sl is not None:
        data = data[sl]
    data = np.fft.fft(np.roll(data, roll_pfb_fft, axis=1), axis=1) if do_pfb_fft else data
    # if do_welch:
    #     freq, data = welch(data, fs=2e6, return_onesided=False, scaling='spectrum', axis=0)
    #     data = np.fft.fftshift(data, axes=0)
    #     freq = np.fft.fftshift(freq, axes=0)
    # else:
    data = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    return data


def extract_opfb_spec(data, OS=2, exclude_overlap=True, linear=False):
    if not exclude_overlap:
        raise NotImplementedError

    N, M = data.shape
    data = np.abs(data) if linear else 20 * np.log10(np.abs(data))
    ndx_bin = int(np.ceil((OS - 1) / (2 * OS) * N)) - 1  # no overlap start of bin

    stride = int(np.floor(1 / OS * N)) + 1
    yax_t = np.zeros(stride * M)
    xax_t = np.arange(M)[:, np.newaxis] + np.linspace(-.5, .5 - 1 / stride, stride)

    ndx_out = 0
    for i in range(M):
        yax_t[ndx_out:ndx_out + stride] = data[ndx_bin:ndx_bin + stride, i]
        ndx_out += stride

    # shift the negative frequency components back to the correct place
    return xax_t.ravel() - M / 2, np.fft.fftshift(yax_t)
