import numpy as np
from fpbinary import FpBinary, OverflowEnum, RoundingEnum

FP16_15 = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x)
FP16_15 = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x)
FP26_26 = lambda x: FpBinary(int_bits=0, frac_bits=26, signed=True, value=x)
FP16_26 = lambda x: FpBinary(int_bits=-10, frac_bits=26, signed=True, value=x)
FP32_31 = lambda x: FpBinary(int_bits=1, frac_bits=31, signed=True, value=x)
FP32_8 = lambda x: FpBinary(int_bits=32 - 9, frac_bits=8, signed=True, value=x)
FP18_17 = lambda x: FpBinary(int_bits=1, frac_bits=17, signed=True, value=x)
FP16_25 = lambda x: FpBinary(int_bits=-9, frac_bits=25, signed=True, value=x)
FP18_25 = lambda x: FpBinary(int_bits=-7, frac_bits=25, signed=True, value=x)
FP16_17 = lambda x: FpBinary(int_bits=-1, frac_bits=17, signed=True, value=x)


def fp_factory(int, frac, signed, frombits=False, include_index=False):
    if isinstance(signed, str):
        signed = True if 'signed' == signed.lower() else False
    else:
        signed = bool(signed)
    if frombits:
        return lambda x: FpBinary(int_bits=int, frac_bits=frac, signed=signed, bit_field=x)
    else:
        if include_index:
            return lambda x: FpBinary(int_bits=int, frac_bits=frac, signed=signed, value=x).__index__()
        else:
            return lambda x: FpBinary(int_bits=int, frac_bits=frac, signed=signed, value=x)


#TODO in progress
# def quantize(a, format, complex='auto', free_pyng=True):
#     """
#     if complex==true a must be 2d and the first or last axis must be shape=2
#     if both the first and last axes are length 2 the last axis will be treated as the complex pair
#
#     if free_pynq is true and a is a pynq buffer it will be freed after processing
#     """
#     if isinstance(a, (list,tuple)):
#         a=np.asarray(a)
#
#     if a.ndim>1 and a.shape[0] ==2 or a.shape[-1] ==2 and complex=='auto':
#         complex = True
#
#     if complex and a.ndim<2 or 2 not in (a.shape[0],a.shape[-1]) and not is:
#         raise ValueError('Can not convert to complex')
#
#     c_axis = -1 if a.shape[-1]==2 else 0
#     if c_axis==0:
#         out_shape = a.shape[1:]
#     elif c_axis == -1:
#         out_shape = a.shape[:-1]
#     else:
#         out_shape = a.shape
#     if complex:
#         out_shape = out_shape +(2,)
#
#     center_fmt = fp_factory(*format, frombits=False, include_index=True)
#     data = np.zeros(out_shape, dtype=np.uint16)
#     if complex and c_axis is None:
#         data[:, 0] = [center_fmt(x.real) for x in a]
#         data[:, 1] = [center_fmt(x.imag) for x in a]
#     elif complex:
#         if c_axis==-1:
#             data[..., 0].flat[:] = [center_fmt(x) for x in a[...,0].flat]
#             data[..., 1].flat[:] = [center_fmt(x)  for x in a[...,1].flat]
#         else:
#             data[..., 0].flat[:] = [center_fmt(x) for x in a[0,...].flat]
#             data[..., 1].flat[:] = [center_fmt(x)  for x in a[0,...].flat]
#
#     if free_pyng:
#         try:
#             a.freebuffer()
#         except AttributeError:
#             pass  # not a pynq buffer
#
#     return data

def fparray(a, fpgen=None):
    """
    Convert a numpy array to a numpy object array of FpBinary numbers.

    Complex numbers gain an additional axis of length 2.

    if fpgen is None 16wide  1integer signed is used
    """
    try:
        fpgen = lambda x: FpBinary(int_bits=fpgen[0], frac_bits=fpgen[1], signed=fpgen[2], value=x)
    except Exception:
        pass
    if fpgen is None:
        fpgen = lambda x: FpBinary(int_bits=1, frac_bits=15, signed=True, value=x)

    if a.dtype in (np.complex, np.complex128, np.complex64):
        out = np.empty(tuple(list(a.shape)+[2]), dtype=object)
        b = [(fpgen(x.real), fpgen(x.imag)) for x in a.flat]
        for i in range(out.size):
            out.flat[i] = b[i//2][i % 2]
        #You'd think this would work, but it doesn't as the FP types get cast to bool
        #out.flat = [(fpgen(x.real), fpgen(x.imag)) for x in a.flat]
    else:
        out = np.empty_like(a, dtype=object)
        b = [fpgen(x) for x in a.flat]
        for i in range(out.size):
            out.flat[i] = b[i]
        #out.flat = [fpgen(x) for x in a.flat]
    return out


def do_fixed_point_pfb(fpcomb, fpcoeff, n_convert=None, truncate=True):
    """Set truncate to false to preserve the full output bitwidth. Truncation is done with FpBinary defaults."""
    n_total_packets = fpcomb.size // 2048 // 2 - 16 if n_convert is None else n_convert
    fft_block = np.zeros((n_total_packets+1, 256, 16), dtype=np.complex64)
    for i in range(0, n_total_packets, 2):  # each packet of ADC samples, 128 new things to a lane 2 packets to feed all channels
        lane_out = np.zeros((2, 256, 16), dtype=np.complex64)
        for l in range(16):
            fresh = np.array([fpcoeff[l, :, 7 - c_i] * fpcomb[i + 2 * c_i:i + 2 * c_i + 2, l::16, :].reshape(256, 2).T
                              for c_i in range(8)]).sum(axis=0)
            delay = np.roll(np.array(
                [fpcoeff[l, :, 7 - c_i] * fpcomb[1 + i + 2 * c_i:1 + i + 2 * c_i + 2, l::16, :].reshape(256, 2).T
                 for c_i in range(8)]).sum(axis=0), 128, axis=1)
            # Sum the multiplies are roll the delayed samples
            if truncate:
                outformat = (-9, sum(fpcomb.flat[0].format) + 9)
                conv = lambda a: np.array(list(map(lambda x: float(x.resize(outformat)), a)))
                lane_out[0, :, l] = conv(fresh[0]) + conv(fresh[1]) * 1j
                lane_out[1, :, l] = conv(delay[0]) + conv(delay[1]) * 1j
            else:
                lane_out[0, :, l] = fresh[0].astype(float) + fresh[1].astype(float) * 1j
                lane_out[1, :, l] = delay[0].astype(float) + delay[1].astype(float) * 1j
        fft_block[i] = lane_out[0]
        fft_block[i + 1] = lane_out[1]
    return fft_block
