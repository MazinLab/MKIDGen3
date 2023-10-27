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
