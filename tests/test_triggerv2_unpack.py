"""Off-hardware tests for the v2 photon record unpacker.

Runs anywhere numpy is available (no pynq, no board). The layout constants
must match trigger_event in gen3-vivado-top rtl/mkidaranth (benexperiment).
"""
import numpy as np
import pytest

from mkidgen3.drivers.triggerv2 import (PHOTON_V2_DTYPE, PHOTON_V2_PACKED_DTYPE,
                                        unpack_photons_v2, pack_photons_v2,
                                        photon_times_ns, CYCLE_NS,
                                        SENTINEL_U64, valid_photon_prefix,
                                        unpack_postage, POSTAGE_SAMPLES,
                                        HOLDOFF_CYCLE_US, us_to_holdoff)


def _random_photons(n, rng):
    p = np.zeros(n, dtype=PHOTON_V2_DTYPE)
    p['phase'] = rng.integers(-0x8000, 0x8000, n)
    p['baseline'] = rng.integers(-0x8000, 0x8000, n)
    p['cycle'] = rng.integers(0, 1 << 44, n)
    p['id'] = rng.integers(0, 2048, n)
    p['read'] = rng.integers(0, 4, n)
    p['x'] = rng.integers(0, 1 << 12, n)
    p['y'] = rng.integers(0, 1 << 12, n)
    return p


def test_roundtrip():
    rng = np.random.default_rng(1234)
    p = _random_photons(10000, rng)
    up = unpack_photons_v2(pack_photons_v2(p))
    for f in PHOTON_V2_DTYPE.names:
        np.testing.assert_array_equal(up[f], p[f], err_msg=f)


def test_known_word():
    # Hand-assembled record: phase=-100, baseline=-3, cycle=0x123_4567_89AB,
    # bin=1234, read=2, x=0, y=0
    lo = ((-100) & 0xffff) | (((-3) & 0xffff) << 16) | ((0x4567_89AB & 0xffffffff) << 32)
    hi = 0x123 | (1234 << 12) | (2 << 23)
    w = np.array([(lo, hi)], dtype=PHOTON_V2_PACKED_DTYPE)
    p = unpack_photons_v2(w)[0]
    assert p['phase'] == -100
    assert p['baseline'] == -3
    assert p['cycle'] == 0x123_4567_89AB
    assert p['id'] == 1234
    assert p['read'] == 2
    assert p['x'] == 0 and p['y'] == 0


def test_raw_bytes_and_uint64_inputs():
    rng = np.random.default_rng(7)
    p = _random_photons(64, rng)
    packed = pack_photons_v2(p)
    as_u64 = np.frombuffer(packed.tobytes(), dtype=np.uint64)
    up = unpack_photons_v2(as_u64)
    np.testing.assert_array_equal(up['cycle'], p['cycle'])
    np.testing.assert_array_equal(up['phase'], p['phase'])


def test_accumulate_into_out():
    rng = np.random.default_rng(3)
    p = _random_photons(10, rng)
    out = np.zeros(25, dtype=PHOTON_V2_DTYPE)
    unpack_photons_v2(pack_photons_v2(p), out=out, n=5)
    np.testing.assert_array_equal(out['id'][5:15], p['id'])
    with pytest.raises(IndexError):
        unpack_photons_v2(pack_photons_v2(p), out=out, n=20)


def test_photon_times():
    p = np.zeros(2, dtype=PHOTON_V2_DTYPE)
    p['cycle'] = (1000, 1010)
    hdr = dict(time_ns=5_000_000_000, cycle=1000)
    t = photon_times_ns(p, hdr)
    assert t[0] == 5_000_000_000
    assert t[1] == 5_000_000_000 + 10 * CYCLE_NS


def test_sentinel_cannot_collide_with_real_records():
    # Real v2 records have zero pad bits [127:113], so the all-ones sentinel
    # used to prefill capture buffers can never match a gateware-written event.
    rng = np.random.default_rng(99)
    packed = pack_photons_v2(_random_photons(50000, rng))
    assert (packed['hi'] != SENTINEL_U64).all()


def test_valid_photon_prefix():
    rng = np.random.default_rng(5)
    buf = np.empty(100, dtype=PHOTON_V2_PACKED_DTYPE)
    buf['lo'] = SENTINEL_U64
    buf['hi'] = SENTINEL_U64
    assert valid_photon_prefix(buf) == 0
    buf[:7] = pack_photons_v2(_random_photons(7, rng))
    assert valid_photon_prefix(buf) == 7
    buf[:] = pack_photons_v2(_random_photons(100, rng))
    assert valid_photon_prefix(buf) == 100
    # raw uint64 input path (as np.array(pynq_buffer) would provide)
    as_u64 = np.frombuffer(buf.tobytes(), dtype=np.uint64)
    assert valid_photon_prefix(as_u64) == 100


def test_unpack_postage():
    # Stamp word layout: real[15:0] | imag[31:16], int16 each, little endian.
    n = 3
    real = np.arange(-64, -64 + n * POSTAGE_SAMPLES, dtype=np.int16)
    imag = np.arange(100, 100 + n * POSTAGE_SAMPLES, dtype=np.int16)
    words = (real.astype(np.uint16).astype(np.uint32)
             | (imag.astype(np.uint16).astype(np.uint32) << 16))
    stamps = unpack_postage(words)
    assert stamps.shape == (n, POSTAGE_SAMPLES) and stamps.dtype == np.complex64
    np.testing.assert_array_equal(stamps.real.ravel(), real.astype(np.float32))
    np.testing.assert_array_equal(stamps.imag.ravel(), imag.astype(np.float32))
    # n_stamps clip and raw-bytes input
    assert unpack_postage(words.tobytes(), n_stamps=2).shape == (2, POSTAGE_SAMPLES)
    # trailing partial stamp is dropped
    assert unpack_postage(words[:-1]).shape == (n - 1, POSTAGE_SAMPLES)


def test_us_to_holdoff():
    assert HOLDOFF_CYCLE_US == 2
    assert us_to_holdoff(20) == 10
    assert us_to_holdoff(0) == 0
    assert us_to_holdoff(5) == 3      # rounds up: never shorter than asked
    assert us_to_holdoff(10_000) == 255  # clipped to the 8-bit field


def test_layout_matches_gateware():
    """Cross-check against the amaranth trigger_event layout if available.

    Skips cleanly on machines without the gateware repo/venv; on wheatley it
    pins the driver's bit offsets to the RTL source of truth.
    """
    import importlib.util
    import os
    src = '/work/bmazin/gen3/gen3-vivado-top/rtl/mkidaranth/src'
    if not os.path.isdir(src):
        pytest.skip('gateware repo not present')
    try:
        import sys
        sys.path.insert(0, src)
        from mkidaranth.trigger import trigger_event  # noqa
        from amaranth.lib import data  # noqa
    except ImportError:
        pytest.skip('amaranth not importable in this environment')
    offsets = {}
    pos = 0
    for name, shape in trigger_event.members.items():
        width = data.Layout.cast(shape).size if isinstance(shape, data.Layout) \
            else (shape.width if hasattr(shape, 'width') else int(shape))
        offsets[name] = (pos, width)
        pos += width
    assert offsets['phase'] == (0, 16)
    assert offsets['baseline'] == (16, 16)
    assert offsets['cycle'] == (32, 44)
    assert offsets['bin'] == (76, 11)
    assert offsets['read'] == (87, 2)
    assert offsets['x'] == (89, 12)
    assert offsets['y'] == (101, 12)
