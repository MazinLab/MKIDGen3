"""Off-hardware tests for the v2 photon record unpacker.

Runs anywhere numpy is available (no pynq, no board). The layout constants
must match trigger_event in gen3-vivado-top rtl/mkidaranth (benexperiment).
"""
import numpy as np
import pytest

from mkidgen3.drivers.triggerv2 import (PHOTON_V2_DTYPE, PHOTON_V2_PACKED_DTYPE,
                                        unpack_photons_v2, pack_photons_v2,
                                        photon_times_ns, CYCLE_NS)


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
