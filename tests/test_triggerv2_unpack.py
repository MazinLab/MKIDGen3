"""Off-hardware tests for the v2 and v3 photon record unpackers.

Runs anywhere numpy is available (no pynq, no board). The layout constants
must match trigger_event in gen3-vivado-top rtl/mkidaranth (benexperiment).
"""
import numpy as np
import pytest

from mkidgen3.drivers.triggerv2 import (PHOTON_V2_DTYPE, PHOTON_V2_PACKED_DTYPE,
                                        PHOTON_V3_DTYPE, PHOTON_PACKED_DTYPE,
                                        unpack_photons_v2, pack_photons_v2,
                                        unpack_photons_v3, pack_photons_v3,
                                        photon_dtype, unpack_photons, pack_photons,
                                        photon_times_ns, TriggerSubsystemV2,
                                        SENTINEL_U64, valid_photon_prefix,
                                        unpack_postage, POSTAGE_SAMPLES,
                                        us_to_holdoff)
from mkidgen3.recordfmt import (DEFAULT_RECORD_VERSION, RECORD_VERSION_OFFSET,
                                SUPPORTED_RECORD_VERSIONS, cycle_ns,
                                record_info)


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
    for version, tick in ((2, 1000), (3, 2000)):
        t = photon_times_ns(p, hdr, version)
        assert t[0] == 5_000_000_000
        assert t[1] == 5_000_000_000 + 10 * tick
        assert cycle_ns(version) == tick


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


def test_us_to_holdoff_is_version_keyed():
    # v3: the holdoff field counts 2 us visits. v2: 1 us visits -- the old
    # module constant claimed 2 us for both and was wrong for v2.
    assert us_to_holdoff(20, 3) == 10
    assert us_to_holdoff(20, 2) == 20
    assert us_to_holdoff(0, 3) == 0
    assert us_to_holdoff(5, 3) == 3        # rounds up: never shorter than asked
    assert us_to_holdoff(5, 2) == 5
    assert us_to_holdoff(10_000, 3) == 255  # clipped to the 8-bit field
    assert us_to_holdoff(10_000, 2) == 255


class _FakeTrigger(TriggerSubsystemV2):
    """Register model; deliberately skips DefaultIP.__init__."""

    def __init__(self, record_version_word=0x0000A203):
        self._init_state()
        self._word = record_version_word
        self.reads = []

    def read(self, offset):
        self.reads.append(offset)
        assert offset == RECORD_VERSION_OFFSET
        return self._word

    def write(self, offset, value):  # pragma: no cover - not used here
        raise AssertionError('unexpected write')


def test_record_version_defaults_to_v2_until_probed():
    t = _FakeTrigger()
    assert t.record_version == DEFAULT_RECORD_VERSION
    assert t.record_version_info is None
    assert t.reads == []          # never probed blind


def test_probe_record_version_reads_0x40_and_caches():
    t = _FakeTrigger()
    assert t.probe_record_version() == dict(version=3, lanes=2, beat_bits=10)
    assert t.reads == [0x40]
    assert t.record_version == 3
    assert t.record_version_info == dict(version=3, lanes=2, beat_bits=10)


def test_probe_record_version_refuses_an_unknown_version():
    t = _FakeTrigger(record_version_word=0x0000A207)
    with pytest.raises(RuntimeError, match='record version'):
        t.probe_record_version()


def test_set_record_version_without_touching_the_bus():
    t = _FakeTrigger()
    t.set_record_version(3)
    assert t.record_version == 3 and t.reads == []
    with pytest.raises(ValueError, match='record version'):
        t.set_record_version(1)


def test_set_record_version_uses_the_canonical_geometry():
    # The complete dict, both versions: a declared version must carry the
    # same lanes/beat_bits as recordfmt's tables, not a private copy that
    # can drift out of step with trigger_lane().
    expected = {2: dict(version=2, lanes=4, beat_bits=9),
                3: dict(version=3, lanes=2, beat_bits=10)}
    for v in SUPPORTED_RECORD_VERSIONS:
        t = _FakeTrigger()
        t.set_record_version(v)
        assert t.record_version_info == expected[v]
        assert t.record_version_info == record_info(v)
        assert t.record_version == v
        assert t.reads == []


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


def _random_photons_v3(n, rng):
    p = np.zeros(n, dtype=PHOTON_V3_DTYPE)
    p['phase'] = rng.integers(-0x8000, 0x8000, n)
    p['baseline'] = rng.integers(-0x8000, 0x8000, n)
    p['cycle'] = rng.integers(0, 1 << 44, n)
    p['id'] = rng.integers(0, 2048, n)
    p['read'] = rng.integers(0, 4, n)
    p['x'] = rng.integers(0, 1 << 12, n)
    p['y'] = rng.integers(0, 1 << 12, n)
    p['dt'] = rng.integers(-128, 128, n)
    p['pileup'] = rng.integers(0, 2, n).astype(bool)
    return p


def test_v3_dtype_fields():
    assert PHOTON_V3_DTYPE.names == ('cycle', 'phase', 'baseline', 'id',
                                     'read', 'x', 'y', 'dt', 'pileup')
    assert PHOTON_V3_DTYPE['dt'] == np.int8
    assert PHOTON_V3_DTYPE['pileup'] == np.bool_
    assert photon_dtype(2) is PHOTON_V2_DTYPE
    assert photon_dtype(3) is PHOTON_V3_DTYPE


def test_v3_roundtrip():
    rng = np.random.default_rng(1234)
    p = _random_photons_v3(10000, rng)
    up = unpack_photons_v3(pack_photons_v3(p))
    for f in PHOTON_V3_DTYPE.names:
        np.testing.assert_array_equal(up[f], p[f], err_msg=f)


def test_v3_known_word():
    # phase=-100, baseline=-3, cycle=0x123_4567_89AB, bin=1234, read=2,
    # x=y=0, dt=-5, pileup=1
    lo = ((-100) & 0xffff) | (((-3) & 0xffff) << 16) \
        | ((0x4567_89AB & 0xffffffff) << 32)
    hi = 0x123 | (1234 << 12) | (2 << 23) | (((-5) & 0xff) << 49) | (1 << 57)
    w = np.array([(lo, hi)], dtype=PHOTON_PACKED_DTYPE)
    p = unpack_photons_v3(w)[0]
    assert p['phase'] == -100
    assert p['baseline'] == -3
    assert p['cycle'] == 0x123_4567_89AB
    assert p['id'] == 1234
    assert p['read'] == 2
    assert p['x'] == 0 and p['y'] == 0
    assert p['dt'] == -5
    assert bool(p['pileup']) is True


def test_v3_dt_is_signed_and_pileup_is_the_next_bit_up():
    p = np.zeros(3, dtype=PHOTON_V3_DTYPE)
    p['dt'] = (-128, 0, 127)
    p['pileup'] = (False, True, False)
    up = unpack_photons_v3(pack_photons_v3(p))
    np.testing.assert_array_equal(up['dt'], (-128, 0, 127))
    np.testing.assert_array_equal(up['pileup'], (False, True, False))


def test_v3_pad_bits_stay_zero_so_the_sentinel_cannot_collide():
    rng = np.random.default_rng(99)
    packed = pack_photons_v3(_random_photons_v3(50000, rng))
    assert ((packed['hi'] >> np.uint64(58)) == 0).all()
    assert (packed['hi'] != SENTINEL_U64).all()
    # and the prefix scan still finds the boundary in a v3 buffer
    buf = np.empty(20, dtype=PHOTON_PACKED_DTYPE)
    buf['lo'] = SENTINEL_U64
    buf['hi'] = SENTINEL_U64
    buf[:6] = pack_photons_v3(_random_photons_v3(6, rng))
    assert valid_photon_prefix(buf) == 6


def test_version_dispatch():
    rng = np.random.default_rng(11)
    p2 = _random_photons(5, rng)
    p3 = _random_photons_v3(5, rng)
    np.testing.assert_array_equal(
        unpack_photons(pack_photons(p2, 2), 2)['id'], p2['id'])
    np.testing.assert_array_equal(
        unpack_photons(pack_photons(p3, 3), 3)['dt'], p3['dt'])
    with pytest.raises(ValueError, match='record version'):
        unpack_photons(pack_photons(p3, 3), 4)


def test_v3_accumulate_into_out():
    rng = np.random.default_rng(3)
    p = _random_photons_v3(10, rng)
    out = np.zeros(25, dtype=PHOTON_V3_DTYPE)
    unpack_photons_v3(pack_photons_v3(p), out=out, n=5)
    np.testing.assert_array_equal(out['dt'][5:15], p['dt'])
    with pytest.raises(IndexError):
        unpack_photons_v3(pack_photons_v3(p), out=out, n=20)


def test_v3_saturated_record_is_still_distinct_from_the_sentinel():
    # The gate the sentinel rests on, stated as one record rather than
    # sampled: every legal v3 field set to all ones at once. hi tops out at
    # bit 57 because [127:122] is pad, so it can never be all ones.
    p = np.zeros(1, dtype=PHOTON_V3_DTYPE)
    p['phase'] = -1
    p['baseline'] = -1
    p['cycle'] = (1 << 44) - 1
    p['id'] = 0x7ff
    p['read'] = 3
    p['x'] = 0xfff
    p['y'] = 0xfff
    p['dt'] = -1
    p['pileup'] = True
    packed = pack_photons_v3(p)
    assert packed['hi'][0] == np.uint64(0x03ff_ffff_ffff_ffff)
    assert packed['hi'][0] != SENTINEL_U64
    # lo DOES saturate to all ones here, which is exactly why the prefix scan
    # keys on hi alone: a legal record can look like drained space in lo.
    assert packed['lo'][0] == SENTINEL_U64
    buf = np.empty(4, dtype=PHOTON_PACKED_DTYPE)
    buf['lo'] = SENTINEL_U64
    buf['hi'] = SENTINEL_U64
    buf[0] = packed[0]
    assert valid_photon_prefix(buf) == 1
    up = unpack_photons_v3(packed)[0]
    assert up['cycle'] == (1 << 44) - 1
    assert up['dt'] == -1
    assert bool(up['pileup']) is True


def test_capture_postage_groups_bins_by_the_version_lane_rule(monkeypatch):
    # Drives the real capture_postage grouping. The per-lane cap raises
    # before any bus access, so patching the pynq flag is enough to reach it.
    monkeypatch.setattr('mkidgen3.drivers.triggerv2._PYNQ', True)
    t = _FakeTrigger()
    t.set_record_version(3)
    # Five even bins are five LANE-0 bins on v3 (lane = bin % 2) and overflow
    # the 4-per-lane cap. Under the old `bin & 3` they spread over lanes 0
    # and 2, slip the cap, and get programmed into the wrong engines.
    with pytest.raises(ValueError, match=r'per lane \(lane = bin % 2\)'):
        t.capture_postage([0, 2, 4, 6, 8])
    t.set_record_version(2)
    with pytest.raises(ValueError, match=r'per lane \(lane = bin % 4\)'):
        t.capture_postage([0, 4, 8, 12, 16])
    assert t.reads == []          # grouping never touches the bus


def test_postage_lane_grouping_is_version_keyed():
    from mkidgen3.recordfmt import trigger_lane
    t = _FakeTrigger()
    assert t.record_version == 2
    assert [trigger_lane(b, t.record_version) for b in (0, 1, 2, 3, 4)] \
        == [0, 1, 2, 3, 0]
    t.set_record_version(3)
    assert [trigger_lane(b, t.record_version) for b in (0, 1, 2, 3, 4)] \
        == [0, 1, 0, 1, 0]
    assert t.photon_dtype is PHOTON_V3_DTYPE
