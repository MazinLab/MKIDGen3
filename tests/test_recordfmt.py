"""Record-version-keyed constants and geometry (numpy only, no pynq).

Values come from the stage-2 driver handoff, 2026-08-10.
"""
import pytest

from mkidgen3.recordfmt import (RECORD_VERSION_OFFSET, DEFAULT_RECORD_VERSION,
                                SUPPORTED_RECORD_VERSIONS,
                                STAGE2_RECORD_VERSION, decode_record_version,
                                check_version, cycle_ns, holdoff_cycle_us,
                                trigger_lane, record_info,
                                CYCLE_NS_BY_VERSION,
                                HOLDOFF_CYCLE_US_BY_VERSION,
                                TRIGGER_LANES_BY_VERSION,
                                FRAME_BEATS_BY_VERSION)


def test_offset_and_default():
    assert RECORD_VERSION_OFFSET == 0x40
    assert DEFAULT_RECORD_VERSION == 2
    assert SUPPORTED_RECORD_VERSIONS == (2, 3)


def test_decode_record_version_stage2_word():
    # version=3, lanes=2, beat_bits=10, packed LSB-first
    assert decode_record_version(0x0000A203) == STAGE2_RECORD_VERSION
    assert STAGE2_RECORD_VERSION == dict(version=3, lanes=2, beat_bits=10)
    assert decode_record_version(0x0000_9402) == dict(version=2, lanes=4,
                                                      beat_bits=9)


def test_check_version():
    assert check_version(3) == 3
    assert check_version(2) == 2
    with pytest.raises(ValueError, match='record version'):
        check_version(1)
    with pytest.raises(ValueError, match='record version'):
        check_version(4)


def test_time_constants_are_version_keyed():
    # v2 visits at 1 MHz, v3 at 500 kHz. The old module-level
    # HOLDOFF_CYCLE_US = 2 was right only for v3.
    assert CYCLE_NS_BY_VERSION == {2: 1000, 3: 2000}
    assert HOLDOFF_CYCLE_US_BY_VERSION == {2: 1, 3: 2}
    assert cycle_ns(2) == 1000 and cycle_ns(3) == 2000
    assert holdoff_cycle_us(2) == 1 and holdoff_cycle_us(3) == 2


def test_lane_and_frame_geometry():
    assert TRIGGER_LANES_BY_VERSION == {2: 4, 3: 2}
    assert FRAME_BEATS_BY_VERSION == {2: 512, 3: 1024}
    assert [trigger_lane(b, 3) for b in range(6)] == [0, 1, 0, 1, 0, 1]
    assert [trigger_lane(b, 2) for b in range(6)] == [0, 1, 2, 3, 0, 1]
    assert trigger_lane(2047, 3) == 1
    assert trigger_lane(2047, 2) == 3


def test_record_info_is_derived_from_the_canonical_tables():
    # The whole point: nobody restates {lanes, beat_bits}, so a declared
    # version can never disagree with trigger_lane() or the frame geometry.
    assert record_info(3) == dict(version=3, lanes=2, beat_bits=10)
    assert record_info(2) == dict(version=2, lanes=4, beat_bits=9)
    assert record_info(3) == STAGE2_RECORD_VERSION  # matches the CSR word
    for v in SUPPORTED_RECORD_VERSIONS:
        info = record_info(v)
        assert info['version'] == v
        assert info['lanes'] == TRIGGER_LANES_BY_VERSION[v]
        assert info['lanes'] == trigger_lane(TRIGGER_LANES_BY_VERSION[v] - 1,
                                             v) + 1
        assert 1 << info['beat_bits'] == FRAME_BEATS_BY_VERSION[v]
    with pytest.raises(ValueError, match='record version'):
        record_info(1)


import numpy as np

from mkidgen3.recordfmt import (FIR_TAPS, N_RES, FIR_LANES_BY_VERSION,
                                FIR_SETS_BY_VERSION, FIR_CONFIG_TDEST,
                                FIR_QUADRATURES_BY_VERSION,
                                UNITY_TAP_BY_VERSION, fir_lane, fir_set,
                                fir_tdest, pack16_to_32, fir_config_packet,
                                fir_reload_packet, unity_coefficient_sets)


def test_pack16_to_32_keeps_the_high_half():
    # numpy >= 2 weak promotion turns uint16 << 16 into 0; the packer must
    # widen first or every word loses its high half off-board.
    out = pack16_to_32(np.array([1, 2, 3, 4], dtype=np.uint16))
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, [1 | (2 << 16), 3 | (4 << 16)])
    # odd length: the last value rides alone in the low half
    out = pack16_to_32(np.array([1, 2, 3], dtype=np.uint16))
    np.testing.assert_array_equal(out, [1 | (2 << 16), 3])
    assert pack16_to_32(np.zeros(0, dtype=np.uint16)).size == 0


def test_fir_geometry_v3():
    assert FIR_LANES_BY_VERSION[3] == 2 and FIR_SETS_BY_VERSION[3] == 1024
    assert FIR_QUADRATURES_BY_VERSION[3] == ('th2', 'd2')
    assert [fir_lane(r, 3) for r in range(5)] == [0, 1, 0, 1, 0]
    assert [fir_set(r, 3) for r in range(5)] == [0, 0, 1, 1, 2]
    assert fir_set(2047, 3) == 1023
    assert fir_tdest(0, 3, 'th2') == 0 and fir_tdest(1, 3, 'th2') == 1
    assert fir_tdest(0, 3, 'd2') == 2 and fir_tdest(1, 3, 'd2') == 3
    assert fir_tdest(2046, 3, 'd2') == 2 and fir_tdest(2047, 3, 'd2') == 3
    assert FIR_CONFIG_TDEST == 4


def test_fir_geometry_v2_is_the_historical_mapping():
    assert FIR_LANES_BY_VERSION[2] == 4 and FIR_SETS_BY_VERSION[2] == 512
    assert FIR_QUADRATURES_BY_VERSION[2] == ('th2',)
    assert [fir_lane(r, 2) for r in range(5)] == [0, 1, 2, 3, 0]
    assert [fir_set(r, 2) for r in range(5)] == [0, 0, 0, 0, 1]
    assert [fir_tdest(r, 2, 'th2') for r in range(4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='no D2'):
        fir_tdest(0, 2, 'd2')


def test_config_packets():
    p3 = fir_config_packet(3)
    assert p3.dtype == np.uint32 and p3.size == 512
    np.testing.assert_array_equal(
        p3, pack16_to_32(np.arange(1024, dtype=np.uint16)))
    p2 = fir_config_packet(2)
    assert p2.size == 256
    np.testing.assert_array_equal(
        p2, pack16_to_32(np.arange(512, dtype=np.uint16)))


def test_reload_packet_is_the_set_number_then_reversed_taps():
    taps = np.arange(100, 100 + FIR_TAPS, dtype=np.int16)
    pkt = fir_reload_packet(5, taps, 3)
    assert pkt.dtype == np.uint32 and pkt.size == 16   # 31 uint16 words
    words = np.empty(32, dtype=np.uint16)
    words[:31] = np.frombuffer(pkt.tobytes(), dtype=np.uint16)[:31]
    assert words[0] == 2                     # set = 5 // 2
    np.testing.assert_array_equal(words[1:31], taps[::-1].astype(np.uint16))
    # v2 puts the same channel in set 1 (5 // 4)
    v2 = np.frombuffer(fir_reload_packet(5, taps, 2).tobytes(), dtype=np.uint16)
    assert v2[0] == 1


def test_reload_packet_v2_matches_the_pre_stage2_construction():
    """Byte-identity guard for the production bitstream."""
    rng = np.random.default_rng(4)
    taps = rng.integers(-32768, 32768, FIR_TAPS).astype(np.int16)
    for res_id in (0, 1, 3, 511, 512, 2047):
        legacy = np.zeros(FIR_TAPS + 1, dtype=np.uint16)
        legacy[0] = res_id // 4
        legacy[1:] = taps[::-1]
        expect = np.array(
            [int(legacy[i]) | (int(legacy[i + 1]) << 16)
             for i in range(0, 30, 2)] + [int(legacy[30])], dtype=np.uint32)
        np.testing.assert_array_equal(
            fir_reload_packet(res_id, taps, 2), expect)


def test_reload_packet_rejects_the_wrong_tap_count():
    with pytest.raises(ValueError, match='30'):
        fir_reload_packet(0, np.zeros(29, dtype=np.int16), 3)


def test_unity_coefficient_sets():
    assert UNITY_TAP_BY_VERSION == {2: 32767, 3: -32768}
    c3 = unity_coefficient_sets(7, 3)
    assert c3.shape == (N_RES, FIR_TAPS) and c3.dtype == np.int16
    assert (c3[:7, 0] == -32768).all() and (c3[7:, 0] == 0).all()
    assert (c3[:, 1:] == 0).all()
    c2 = unity_coefficient_sets(7, 2)
    assert (c2[:7, 0] == 32767).all()
