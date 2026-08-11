"""Record-version-keyed constants and geometry (numpy only, no pynq).

Values come from the stage-2 driver handoff, 2026-08-10.
"""
import pytest

from mkidgen3.recordfmt import (RECORD_VERSION_OFFSET, DEFAULT_RECORD_VERSION,
                                SUPPORTED_RECORD_VERSIONS,
                                STAGE2_RECORD_VERSION, decode_record_version,
                                check_version, cycle_ns, holdoff_cycle_us,
                                trigger_lane, CYCLE_NS_BY_VERSION,
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
