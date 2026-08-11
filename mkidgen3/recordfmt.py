"""Record-version-keyed constants, lane arithmetic and filter geometry.

Two photon-record contracts are live in this driver:

* **v2** -- the production 4-lane build. Visits every channel every 1 us,
  ``lane = r % 4``, 512 beats per frame, holdoff CSR counts 1 us visits.
* **v3** -- the stage-1/stage-2 build. Visits every 2 us (500 kHz),
  ``lane = r % 2`` downstream of the phase halfband, 1024 beats per frame,
  holdoff CSR counts 2 us visits, records carry ``dt`` and ``pileup``.

One wheel serves both bitstreams, so every fork is keyed on the record
version and written down exactly once -- here. This module is numpy-only and
imports without pynq, so all of it is unit-testable off-board.

NOTE the transform peripheral (mkidgen3.drivers.iqtransform) keeps its OWN
4-lane convention on the v3 build: it sits upstream of the halfband on the
4-lane 1 MHz stream, so its constant table is indexed with ``lane = r % 4``
and ``row = r // 4``, which the fabric derives from the flat index itself.
Nothing in this module applies to it.

Source: the stage-2 driver handoff, 2026-08-10, ss3.2-3.3 and s4.
"""

# --- record version CSR (trigger subsystem, byte offset 0x40) --------------
RECORD_VERSION_OFFSET = 0x40
DEFAULT_RECORD_VERSION = 2
SUPPORTED_RECORD_VERSIONS = (2, 3)
STAGE2_RECORD_VERSION = dict(version=3, lanes=2, beat_bits=10)


def decode_record_version(word):
    """{version[7:0], lanes[11:8], beat_bits[15:12]} from the CSR word."""
    w = int(word)
    return dict(version=w & 0xff, lanes=(w >> 8) & 0xf,
                beat_bits=(w >> 12) & 0xf)


def check_version(version):
    v = int(version)
    if v not in SUPPORTED_RECORD_VERSIONS:
        raise ValueError(f'unsupported record version {v}; this driver '
                         f'implements {SUPPORTED_RECORD_VERSIONS}')
    return v


# --- time and lane constants ----------------------------------------------
CYCLE_NS_BY_VERSION = {2: 1000, 3: 2000}
HOLDOFF_CYCLE_US_BY_VERSION = {2: 1, 3: 2}
TRIGGER_LANES_BY_VERSION = {2: 4, 3: 2}
FRAME_BEATS_BY_VERSION = {2: 512, 3: 1024}


def cycle_ns(version):
    """ns per unit of the record's ``cycle`` field (= one visit)."""
    return CYCLE_NS_BY_VERSION[check_version(version)]


def holdoff_cycle_us(version):
    """Microseconds per count of the 8-bit trigger holdoff field."""
    return HOLDOFF_CYCLE_US_BY_VERSION[check_version(version)]


def trigger_lane(bin, version):
    """Lane a channel's records and postage engine live on."""
    return int(bin) % TRIGGER_LANES_BY_VERSION[check_version(version)]
