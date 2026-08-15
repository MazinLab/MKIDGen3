"""The v3 FIR commit vector must cross the 512-deep TX FIFO in chunks.

The hardware FIFO (C_TX_FIFO_DEPTH=512, PG080) tops out at 508 words of
vacancy, and fir_config_packet(3) is 512 words: a single-frame tx fails
deterministically on every v3 commit, which is exactly how the ipfA
bring-up found it on 2026-08-11.
"""
import pathlib
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mkidgen3.drivers.axififo import AxisFIFO
from mkidgen3.recordfmt import FIR_CONFIG_TDEST, fir_config_packet


class FakeFifo:
    """Vacancy-accounting stand-in: writes consume room, reads drain it."""

    tx_chunked = AxisFIFO.tx_chunked

    def __init__(self, depth=512, drains=True):
        self._max = depth - 4
        self._vacancy = self._max
        self._drains = drains
        self.frames = []

    @property
    def tx_vacancy(self):
        vacancy = self._vacancy
        if self._drains:
            self._vacancy = self._max
        return vacancy

    def tx(self, data, destination=0, last_bytes=4, wait=False,
           check_vacancy=True):
        data = np.asarray(data)
        assert data.size <= self._vacancy, "tx would overfill the fake FIFO"
        self._vacancy -= data.size
        self.frames.append((data.copy(), destination, last_bytes))


def test_v3_config_vector_crosses_as_two_frames_on_the_config_tdest():
    fifo = FakeFifo()
    packet = fir_config_packet(3)
    assert packet.size == 512  # the defect precondition: > depth - 4

    fifo.tx_chunked(packet, destination=FIR_CONFIG_TDEST, last_bytes=4)

    assert [frame.size for frame, _, _ in fifo.frames] == [256, 256]
    assert {dest for _, dest, _ in fifo.frames} == {FIR_CONFIG_TDEST}
    rejoined = np.concatenate([frame for frame, _, _ in fifo.frames])
    np.testing.assert_array_equal(rejoined, packet)


def test_last_bytes_lands_only_on_the_final_frame():
    fifo = FakeFifo()
    fifo.tx_chunked(np.arange(300, dtype=np.uint32), destination=1,
                    last_bytes=2)
    assert [(frame.size, lb) for frame, _, lb in fifo.frames] == [
        (256, 4), (44, 2)
    ]


def test_v2_vector_still_goes_out_as_a_single_frame():
    fifo = FakeFifo()
    packet = fir_config_packet(2)
    fifo.tx_chunked(packet, destination=FIR_CONFIG_TDEST)
    assert [frame.size for frame, _, _ in fifo.frames] == [256]


def test_a_consumer_that_never_drains_raises_instead_of_hanging():
    fifo = FakeFifo(drains=False)
    fifo._vacancy = 0
    with pytest.raises(TimeoutError, match="consumer clocked"):
        fifo.tx_chunked(np.arange(16, dtype=np.uint32), timeout_s=0.05)


def test_send_config_uses_the_chunked_path():
    # phasematch imports pynq at module top, so pin the production call
    # site textually: a regression back to single-frame tx() would
    # deterministically fail every v3 commit on the 512-deep FIFO.
    source = (pathlib.Path(__file__).parent.parent / "mkidgen3" / "drivers"
              / "phasematch.py").read_text()
    send_config = source.split("def _send_config", 1)[1].split("\n    def ")[0]
    assert "tx_chunked(fir_config_packet" in send_config
    assert "self.fifo.tx(" not in send_config


def test_mixed_width_driver_sends_native_packets_and_tlast(monkeypatch):
    """One channel emits 31 TH2 halfwords and 16 D2 halfwords, independently."""
    monkeypatch.setitem(
        sys.modules, "pynq", SimpleNamespace(DefaultHierarchy=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "mkidgen3.fixedpoint",
        SimpleNamespace(fp_factory=lambda *_args, **_kwargs: None),
    )
    sys.modules.pop("mkidgen3.drivers.phasematch", None)
    module = importlib.import_module("mkidgen3.drivers.phasematch")
    driver = object.__new__(module.PhasematchDriver)
    driver._record_version = 3
    driver._taps_by_tdest = (30, 30, 15, 15)
    driver._pending = {}

    class RecordingFifo:
        def __init__(self):
            self.frames = []

        def tx(self, data, destination=0, last_bytes=4, wait=False):
            self.frames.append(
                (np.asarray(data).copy(), destination, last_bytes, wait)
            )

    driver.fifo = RecordingFifo()
    th2 = np.arange(30, dtype=np.int16)
    d2 = np.arange(100, 115, dtype=np.int16)

    driver.load_coeff(
        0,
        th2,
        d2_coeffs=d2,
        raw=True,
        defer_commit=True,
    )

    assert [(data.size, dest, last) for data, dest, last, _ in driver.fifo.frames] == [
        (16, 0, 2),
        (8, 2, 4),
    ]
