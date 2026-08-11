"""HP0 reachability for CMA capture buffers on NODDR overlays.

On a NODDR build axis2mm writes into PS DDR through HP0, and an address
outside the two HP0 windows DECERRs with no symptom the daemon can see. The
check is pure arithmetic and lives in mkidgen3.mkidpynq; the allocation path
that uses it needs pynq and is board-verified by tools/sweepacc_bringup.py.
"""
import pytest

from mkidgen3.mkidpynq import (HP0_WINDOWS, CAPTURE_BEAT_BYTES, hp0_reachable,
                               flush_transfer, arm_fault, axis2mm_quiesced,
                               AXIS2MM_QUIESCENT_BITS)


def test_windows():
    assert HP0_WINDOWS == ((0x0, 0x8000_0000),
                           (0x8_0000_0000, 0x9_0000_0000))


def test_low_window():
    assert hp0_reachable(0x0, 1)
    assert hp0_reachable(0x7FFF_0000, 0x10000)
    assert not hp0_reachable(0x7FFF_0000, 0x10001)   # runs off the end
    assert not hp0_reachable(0x8000_0000, 1)


def test_high_window():
    assert hp0_reachable(0x8_0000_0000, 1)
    assert hp0_reachable(0x8_FFFF_FFFF, 1)
    assert not hp0_reachable(0x9_0000_0000, 1)


def test_the_old_pl_ddr4_window_is_unreachable():
    # 0x5_0000_0000 is where PL DDR4 used to live; writes there DECERR now.
    assert not hp0_reachable(0x5_0000_0000, 4096)


def test_a_buffer_may_not_straddle_the_gap():
    assert not hp0_reachable(0x7FFF_FF00, 0x1_0000)
    assert not hp0_reachable(-1, 1)


# --- what axis2mm is actually programmed with -------------------------------
#
# The length register is in bytes and the buffer is a typed array. flush() got
# that wrong for as long as it has existed: it allocated n u64 words and armed
# the DMA for 64*n bytes, so seven eighths of the transfer ran off the end of
# the buffer -- and once the buffer is CMA in PS DDR, off the end can be off
# the window, which is the DECERR this module exists to prevent.


def test_flush_transfer_covers_the_whole_programmed_length():
    words, nbytes = flush_transfer(1)
    assert nbytes == CAPTURE_BEAT_BYTES
    assert words * 8 == nbytes          # u64 words, eight per beat
    words, nbytes = flush_transfer(37)
    assert nbytes == 37 * 64
    assert words == 37 * 8


def test_flush_transfer_rejects_a_nonpositive_beat_count():
    with pytest.raises(ValueError):
        flush_transfer(0)
    with pytest.raises(ValueError):
        flush_transfer(-1)


def test_arm_fault_passes_a_sound_transfer():
    assert arm_fault(0x1000, 4096, 4096) is None
    assert arm_fault(0x1000, 4096, 1 << 20) is None
    # PL DDR4 is outside both HP0 windows and does not go through HP0.
    assert arm_fault(0x5_0000_0000, 4096, 4096, hp0=False) is None


def test_arm_fault_catches_a_transfer_larger_than_its_buffer():
    why = arm_fault(0x1000, 64 * 37, 8 * 37)   # the historical flush bug
    assert why is not None and 'overruns' in why
    # and it catches it even where the address itself is reachable
    assert arm_fault(0x1000, 128, 64, hp0=True) is not None


def test_arm_fault_catches_an_unreachable_interval():
    why = arm_fault(0x5_0000_0000, 4096, 4096)
    assert why is not None and 'DECERR' in why
    # a buffer that starts inside the low window but runs out of it
    assert arm_fault(0x7FFF_FF00, 4096, 4096) is not None


def test_arm_fault_rejects_partial_beats_and_empty_transfers():
    assert arm_fault(0x1000, 0, 4096) is not None
    assert arm_fault(0x1000, -64, 4096) is not None
    assert arm_fault(0x1000, 100, 4096) is not None    # not a whole beat


# --- when it is safe to give the memory back --------------------------------
#
# abort() is one register write and returns immediately; the core keeps
# writing for a while afterwards. Freeing CMA on the strength of the write
# alone hands live DMA target pages to the next allocation.


def _status(**kw):
    """A decoded cmd_ctrl_reg with everything clear except what is named."""
    stat = dict(r_busy=False, r_err=False, r_complete=False, r_continuous=False,
                r_increment_n=False, r_tlast_syncd_n=False, decode_error=False,
                slave_error=False, overflow_error=False, aborting=False,
                fifo_len=0, abort=0)
    stat.update(kw)
    return stat


def test_quiescent_bits_are_the_two_that_mean_still_writing():
    assert AXIS2MM_QUIESCENT_BITS == ('r_busy', 'aborting')


def test_quiesced_only_when_both_bits_clear():
    assert axis2mm_quiesced(_status())
    assert not axis2mm_quiesced(_status(r_busy=True))
    assert not axis2mm_quiesced(_status(aborting=True))
    assert not axis2mm_quiesced(_status(r_busy=True, aborting=True))


def test_a_pending_error_does_not_block_quiescence():
    # abort sets r_err; clear_error is what happens after the core goes quiet,
    # so requiring r_err clear here would deadlock the settle loop forever.
    assert axis2mm_quiesced(_status(r_err=True, decode_error=True))
    assert not axis2mm_quiesced(_status(r_err=True, aborting=True))
