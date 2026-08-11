"""HP0 reachability for CMA capture buffers on NODDR overlays.

On a NODDR build axis2mm writes into PS DDR through HP0, and an address
outside the two HP0 windows DECERRs with no symptom the daemon can see. The
check is pure arithmetic and lives in mkidgen3.mkidpynq; the allocation path
that uses it needs pynq and is board-verified by tools/sweepacc_bringup.py.
"""
from mkidgen3.mkidpynq import HP0_WINDOWS, hp0_reachable


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
