"""Off-board tests for the stage-2 IQ transform driver.

The CSR decoders and table vetting are pure functions. The register
sequences (write_table / commit / bypass) run against a dict-backed fake
register file that models auto-increment and the commit handshake, so the
ordering is checked here; the actual AXI transactions are board-verified by
tools/stage2_loopback.py and tools/stage2_postage_closure.py.
"""
import numpy as np
import pytest

from mkidgen3.drivers.iqtransform import (IQTransform, decode_control,
                                          decode_version, decode_format,
                                          decode_format2, vet_table,
                                          EXPECTED_VERSION, EXPECTED_FORMAT,
                                          EXPECTED_FORMAT2, N_TABLE_WORDS,
                                          CLAMP_THRESHOLD_RESET)

# Word values this bitstream reports (handoff s1.1), assembled LSB-first.
VERSION_WORD = 0x01889401
FORMAT_WORD = 0x2CA231AF
FORMAT2_WORD = 0x0830DB08


class FakeTransform(IQTransform):
    """Register model. Deliberately does not call DefaultIP.__init__."""

    def __init__(self, version=VERSION_WORD, fmt=FORMAT_WORD, fmt2=FORMAT2_WORD):
        self.control = 0b0_1000          # write_bank=1, bypass=0, reset state
        self.index = 0
        self.mem = np.zeros(N_TABLE_WORDS, dtype=np.uint32)
        self.clamp = CLAMP_THRESHOLD_RESET
        self.pending_frames = 0          # commits left un-taken by the fabric
        self.writes = []                 # (offset, value) trace
        self._ro = {0x10: version, 0x14: fmt, 0x18: fmt2}

    def read(self, offset):
        if offset in self._ro:
            return self._ro[offset]
        if offset == 0x00:
            return self.control | (0b100 if self.pending_frames else 0)
        if offset == 0x04:
            return (self.index & 0x3fff) << 14
        if offset == 0x0c:
            return self.clamp
        raise AssertionError(f'read of undocumented offset {offset:#x}')

    def write(self, offset, value):
        self.writes.append((offset, int(value)))
        if offset == 0x00:
            # commit is write-only and self-clearing; bypass/manual_index stick
            self.control = int(value) & 0b1_0001 | (self.control & 0b1000)
            if int(value) & 0b10:
                self.pending_frames = 2   # two frames until every lane swaps
        elif offset == 0x04:
            self.index = int(value) & 0x3fff
        elif offset == 0x08:
            self.mem[self.index] = np.uint32(int(value) & 0xffffffff)
            self.index += 1
        else:
            raise AssertionError(f'write to undocumented offset {offset:#x}')

    def tick(self):
        """One frame of streaming: the fabric takes one step of the commit."""
        if self.pending_frames:
            self.pending_frames -= 1


def a_valid_table():
    rows = np.zeros((2048, 8), dtype=np.int64)
    rows[:, 0] = 1000            # b1
    rows[:, 1] = -2000           # b2
    rows[:, 2] = 30              # a3
    rows[:, 3] = 1000            # b4
    rows[:, 4] = -2000           # b5
    rows[:, 5] = -30             # a6
    rows[:, 6] = 1 << 14         # n = nominal off-resonance amplitude
    return rows


def test_decoders_match_the_bitstream():
    assert decode_version(VERSION_WORD) == EXPECTED_VERSION
    assert decode_version(VERSION_WORD) == dict(version=1, lanes=4, beat_bits=9,
                                                columns=8, latency=24)
    assert decode_format(FORMAT_WORD) == EXPECTED_FORMAT
    assert decode_format(FORMAT_WORD) == dict(in_frac=15, out_frac_rad=13,
                                              c_frac=12, off_frac=4, guard=10,
                                              recip_frac=22)
    assert decode_format2(FORMAT2_WORD) == EXPECTED_FORMAT2
    assert decode_format2(FORMAT2_WORD) == dict(lut_addr_bits=8, mant_bits=24,
                                                c_bits=27, off_bits=24,
                                                n_bits=16)


def test_decode_control():
    assert decode_control(0b1_0000_1001) == dict(bypass=True, pending=False,
                                                 write_bank=True,
                                                 manual_index=False, overflow=8)
    assert decode_control(0b0_0100) == dict(bypass=False, pending=True,
                                            write_bank=False,
                                            manual_index=False, overflow=0)


def test_check_identity_accepts_this_bitstream():
    FakeTransform().check_identity()


@pytest.mark.parametrize('reg,word', [('version', 0x01889402),
                                      ('fmt', 0x2CA231AE),
                                      ('fmt2', 0x0830DB07)])
def test_check_identity_raises_on_any_field_mismatch(reg, word):
    t = FakeTransform(**{reg: word})
    with pytest.raises(RuntimeError, match='quantization'):
        t.check_identity()


def test_vet_table_shape_and_ranges():
    assert vet_table(a_valid_table()).shape == (2048, 8)
    assert vet_table(a_valid_table()).dtype == np.uint32
    with pytest.raises(ValueError, match='shape'):
        vet_table(np.zeros((2048, 7), dtype=np.int64))
    for col, bad in ((0, 1 << 26), (1, -(1 << 26)), (2, 1 << 23),
                     (3, 1 << 26), (4, 1 << 26), (5, -(1 << 23))):
        rows = a_valid_table()
        rows[7, col] = bad
        with pytest.raises(ValueError, match='channel 7'):
            vet_table(rows)
    rows = a_valid_table()
    rows[3, 6] = 1 << 16
    with pytest.raises(ValueError, match='channel 3'):
        vet_table(rows)
    rows = a_valid_table()
    rows[5, 7] = 1
    with pytest.raises(ValueError, match='channel 5'):
        vet_table(rows)


def test_vet_table_encodes_negatives_twos_complement():
    rows = a_valid_table()
    words = vet_table(rows)
    assert words[0, 1] == np.uint32((-2000) & 0xffffffff)
    # uint32 input (already encoded) round-trips through unchanged
    assert (vet_table(words) == words).all()


def test_write_table_is_one_index_write_then_16384_data_writes():
    t = FakeTransform()
    rows = a_valid_table()
    rows[9, 6] = 4242
    t.write_table(rows)
    index_writes = [w for w in t.writes if w[0] == 0x04]
    data_writes = [w for w in t.writes if w[0] == 0x08]
    assert index_writes == [(0x04, 0)]
    assert len(data_writes) == N_TABLE_WORDS
    assert t.mem[9 * 8 + 6] == 4242
    assert (t.mem == vet_table(rows).reshape(-1)).all()


def test_commit_polls_until_pending_clears():
    t = FakeTransform()
    t.bypass = True
    calls = {'n': 0}
    real_read = t.read

    def read(offset):
        if offset == 0x00:
            calls['n'] += 1
            t.tick()
        return real_read(offset)

    t.read = read
    t.commit(timeout_s=1.0)
    ctrl_writes = [w for w in t.writes if w[0] == 0x00]
    # the commit write preserves bypass (bit 0) and manual_index (bit 4)
    assert ctrl_writes[-1] == (0x00, 0b11)
    assert t.pending_frames == 0


def test_commit_timeout_names_the_clock_trap():
    t = FakeTransform()   # never ticks: no TLAST, so pending never clears
    with pytest.raises(TimeoutError, match='512 MHz'):
        t.commit(timeout_s=0.05)


def test_bypass_roundtrip_preserves_other_control_bits():
    t = FakeTransform()
    assert t.bypass is False
    t.bypass = True
    assert t.bypass is True
    assert t.write_bank is True          # untouched by the bypass write
    t.bypass = False
    assert t.bypass is False


def test_status_reports_every_readable_field():
    t = FakeTransform()
    s = t.status()
    assert s['version'] == 1 and s['lanes'] == 4 and s['beat_bits'] == 9
    assert s['columns'] == 8 and s['latency'] == 24
    assert s['in_frac'] == 15 and s['recip_frac'] == 22 and s['n_bits'] == 16
    assert s['bypass'] is False and s['pending'] is False
    assert s['write_bank'] is True and s['overflow'] == 0
    assert s['clamp_threshold'] == CLAMP_THRESHOLD_RESET
