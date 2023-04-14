from pynq import DefaultIP
import numpy as np


class PhotonTrigger(DefaultIP):
    resmap_addr = 0x1000
    bindto = ['mazinlab:mkidgen3:trigger:0.2']

    def __init__(self, description):
        """
        The core uses an array of 512 values, each consisting of 4 pairs of values packed into a 64bit dword.
        These pairs of values consist of the threshold (a byte treated as an ap_fixed<8,1> number for comparison
        against the ap_fixed<16,1> phase) and a holdoff value (a byte that specified the number of samples over which to
        pick the minimal phase for a photon).

        The holdoff value is in units of the phase stream sample rate (nominally 1MHz). It must not exceed 245 as the
        driver needs to add 1 to the requested value internally. It is expected that a value below ~7 will cause
        aberrant and unpredictable behavior in the full design even if the core itself supports it

        Although the core will issue information about photon overflow as a stream it does not maintain a register
        for it and, as such, this driver does not provide information in that regard.

        x[0] consists of (from low to high byte) thresh0 hoff0 thresh1 hoff1 thresh2 hoff2 thresh3 hoff3
        x[511] consists of (from low to high byte) thresh2044 hoff2044 ... thresh2047 hoff2047

        control
        0x1000 ~
        0x1fff : Memory 'threshoffs' (512 * 64b)
                 Word 2n   : bit [31:0] - threshoffs[n][31: 0]
                 Word 2n+1 : bit [31:0] - threshoffs[n][63:32]
        """
        super().__init__(description=description)

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 511:
            raise ValueError('group_ndx must be in [0,511]')

    def _read_group(self, group_ndx):
        """return a list of the 8 byte values in the group"""
        self._checkgroup(group_ndx)
        g = 0
        vals = [self.read(self.resmap_addr + 8 * group_ndx + 4 * i) for i in range(2)]
        for i, v in enumerate(vals):
            g |= v << (32 * i)
        return [((g >> (8 * j)) & 0xff) for j in range(8)]

    def _write_group(self, group_ndx, group):
        self._checkgroup(group_ndx)
        if len(group) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        for i, g in enumerate(group):
            bits |= (int(g) & 0xff) << (8 * i)
        data = bits.to_bytes(12, 'little', signed=False)
        self.write(self.resmap_addr + 16 * group_ndx, data)

    def _fetch(self):
        """ Return arrays of the thresholds (in raw format) and the holdoffs assigned to each of the  2048 resonator
        channels
        """
        groups = np.array([self._read_group(i) for i in range(511)], dtype=int)
        thresh = groups[:, ::2].ravel()  # /256-.5
        hoff = groups[:, 1::2].ravel() - 1
        return thresh, hoff

    def configure(self, thresholds=None, holdoffs=None):
        """Thresholds shall be floating point numbers in [-1,1) and will be converted to ap_fixed<8,0>"""
        if thresholds is None and holdoffs is None:
            return

        _thresholds, _holdoffs = None, None
        if thresholds is not None:
            thresholds = (thresholds.clip(-1, 1) * 128).round().astype(int).clip(-128, 127)
            if len(thresholds) != 2048:
                raise ValueError('len(thresholds)!=2048')
        else:
            _thresholds, _holdoffs = self._fetch()

        if holdoffs is not None:
            holdoffs = holdoffs.astype(int).clip(0, 254) + 1
            if len(holdoffs) != 2048:
                raise ValueError('len(holdoffs)!=2048')
        elif _holdoffs is None:
            _thresholds, _holdeffos = self._fetch()

        if thresholds is None:
            thresholds = _thresholds
        if holdoffs is None:
            holdoffs = _holdoffs + 1

        assert thresholds.size == holdoffs.size

        t_iter = iter(thresholds)
        h_iter = iter(holdoffs)
        groups = [g for g in zip(t_iter, h_iter, t_iter, h_iter, t_iter, h_iter, t_iter, h_iter)]

        for i, g in enumerate(groups):
            self._write_group(i, g)
