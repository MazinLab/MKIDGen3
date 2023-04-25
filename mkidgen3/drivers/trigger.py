from pynq import DefaultIP, buffer
import numpy as np
from logging import getLogger

class PhotonTrigger(DefaultIP):
    ADDR_RESMAP = 0x1000
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
        vals = [self.read(self.ADDR_RESMAP + 8 * group_ndx + 4 * i) for i in range(2)]
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
        self.write(self.ADDR_RESMAP + 16 * group_ndx, data)

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


class PhotonPostageFilter(DefaultIP):
    ADDR_MONITOR_CHAN = tuple(range(0x10, 0x49, 0x8))
    bindto = ['mazinlab:mkidgen3:postage_filter:0.1']

    def __init__(self, description):
        """
        The core watches for trigger events on 8 resonator channels and forwards IQ snippets around trigger events on
        for capture by other parts of the firmware. It is configured with 8 resonator channel values for monitoring.

        control
        0x10 : Data signal of monitor_0
               bit 11~0 - monitor_0[11:0] (Read/Write)
               others   - reserved
        0x18, 0x20, 0x28,0x30,0x38,0x40 0x48 : Data signal of monitor_1 - 7

        """
        super().__init__(description=description)

    @property
    def monitor_channels(self):
        return tuple(self.read(a) & 0x7ff for a in self.ADDR_MONITOR_CHAN)

    def configure(self, monitor_channels=None):
        """ monitor_channels shall be 8 integers in [0,2047] """
        if monitor_channels is None:
            return
        monitor_channels = np.asarray(monitor_channels, dtype=int).clip(0, 2047)
        for a, c in zip(self.ADDR_MONITOR_CHAN, monitor_channels):
            self.write(a, c)


class PhotonPostageMAXI(DefaultIP):
    POSTAGE_BUFFER_LEN = 1000
    N_CAPDATA = 90
    bindto = ['mazinlab:mkidgen3:photons_maxi_id:0.1']

    def __init__(self, description):
        """
        The core captures up to some number of Photon events to a pynq buffer. The resonators for monitoring are
        configured by PhotonPostageFilter.

        // control
        // 0x00 : Control signals
        //        bit 0  - ap_start (Read/Write/COH)
        //        bit 1  - ap_done (Read/COR)
        //        bit 2  - ap_idle (Read)
        //        bit 3  - ap_ready (Read/COR)
        //        bit 7  - auto_restart (Read/Write)
        //        bit 9  - interrupt (Read)
        //        others - reserved
        // 0x04 : Global Interrupt Enable Register
        //        bit 0  - Global Interrupt Enable (Read/Write)
        //        others - reserved
        // 0x08 : IP Interrupt Enable Register (Read/Write)
        //        bit 0 - enable ap_done interrupt (Read/Write)
        //        bit 1 - enable ap_ready interrupt (Read/Write)
        //        others - reserved
        // 0x0c : IP Interrupt Status Register (Read/COR)
        //        bit 0 - ap_done (Read/COR)
        //        bit 1 - ap_ready (Read/COR)
        //        others - reserved
        // 0x10 : Data signal of iq
        //        bit 31~0 - iq[31:0] (Read/Write)
        // 0x14 : Data signal of iq
        //        bit 31~0 - iq[63:32] (Read/Write)
        // 0x20 ~
        // 0x2f : Memory 'event_count' (8 * 16b)
        //        Word n : bit [15: 0] - event_count[2n]
        //                 bit [31:16] - event_count[2n+1]

        """
        super().__init__(description=description)
        self._buf = None

    def capture(self):
        if not self.register_map.AP_CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf
            raise RuntimeError('Core already capturing')
        self.write(0x08, 1)
        self.write(0x04, 1)
        self.read(0x0C)
        self._buf = buffer.allocate((8, self.POSTAGE_BUFFER_LEN, self.N_CAPDATA, 2), dtype=np.int16)
        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())
        self.register_map.AP_CTRL.AP_START=1
        return self._buf

    @property
    def event_count(self):
        return np.frombuffer(self.mmio.array[0x20/4:0x2f/4+1].copy(), dtype=np.uint16)

    def configure(self):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture()


class PhotonIDMAXI(DefaultIP):
    N_PHOTON_BUFFERS=2
    PHOTON_BUFF_N=8192
    PHOTON_DTYPE = np.dtype([('time', np.uint16), ('phase', np.int16), ('id', np.int16)])
    bindto = ['mazinlab:mkidgen3:photons_maxi_id:0.1']

    def __init__(self, description):
        """
        The core watches for trigger events on 8 resonator channels and forwards IQ snippets around trigger events on
        for capture by other parts of the firmware. It is configured with 8 resonator channel values for monitoring.

        // control
        // 0x00 : Control signals
        //        bit 0  - ap_start (Read/Write/COH)
        //        bit 1  - ap_done (Read/COR)
        //        bit 2  - ap_idle (Read)
        //        bit 3  - ap_ready (Read/COR)
        //        bit 7  - auto_restart (Read/Write)
        //        bit 9  - interrupt (Read)
        //        others - reserved
        // 0x04 : Global Interrupt Enable Register
        //        bit 0  - Global Interrupt Enable (Read/Write)
        //        others - reserved
        // 0x08 : IP Interrupt Enable Register (Read/Write)
        //        bit 0 - enable ap_done interrupt (Read/Write)
        //        bit 1 - enable ap_ready interrupt (Read/Write)
        //        others - reserved
        // 0x0c : IP Interrupt Status Register (Read/COR)
        //        bit 0 - ap_done (Read/COR)
        //        bit 1 - ap_ready (Read/COR)
        //        others - reserved
        // 0x10 : Data signal of photons_out
        //        bit 31~0 - photons_out[31:0] (Read/Write)
        // 0x14 : Data signal of photons_out
        //        bit 31~0 - photons_out[63:32] (Read/Write)
        // 0x18 : reserved
        // 0x28 : Data signal of active_buffer
        //        bit 7~0 - active_buffer[7:0] (Read)
        //        others  - reserved
        // 0x2c : Control signal of active_buffer
        //        bit 0  - active_buffer_ap_vld (Read/COR)
        //        others - reserved
        // 0x20 ~
        // 0x27 : Memory 'n_photons' (2 * 13b)
        //        Word n : bit [12: 0] - n_photons[2n]
        //                 bit [28:16] - n_photons[2n+1]
        //                 others      - reserved
        """
        super().__init__(description=description)
        self._buf = None

    def get_photons(self):
        ab = self.read(0x28) & 0xff
        ret = self._buf[0 if ab else 1].copy()
        count = self.read(0x20)
        ab2 = self.read(0x28) & 0xff
        counts = [count & 0x1fff, (count >> 16) & 0x1fff]
        getLogger(__name__).debug(f'Active Buffer: {ab}. Buffer counts: {counts}')
        if not ab == ab2:
            raise RuntimeError('Buffer changed during read')
        return ret[:count]

    def capture(self):
        if not self.register_map.AP_CTRL.AP_IDLE:
            getLogger(__name__).debug('Core already capturing, returning existing buffer')
            return self._buf
        self.write(0x08, 1)
        self.write(0x04, 1)
        self.read(0x0C)
        self._buf = buffer.allocate((self.N_PHOTON_BUFFERS, self.PHOTON_BUFF_N), dtype=self.PHOTON_DTYPE)
        self.write(0x10, np.asarray([self._buf.device_address]).tobytes())
        self.register_map.AP_CTRL.AP_AUTO_RESTART = 1
        self.register_map.AP_CTRL.AP_START = 1
        return self._buf

    def configure(self):
        """ monitor_channels shall be 8 integers in [0,2047] """
        return self.capture()
