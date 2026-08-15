import numpy as np
import pynq
from logging import getLogger
from mkidgen3.mkidpynq import check_description_for
from mkidgen3.fixedpoint import fp_factory
from mkidgen3.recordfmt import (DEFAULT_RECORD_VERSION, FIR_CONFIG_TDEST,
                                FIR_LANES_BY_VERSION,
                                FIR_QUADRATURES_BY_VERSION, FIR_SETS_BY_VERSION,
                                FIR_TAPS, check_version, fir_config_packet,
                                fir_reload_last_bytes, fir_reload_packet,
                                fir_taps_by_quadrature,
                                fir_taps_from_description, fir_tdest,
                                unity_coefficient_sets)


class PhasematchDriver(pynq.DefaultHierarchy):
    """Matched filter reload over the AXI FIFO at 0x800E_0000.

    The bank layout depends on the record version the overlay implements
    (see mkidgen3.recordfmt):

    * v2 -- one quadrature on four lanes, lane = r % 4, 512 sets per bank,
      reload TDEST = lane, config packet arange(512) on TDEST 4.
    * v3 (stage 1/2) -- two quadratures on two lanes each, lane = r % 2,
      1024 sets per bank, TH2 reload TDEST = lane, D2 reload TDEST =
      2 + lane, config packet arange(1024) on TDEST 4. Both quadratures of a
      channel go out together and are committed by the same config packet,
      so the two planes can never carry templates from different loads.

    The driver defaults to v2. Whoever knows which bitstream is loaded (the
    daemon, from the hwh) sets ``record_version`` before loading taps.
    """

    N_TEMPLATE_TAPS = FIR_TAPS
    N_RES = 2048
    N_SLOTS = 2
    N_FIFO_SIZE = 512
    COEFF_FORMAT = (1, 15)

    def __init__(self, description):
        super().__init__(description)
        self.fifo = self.reload.axi_fifo_mm_s_0
        self._taps_by_tdest = fir_taps_from_description(description)
        self._record_version = DEFAULT_RECORD_VERSION
        self._pending = {}          # reload TDEST -> pending reload slots

    @staticmethod
    def checkhierarchy(description):
        if 'reload' not in description.get('hierarchies', {}):
            return False
        return bool(len(check_description_for(description['hierarchies']['reload'], 'xilinx.com:ip:axi_fifo_mm_s')))

    @property
    def record_version(self):
        """Record version whose bank geometry this driver is using."""
        return self._record_version

    @record_version.setter
    def record_version(self, version):
        v = check_version(version)
        if v != self._record_version:
            # Validate before publishing the new geometry. A v3 mixed-width
            # hierarchy is deliberately invalid under the v2 four-lane view,
            # which is why this is not done against the default in __init__.
            fir_taps_by_quadrature(self._taps_by_tdest, v)
            self._record_version = v
            self._pending = {}      # slot accounting is per-bank, per-geometry

    @property
    def n_lanes(self):
        return FIR_LANES_BY_VERSION[self._record_version]

    @property
    def n_sets(self):
        return FIR_SETS_BY_VERSION[self._record_version]

    @property
    def quadratures(self):
        return FIR_QUADRATURES_BY_VERSION[self._record_version]

    @property
    def taps_by_tdest(self):
        """Read-only reload widths keyed by the switch's TDEST."""
        return {tdest: taps for tdest, taps in enumerate(self._taps_by_tdest)}

    @property
    def filter_taps(self):
        """Read-only FIR widths keyed by ``th2`` and, on v3, ``d2``."""
        return fir_taps_by_quadrature(
            self._taps_by_tdest, self._record_version
        )

    def tap_count(self, quadrature):
        try:
            return self.filter_taps[quadrature]
        except KeyError as error:
            raise ValueError(
                f'record version {self._record_version} has no '
                f'{quadrature.upper()} quadrature'
            ) from error

    @staticmethod
    def vet_coeffs(coeffs, expected_taps=FIR_TAPS, label='coefficients'):
        array = np.asarray(coeffs)
        actual_taps = array.shape[-1] if array.ndim else 0
        if actual_taps != expected_taps:
            raise ValueError(
                f'{label} expected {expected_taps} taps '
                f'({expected_taps + 1} reload words), got {actual_taps} '
                f'({actual_taps + 1} reload words)'
            )
        if array.dtype != np.int16 and abs(array).max() > 1:
            raise ValueError(f'Coefficients must be <= 1 if floating point')

    @staticmethod
    def vet_res_id(res_id):
        if 0 > res_id or res_id >= PhasematchDriver.N_RES:
            raise ValueError(f'resID must be in [0-{PhasematchDriver.N_RES}-1]')

    @staticmethod
    def reorder_coeffs(coeffs):
        """convert taps to order needed by a reload packet"""
        return coeffs[::-1]  # see coefficient reload tab for order in block design

    def _send_config(self, wait=False):
        """Commit every pending reload slot on every bank.

        Chunked because the v3 commit vector (1024 uint16 = 512 words)
        exceeds the 512-deep TX FIFO's usable vacancy (depth - 4, PG080);
        the config channel consumes vector entries in order across the
        resulting TLAST boundaries. ``wait`` is retained for signature
        compatibility: the chunked path already polls the FIFO for drain,
        which is the stronger guarantee.
        """
        self.fifo.tx_chunked(fir_config_packet(self._record_version),
                             destination=FIR_CONFIG_TDEST)
        self._pending = {}

    def load_coeff(self, res_id, coeffs, d2_coeffs=None, vet=True, force_commit=False,
                   raw=False, wait=False, defer_commit=False):
        """
        A reload packet consists of the coefficient set number and the coefficients.

        If raw coeffs will be converted to np.uint16 via numpy casting/type coercion rules.

        d2_coeffs is the second (D2) quadrature and requires a v3 overlay. On v3 it
        defaults to zeros -- a muted quadrature is a valid single-quadrature
        configuration, whereas leaving the shipped placeholder in place (tap 29 = -1,
        i.e. -90 dB through the >>15 output stage) would add a faint copy of the
        signal to every channel.

        FIRs have two reload slots and are in "on vector" update mode; a config packet
        is sent before a bank would need a third.

        set defer_commit to skip sending a config packet even if the packet sent filled
        up the number of usable slots

        See pg149 pg 18
        """
        version = self._record_version
        if vet:
            self.vet_res_id(res_id)
            self.vet_coeffs(coeffs, self.tap_count('th2'), 'TH2 coefficients')
            if d2_coeffs is not None:
                self.vet_coeffs(
                    d2_coeffs, self.tap_count('d2'), 'D2 coefficients'
                )
        if d2_coeffs is not None and 'd2' not in self.quadratures:
            raise ValueError('this overlay has no D2 quadrature (record '
                             f'version {version})')

        if raw:
            fp_format = lambda x: x
        else:
            fp_format = fp_factory(*self.COEFF_FORMAT, True, include_index=True)

        payloads = [('th2', coeffs)]
        if 'd2' in self.quadratures:
            payloads.append(('d2', np.zeros(self.tap_count('d2'), dtype=np.int16)
                             if d2_coeffs is None else d2_coeffs))
        packets = []
        for quadrature, taps in payloads:
            tdest = fir_tdest(res_id, version, quadrature)
            n_taps = self.tap_count(quadrature)
            packets.append((
                tdest,
                fir_reload_packet(
                    res_id, [fp_format(c) for c in taps], version,
                    expected_taps=n_taps, tdest=tdest
                ),
                fir_reload_last_bytes(n_taps),
            ))

        if any(self._pending.get(t, 0) >= self.N_SLOTS
               for t, _, _ in packets):
            getLogger(__name__).debug('Reload slots are full, sending config packet first')
            self._send_config(wait)

        for tdest, packet, last_bytes in packets:
            self.fifo.tx(packet, destination=tdest,
                         last_bytes=last_bytes, wait=wait)
            self._pending[tdest] = self._pending.get(tdest, 0) + 1

        if force_commit or (not defer_commit
                            and any(self._pending[t] >= self.N_SLOTS
                                    for t, _, _ in packets)):
            if not force_commit:
                getLogger(__name__).debug('Sending config packet')
            self._send_config(wait)

    def load_coeff_sets(self, coeff_sets, d2_coeff_sets=None, raw=False):
        """
        Program coefficients for all the resonator channels
        Args:
            coeff_sets: (N_RES, N_TAP) array of coefficients, will be vetted
            d2_coeff_sets: (N_RES, N_TAP) second quadrature, v3 overlays only.
                None writes zeros to the D2 banks.
            raw: (optional) whether to load the coefficients as is or convert to fixed
                point, see load_coeff

        Returns: None
        """
        self.vet_coeffs(
            coeff_sets, self.tap_count('th2'), 'TH2 coefficient bank'
        )
        if d2_coeff_sets is not None:
            self.vet_coeffs(
                d2_coeff_sets, self.tap_count('d2'), 'D2 coefficient bank'
            )
        for res in range(self.N_RES):
            self.load_coeff(res, coeff_sets[res],
                            d2_coeffs=None if d2_coeff_sets is None else d2_coeff_sets[res],
                            vet=False, defer_commit=True, force_commit=res == self.N_RES - 1,
                            wait=False, raw=raw)

    @staticmethod
    def _unity_request(coefficients):
        """The channel count of a ``'unityN'`` request, or None if not one."""
        if not (isinstance(coefficients, str)
                and coefficients.startswith('unity')):
            return None
        try:
            return min(max(1, int(coefficients.strip('unity'))), 2048)
        except ValueError:
            return 2048

    def configure(self, coefficients=None, d2_coefficients=None):
        if coefficients is None:
            return
        getLogger(__name__).info(f'Configuring phasematch with {coefficients}')
        # Both quadratures accept a 'unityN' string -- the daemon's raw-phase
        # path sends the same string for TH2 and D2 on dual-quad builds, and
        # only converting the first one handed d2_coefficients (a str) to the
        # .shape check below ('str' object has no attribute 'shape',
        # 2026-08-12 first Load Cal on s2t4dnp).
        n = self._unity_request(coefficients)
        if n is not None:
            # v2: 2**15 - 1 (historical). v3: -32768, the only representable
            # unity magnitude in signed 16 bits, which inverts the stream.
            coefficients = unity_coefficient_sets(n, self._record_version,
                                                  n_res=self.N_RES,
                                                  n_taps=self.tap_count('th2'))
        n = self._unity_request(d2_coefficients)
        if n is not None:
            d2_coefficients = unity_coefficient_sets(n, self._record_version,
                                                     n_res=self.N_RES,
                                                     n_taps=self.tap_count('d2'))
        th2_taps = self.tap_count('th2')
        if coefficients.shape != (self.N_RES, th2_taps) \
                or coefficients.dtype != 'int16':
            actual_taps = coefficients.shape[-1] if coefficients.ndim else 0
            raise ValueError(
                f'coefficients must be a ({self.N_RES},{th2_taps}) int16 '
                f'array ({th2_taps + 1} reload words), got shape '
                f'{coefficients.shape} ({actual_taps + 1} reload words)'
            )
        if d2_coefficients is not None:
            if 'd2' not in self.quadratures:
                raise ValueError('this overlay has no D2 quadrature (record '
                                 f'version {self._record_version})')
            d2_taps = self.tap_count('d2')
            if d2_coefficients.shape != (self.N_RES, d2_taps) \
                    or d2_coefficients.dtype != 'int16':
                actual_taps = (d2_coefficients.shape[-1]
                               if d2_coefficients.ndim else 0)
                raise ValueError(f'd2_coefficients must be a ({self.N_RES},'
                                 f'{d2_taps}) int16 array ({d2_taps + 1} '
                                 f'reload words), got shape '
                                 f'{d2_coefficients.shape} '
                                 f'({actual_taps + 1} reload words)')

        self.load_coeff_sets(coefficients, d2_coeff_sets=d2_coefficients, raw=True)
