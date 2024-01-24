import numpy as np
import pynq
from logging import getLogger
from mkidgen3.mkidpynq import check_description_for
from mkidgen3.fixedpoint import fp_factory
from mkidgen3.util import pack16_to_32
import time


class PhasematchDriver(pynq.DefaultHierarchy):
    N_TEMPLATE_TAPS = 30
    N_RES = 2048
    N_RES_P_LANE = 512
    N_LANES = 4
    N_SLOTS = 2
    N_FIFO_SIZE = 512
    COEFF_FORMAT = (1, 15)

    def __init__(self, description):
        super().__init__(description)
        self.fifo = self.reload.axi_fifo_mm_s_0
        self._pending = np.zeros(self.N_LANES, dtype=int)

    @staticmethod
    def checkhierarchy(description):
        if 'reload' not in description.get('hierarchies', {}):
            return False
        return bool(len(check_description_for(description['hierarchies']['reload'], 'xilinx.com:ip:axi_fifo_mm_s')))

    @staticmethod
    def vet_coeffs(coeffs):
        if coeffs.shape[-1] != PhasematchDriver.N_TEMPLATE_TAPS:
            raise ValueError('Incorrect number of taps')
        if np.asarray(coeffs).dtype != np.int16 and abs(np.asarray(coeffs)).max() > 1:
            raise ValueError(f'Coefficients must be <= 1 if floating point')

    @staticmethod
    def vet_res_id(res_id):
        if 0 > res_id or res_id >= PhasematchDriver.N_RES:
            raise ValueError(f'resID must be in [0-{PhasematchDriver.N_RES}-1]')

    @staticmethod
    def reorder_coeffs(coeffs):
        """convert taps to order needed by a reload packet"""
        return coeffs[::-1]  # see coefficient reload tab for order in block design

    def load_coeff(self, res_id, coeffs, vet=True, force_commit=False, raw=False, wait=False, defer_commit=False):
        """
        A reload packet consists of the coefficients and the coefficient set number

        If raw coeffs will be converted to np.uint16 via numpy casting/type coercion rules.

        See block diagram for layout. Resonators assigned to lanes 0-3 in consecutive sets of 512.

        FIRs have two reload slots and are in "on vector" update mode.

        set defer_commit to skip sending a config packet even if the packet sent filled up the number of usable slots

        See pg149 pg 18
        """
        if vet:
            self.vet_res_id(res_id)
            self.vet_coeffs(coeffs)

        if raw:
            fp_format = lambda x: x
        else:
            fp_format = fp_factory(*self.COEFF_FORMAT, True, include_index=True)

        lane = res_id % self.N_LANES
        reload_packet = np.zeros(self.N_TEMPLATE_TAPS + 1, dtype=np.uint16)
        reload_packet[0] = res_id // self.N_LANES
        reload_packet[1:] = [fp_format(c) for c in self.reorder_coeffs(coeffs)]

        cfg_packet = pack16_to_32(np.arange(self.N_RES_P_LANE, dtype=np.uint16))

        if self._pending[lane] >= self.N_SLOTS:
            getLogger(__name__).debug(f'Reload slots for lane {lane} are full, sending config packet first')
            self.fifo.tx(cfg_packet, destination=4, wait=wait)  # Send a config packet to trigger reload
            self._pending[:] = 0

        self.fifo.tx(pack16_to_32(reload_packet), destination=lane, last_bytes=2, wait=wait)  # reload channels are 0,2,4,6
        self._pending[lane] += 1

        if force_commit or (self._pending[lane] >= self.N_SLOTS and not defer_commit):
            if not force_commit:
                getLogger(__name__).debug('Sending config packet')
            self.fifo.tx(cfg_packet, destination=4, wait=wait)  # Send a config packet to trigger reload
            self._pending[:] = 0

    def load_coeff_sets(self, coeff_sets, raw=False):
        """
        Program coefficients for all the resonator channels
        Args:
            coeff_sets: (N_RES, N_TAP) array of coefficients, will be vetted
            raw: (optional) whether to load the coefficients as is or convert to fixed point, see load_coeff

        Returns: None
        """
        self.vet_coeffs(coeff_sets)
        for res in range(self.N_RES):
            self.load_coeff(res, coeff_sets[res], vet=False, defer_commit=True, force_commit=res == self.N_RES-1,
                            wait=False, raw=raw)

    def configure(self, coefficients=None):
        if coefficients is None:
            return
        getLogger(__name__).info(f'Configuring phasematch with {coefficients}')
        if isinstance(coefficients, str) and coefficients.startswith('unity'):
            try:
                n = min(max(1, int(coefficients.strip('unity'))), 2048)
            except:
                n = 2048
            coefficients = np.zeros((2048, self.N_TEMPLATE_TAPS), dtype=np.int16)
            coefficients[:n, 0] = 2 ** 15 - 1
        if coefficients.shape != (2048, self.N_TEMPLATE_TAPS) or coefficients.dtype != 'int16':
            raise ValueError(f'coefficients must be a ({self.N_RES},{self.N_TEMPLATE_TAPS}) int16 array')

        self.load_coeff_sets(coefficients, raw=True)
