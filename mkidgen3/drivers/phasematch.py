import numpy as np

from mkidgen3.mkidpynq import FP32_8, pack16_to_32


class PhasematchDriver(DefaultHierarchy):
    N_TEMPLATE_TAPS = 30
    N_RES = 2048
    N_RES_P_LANE = 512
    N_LANES = 4
    MAX_COEFF_VALUE = 127  # 16 bits, 1 sign, 8 fractional

    def __init__(self, description):
        super().__init__(description)
        self.fifo = self.axi_fifo_mm_s_0

    @staticmethod
    def hierarchy(description):
        for k in ('axi_fifo_mm_s_0',):
            if k not in description['ip']:
                return False
        return True

    @staticmethod
    def vet_coeffs(coeffs):
        if coeffs.size != PhasematchDriver.N_TEMPLATE_TAPS:
            raise ValueError('Incorrect number of taps')
        if max(abs(coeffs)) > PhasematchDriver.MAX_COEFF_VALUE:
            raise ValueError(f'Coefficients must be <= {PhasematchDriver.MAX_COEFF_VALUE}')

    @staticmethod
    def vet_res_id(res_id):
        if 0 > res_id or res_id >= PhasematchDriver.N_RES:
            raise ValueError(f'resID must be in [0-{PhasematchDriver.N_RES}]')

    @staticmethod
    def reorder_coeffs(coeffs):
        """convert taps to order needed by a reload packet"""
        PhasematchDriver.vet_coeffs(coeffs)
        return coeffs[::-1]  # see coefficient reload tab for order in block design

    def load_coeff(self, res_id, coeffs):
        """

        A reload packet consists of the coefficients and the coefficient set number

        See block diagram for layout. Resonators assigned to lanes 0-3 in consecutive sets of 512.

        FIRs have one reload slot and are in "on vector" update mode.

        See pg149 pg 18
        """
        self.vet_res_id(res_id)
        lane = res_id // PhasematchDriver.N_RES_P_LANE
        reload_packet = np.zeros(coeffs.size + 1, dtype=np.uint16)
        reload_packet[0] = res_id % PhasematchDriver.N_RES_P_LANE
        reload_packet[1:] = [FP32_8(c).__index__() for c in PhasematchDriver.reorder_coeffs(coeffs)]

        cfg_packet = np.arange(PhasematchDriver.N_RES_P_LANE, dtype=np.uint16)

        reload_packet = pack16_to_32(reload_packet)
        cfg_packet = pack16_to_32(cfg_packet)

        self.fifo.tx(reload_packet, destination=lane * 2, last_bytes=2)  # reload channels are 0,2,4,6
        self.fifo.tx(cfg_packet, destination=2 * lane + 1)  # Send a config packet to trigger the reload

    def load_coeff_sets(self, coeff_sets):
        for res in range(PhasematchDriver.N_RES):
            self.load_coeff(res, coeff_sets[res])