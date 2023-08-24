import re
import pkg_resources
from logging import getLogger
from mkidgen3.mkidpynq import get_board_name


def _parse_ticspro(file):
    with open(file, 'r') as f:
        lines = [l.rstrip("\n") for l in f]

        registers = []
        for i in lines:
            m = re.search('[\t]*(0x[0-9A-F]*)', i)
            registers.append(int(m.group(1), 16), )
    return registers


def _patch_xrfclk_lmk():
    # access with     xrfdc.set_ref_clks(lmk_freq='122.88_viaext10M')
    import xrfclk

    _LMK04208_FILES = {
        '122.88_viaext10M': 'config/ZCU111_LMK04208_10MHz_Ref_J109SMA.txt'
    }

    _LMK04828_FILES = {
        '256.0_MTS': 'config/LMK04828_256.0_MTS.txt',
        '512.0_MTS': 'config/LMK04828_512.0_MTS.txt',
        '512.0_MTS_dualloop': 'config/LMK04828_512_dualloop.txt',
        '500.0_MTS': 'config/LMK04828_500.0_MTS.txt'
    }

    _LMX2594_FILES = {
        '500.0_MTS': 'config/LMX2594_500.0_MTS.txt',
        '512.0_MTS': 'config/LMX2594_512.0_MTS.txt',
        '512.0_MTS_dualloop': 'config/LMX2594_512.0_MTS.txt',
        '409.6_MTS': 'config/LMX2594_409.6_256FoscMTS.txt'
    }

    _CLOCK_CONFIG_DICT = {
    'lmk04208': _LMK04208_FILES,
    'lmk04828': _LMK04828_FILES,
    'lmx2594': _LMX2594_FILES
    }

    for clock_part in _CLOCK_CONFIG_DICT:
        for programming_key, fname in _CLOCK_CONFIG_DICT[clock_part].items():
            tpro_file = pkg_resources.resource_filename('mkidgen3',fname)
            xrfclk.xrfclk._Config[clock_part][programming_key] = _parse_ticspro(tpro_file)


def start_clocks(programming_key='', ref='file-defined'):
    """
    - 'external_10mhz' pull LMK clock source from 10 MHz Ref (ZCU111 Only for now)
    - '4.096GSPS_MTS' MTS compatible with 4.096 GSPS Sampling Fequency (RFSoC4x2 Only)
    - '5.000GSPS_MTS' MTS compatible with 5.000 GSPS Sampling Frequency (RFSoC4x2 Only)
    """
    try:
        import xrfclk, xrfdc
    except ImportError:
        getLogger(__name__).warning('xrfclk/xrfdc unavaiable, clock will not be started')
        return
    if programming_key:
        _patch_xrfclk_lmk()

    board_name = get_board_name()

    if board_name == 'RFSoC4x2':
        if programming_key == '4.096GSPS_MTS':
            xrfclk.set_ref_clks(lmk_freq='512.0_MTS', lmx_freq='512.0_MTS')
        if programming_key == '4.096GSPS_MTS_dualloop':
            xrfclk.set_ref_clks(lmk_freq='512.0_MTS_dualloop', lmx_freq='512.0_MTS_dualloop')
        elif programming_key == '5.000GSPS_MTS':
            xrfclk.set_ref_clks(lmk_freq='500.0_MTS', lmx_freq='500.0_MTS')
        else:
            xrfclk.set_ref_clks(lmk_freq=245.76, lmx_freq=409.6)
        for lmk in xrfclk.lmk_devices:
            if ref == 'external':
                xrfclk.xrfclk._write_LMK_regs([0x1470a], lmk)
            if ref == 'internal':
                xrfclk.xrfclk._write_LMK_regs([0x1471a], lmk)

    elif board_name == 'ZCU111':
        if programming_key == 'external_10mhz':
            xrfclk.set_ref_clks(lmk_freq='122.88_viaext10M', lmx_freq=409.6)
        else:
            xrfclk.set_ref_clks(lmk_freq=122.88, lmx_freq=409.6)

    else:
        raise ValueError('Unknown board name. Cannot proceed with clock programming.')
