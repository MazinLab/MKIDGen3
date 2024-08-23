import re
import importlib.resources
from logging import getLogger
from mkidgen3.mkidpynq import get_board_name

_clock_reference = None
_source = None

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
        '4096.0_MTS': 'config/LMK04828_512_dualloop.txt',
        '500.0_MTS': 'config/LMK04828_500.0_MTS.txt'
    }

    _LMX2594_FILES = {
        '500.0_MTS': 'config/LMX2594_500.0_MTS.txt',
        '512.0_MTS': 'config/LMX2594_512.0_MTS.txt',
        '512.0_MTS_dualloop': 'config/LMX2594_512.0_MTS.txt',
        '4096.0_MTS': 'config/LMX2594_4096.0.txt',
        '409.6_MTS': 'config/LMX2594_409.6_256FoscMTS.txt'
    }

    _CLOCK_CONFIG_DICT = {
    'lmk04208': _LMK04208_FILES,
    'lmk04828': _LMK04828_FILES,
    'lmx2594': _LMX2594_FILES
    }

    for clock_part in _CLOCK_CONFIG_DICT:
        for programming_key, fname in _CLOCK_CONFIG_DICT[clock_part].items():
            tpro_file = importlib.resources.files('mkidgen3').joinpath(fname)
            xrfclk.xrfclk._Config[clock_part][programming_key] = _parse_ticspro(tpro_file)


def configure(programming_key: str | None = None, clock_source: str| None = None):
    """

    Args:
        programming_key: Ignored on the ZCU111.
            '4.096GSPS_MTS' MTS compatible with 4.096 GSPS Sampling Fequency (RFSoC4x2 Only)
            '4.096GSPS_MTS_dualloop'
            '4.096GSPS_MTS_direct'
            '5.000GSPS_MTS' MTS compatible with 5.000 GSPS Sampling Frequency (RFSoC4x2 Only)
        clock_source: internal | external | None. External pulls LMK clock source from 10 MHz Ref

    Returns: Nothing

    """
    global _clock_reference, _source

    programming_key = '' if programming_key is None else programming_key
    clock_source = 'internal' if clock_source is None else clock_source

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
        elif programming_key == '4.096GSPS_MTS_dualloop':
            xrfclk.set_ref_clks(lmk_freq='512.0_MTS_dualloop', lmx_freq='512.0_MTS_dualloop')
        elif programming_key == '4.096GSPS_MTS_direct':
            xrfclk.set_ref_clks(lmk_freq='4096.0_MTS', lmx_freq='4096.0_MTS')
        elif programming_key == '5.000GSPS_MTS':
            xrfclk.set_ref_clks(lmk_freq='500.0_MTS', lmx_freq='500.0_MTS')
        else:
            getLogger(__name__).debug('No programming key specified, defaulting to lmk freq: 245.76 MHz, lmx freq: 409.6 MHz')
            xrfclk.set_ref_clks(lmk_freq=245.76, lmx_freq=409.6)
        for lmk in xrfclk.lmk_devices:
            if clock_source == 'external':
                getLogger(__name__).debug('setting lmk to use external 10 MHz reference')
                xrfclk.xrfclk._write_LMK_regs([0x1470a], lmk)
            if clock_source == 'internal':
                getLogger(__name__).debug('setting lmk to use internal 10 MHz reference')
                xrfclk.xrfclk._write_LMK_regs([0x1471a], lmk)

    elif board_name == 'ZCU111':
        if clock_source == 'external':
            xrfclk.set_ref_clks(lmk_freq='122.88_viaext10M', lmx_freq=409.6)
        else:
            xrfclk.set_ref_clks(lmk_freq=122.88, lmx_freq=409.6)

    else:
        raise ValueError('Unknown board name. Cannot proceed with clock programming.')

    _clock_reference = clock_source
    _source = programming_key


def get_clock():
    return _source, _clock_reference
