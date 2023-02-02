import re
import pkg_resources
from logging import getLogger

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
    tpro_file = pkg_resources.resource_filename('mkidgen3','config/ZCU111_LMK04208_10MHz_Ref_J109SMA.txt')
    xrfclk.xrfclk._Config['lmk04208']['122.88_viaext10M'] = _parse_ticspro(tpro_file)


def start_clocks(external_10mhz=False):
    try:
        import xrfclk, xrfdc
    except ImportError:
        getLogger(__name__).warning('xrfclk/xrfdc unavaiable, clock will not be started')
        return
    if external_10mhz:
        _patch_xrfclk_lmk()
        xrfclk.set_ref_clks(lmk_freq='122.88_viaext10M')
    else:
        xrfclk.set_ref_clks()
