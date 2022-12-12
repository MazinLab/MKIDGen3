import mkidgen3 as g3
from mkidgen3.drivers.ifboard import IFBoard
from logging import getLogger, basicConfig

basicConfig()
getLogger("mkidgen3.drivers.ifboard").setLevel("DEBUG")


bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'

ol = g3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)
# Connect IF Board
if_board = IFBoard(connect=True)
if_board.power_off()
if_board.power_on()
if_board.set_lo(5960)
if_board.set_attens(38.34,35.2)
a = if_board.status()
print(a)