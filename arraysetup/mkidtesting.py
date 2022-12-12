import mkidgen3 as g3
from mkidgen3.drivers.ifboard import IFBoard

bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'

ol = g3.overlay_helpers.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=True, download=True)
# Connect IF Board
if_board = IFBoard(connect=True)
if_board.power_on()
a = if_board.status()
print(a)