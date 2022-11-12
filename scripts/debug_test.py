import time
import mkidgen3 as g3
bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'
ol = g3.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=False, download=True)
print(ol.dac_table.register_map)
import matplotlib.pyplot as plt
import numpy as np
a = np.linspace(0,5,6)
plt.plot(a)
plt.show()

#while True:
#    time.sleep(1)
#    print('hi')