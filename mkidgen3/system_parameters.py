import numpy as np

DAC_MAX_OUTPUT_DBM = 1  # [dBm] see Xilinx DS926
DAC_MAX_INT = 8191  # see PG269 p.219
ADC_DAC_INTERFACE_WORD_LENGTH = 16  # bits see PG269 p.219
ADC_MAX_INPUT_DBM = 1  # [dBm] see Xilinx DS926
ADC_MAX_V = 1/(2*np.sqrt(5))  # [V]. 1 dbM is 1 mW, using P = V^2/R where P is 1 mW and R is 100 ohms,
# V is 1/sqrt(10) volts at the die termination which means the max-scale V is  1/2*sqrt(5) at the SMA input.
DAC_RESOLUTION = 14  # bits see PG269 p.219
DAC_LUT_SIZE = 2 ** 19  # values
DAC_SAMPLE_RATE = 4.096e9  # GSPS
N_OPFB_CHANNELS = 4096  # Number of OPFB channels
N_CHANNELS = 2048  # Number of DDC (resonator) channels
SYSTEM_BANDWIDTH = 4.096e9  # Hz Full readout bandwidth
OS = 2  # OPFB Overlap factor

PL_TOTAL_BYTES = 4 * 1024 ** 3
SYSTEM_OVERHEAD_BYTES = 768 * 1024 ** 2

# per DS926 Table RF-ADC Electrical Characteristics for ZU2xDR Devices
# (https://docs.xilinx.com/r/en-US/ds926-zynq-ultrascale-plus-rfsoc/RF-ADC-Electrical-Characteristics)
ADC_MAX_VOLTAGE = 0.22360679774997896  # 0.5/np.sqrt(5) per p.


from mkidgen3.drivers.ifboard import MAX_IN_ATTEN, IF_ATTN_STEP, MAX_OUT_ATTEN
