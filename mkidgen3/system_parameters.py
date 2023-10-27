DAC_MAX_OUTPUT_DBM = 1  # [dBm] see Xilinx DS926
DAC_MAX_INT = 8191  # see Xilinx docs
ADC_DAC_INTERFACE_WORD_LENGTH = 16  # bits see Xilinx docs
DAC_RESOLUTION = 14  # bits
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
