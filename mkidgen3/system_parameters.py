import numpy as np
from typing import Iterable

DAC_MAX_OUTPUT_DBM = 1  # [dBm] see Xilinx DS926
DAC_MAX_INT = 8191  # see PG269 p.219
ADC_RESOLUTION = 14 # bits see PG269 p.219
ADC_DAC_INTERFACE_WORD_LENGTH = 16  # bits see PG269 p.219
# This computes the max int the ADC can register, For the RFSOC4x2 this is 0x7FFC, which is given in the table titled
# "RF-ADC and RF-DAC Word Interface" in PG269 under section "Digital Data Format"
ADC_MAX_INT = 2**(ADC_RESOLUTION-1)-1 << (ADC_DAC_INTERFACE_WORD_LENGTH-ADC_RESOLUTION) # see above
ADC_INPUT_WARN = 0.7*ADC_MAX_INT
ADC_MAX_INPUT_DBM = 1  # [dBm] see Xilinx DS926

# per DS926 Table RF-ADC Electrical Characteristics for ZU4xDR Devices
# (https://docs.xilinx.com/r/en-US/ds926-zynq-ultrascale-plus-rfsoc/RF-ADC-Electrical-Characteristics)
ADC_MAX_V = 0.7
ADC_MAX_V_DIFF = 1.4
ADC_MAX_V_CONSERVATIVE = 0.4
ADC_MAX_V_DIFF_CONSERVATIVR = 0.8

DAC_RESOLUTION = 14  # bits see PG269 p.219
DAC_LUT_SIZE = 2 ** 19  # values
ADC_SAMPLE_RATE = 4.096e9  # Hz
DAC_SAMPLE_RATE = 4.096e9  # Hz
DAC_FREQ_RES = DAC_SAMPLE_RATE / DAC_LUT_SIZE
DAC_FREQ_MAX = (DAC_SAMPLE_RATE / 2) - DAC_FREQ_RES
DAC_FREQ_MIN = -DAC_SAMPLE_RATE / 2
N_OPFB_CHANNELS = 4096  # Number of OPFB channels
N_CHANNELS = 2048  # Number of DDC (resonator) channels
N_PHASE_GROUPS = 128  # See HLS and the phase drivers, the AXI streams are 16 channels wide
N_IQ_GROUPS = 256  # See HLS and the iq drivers, the AXI streams are 8 channels wide
N_POSTAGE_CHANNELS = 8  # see HLS and the postage filter driver
SYSTEM_BANDWIDTH = 4.096e9  # Hz Full readout bandwidth
OS = 2  # OPFB Overlap factor
OPFB_CHANNEL_SAMPLE_RATE = OS * ADC_SAMPLE_RATE / N_OPFB_CHANNELS


# PHASE_IQ_INPUT_FRACTIONAL_BITS = 14  #commented as no clear meaning, no doc, and no usage in codebase

# cordic output scaled radians [-1,1] is 18 bits, signed, 3 integer, 15 fractional, truncated to the low 16 bits
PHASE_FRACTIONAL_BITS = 15

PL_TOTAL_BYTES = 4 * 1024 ** 3
SYSTEM_OVERHEAD_BYTES = 2 * 768 * 1024 ** 2
COMPRESSION_OVERHEAD_BYTES = 128 * 1024 ** 2  #this is complete speculation
LOWPASSED_IQ_SAMPLE_RATE = 1e6
MAXIMUM_DESIGN_COUNTRATE_PER_S = 5000

PHOTON_POSTAGE_WINDOW_LENGTH = 127  # Must be 1 less than the HLS value, this is the number of IQ values captured for a photon event


from mkidgen3.equipment_drivers.ifboard import MAX_IN_ATTEN, IF_ATTN_STEP, MAX_OUT_ATTEN


def channel_to_iqgroup(channels: Iterable | int) -> set:
    """
    Convert channels to IQ groups which contains those channels.
    Args:
        channels: resonator channels (0-2047)

    Returns: set of IQ groups

    """
    try:
        return set([g // 8 for g in channels])
    except TypeError:
        return {channels//8}


def iqgroup_to_channel(iq_group: Iterable | int) -> set:
    """
    Convert IQ groups to channels.
    Args:
        iq_group: (0-128)

    Returns: set of corresponding channels (0-2047)

    """
    chan_per_group = N_CHANNELS // N_IQ_GROUPS
    if isinstance(iq_group, int):
        return set(np.arange(chan_per_group)+iq_group*chan_per_group)
    else:
        iq_group = np.fromiter(iq_group, int, len(iq_group))
        return set((np.arange(chan_per_group)+(iq_group*chan_per_group)[:, np.newaxis]).flatten())


def channel_to_phasegroup(channels: Iterable | int) -> set:
    """
    Convert phase channels to IQ groups which contains those channels.
    Args:
        channels: resonator channels (0-2047)

    Returns: set of IQ groups (0-255)

    """
    try:
        return set([g // 16 for g in channels])
    except TypeError:
        return {channels//16}


def phasegroup_to_channel(phase_group: Iterable | int) -> set:
    """
    Convert phase groups to channels.
    Args:
        phase_group: (0-127)

    Returns: set of corresponding channels (0-2047)

    """
    chan_per_group = N_CHANNELS // N_PHASE_GROUPS
    if isinstance(phase_group, int):
        return set(np.arange(chan_per_group)+phase_group*chan_per_group)
    else:
        phase_group = np.fromiter(phase_group, int, len(phase_group))
        return set((np.arange(chan_per_group)+(phase_group*chan_per_group)[:, np.newaxis]).flatten())
