from pynq import DefaultIP
from mkidgen3.system_parameters import DAC_LUT_SIZE, DAC_SAMPLE_RATE, PL_DDS_M
import logging

class SweepDDS(DefaultIP):
    """
    A class to interface with the re-programmable DDS implmented in the programmable logic (PL DDS).
    The PL DDS is able to generate frequencies between the DAC table resolution (7.8125 kHz) and the 
    input clock (128 MHz) with <DAC table resoluton> precision though in practice it performs best 
    at lower frequencies (<3 MHz).


    Attributes:
        see DefaultIP

    Methods:
        configure()
            Program PL DDS.

    Example:
       pl_dds = ol.freq_sweep.PL_DDS.dds_pinc_1
       pl_dds.configure(7812.5)
    """

    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['slaclab:hls:dds_pinc:1.0']

    def configure(self, freq):
        """
        Program DDS to output a given frequency.

        Args:
            freq: Desired PL DDS output frequency in Hz. Allowed Values: (7812.5 to 128e6).
        """

        fres = DAC_SAMPLE_RATE/DAC_LUT_SIZE
        max_freq = fres*PL_DDS_M
        
        if not ((fres <= freq <= max_freq) or freq == 0): # check freq
            raise ValueError(f"DDS frequency must be either 0 or between {fres} Hz and {max_freq} Hz.")
        
        pinc = int(freq/(fres)) & 0x3FFF # quantize & ensure 14-bit expected by DDS IP
        
        # log messages
        logging.getLogger(__name__).info(f"Programming PL DDS with {pinc*fres} Hz")
        logging.getLogger(__name__).debug(f"Setting phase increment to {pinc}.")
        
        self.write(0x10, pinc)
        return
    
    def off(self):
        """
        Turn DDS off by making it output a constant 1 (identity) that doesn't up or downconvert anything.
        """
        self.configure(0)
