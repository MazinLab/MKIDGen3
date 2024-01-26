import zmq
from mkidgen3.server.feedline_config import IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest,StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.funcs import compute_lo_steps, compute_power_sweep_attenuations
from mkidgen3.power_sweep_helpers import *
from mkidgen3.opfb import opfb_bin_number
import numpy as np
from mkidgen3.util import setup_logging
setup_logging('feedlineclient')
# ctx = zmq.Context.instance()
# ctx.linger = 0



frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)

# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz

# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)

# IF Config
if_board = IFConfig(lo=6000, adc_attn=50, dac_attn=50)

# DAC Config
waveform_vals = WaveformFactory(n_uniform_tones=512)

waveform = WaveformConfig(waveform=waveform_vals)
output_waveform = waveform.waveform.output_waveform # trigger waveform computation

# Bin2Res Config
chan = waveform.default_channel_config

# DDC Config
ddc = waveform.default_ddc_config

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)
#fc2= FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
#                    if_board=if_board, waveform=waveform2, chan=chan, ddc=ddc,
#                    filter=FilterConfig(coefficients='unity20'),
#                    trig=TriggerConfig(holdoffs=[20]*2048, thresholds=[0]*2048))
#gsm = StatusListener(b'', frsb.status_url)
cr = CaptureRequest(2**14, 'ddciq', fc, frsa)
#cr = CaptureRequest(1024**3//4//2048, 'iq', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/iq1024MiB.npz')
#cr = CaptureRequest(1024**3//2//2048, 'phase', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/phase1024MiB.npz')
#cr = CaptureRequest(3024**3//2//2048, 'phase', fc, frsa)
#cr = CaptureRequest(2**19, 'adc', fc, frsa)


paramp = Amplifier("paramp", gain=12, saturation=-40)
hemt = Amplifier("hemt", gain=40, saturation=-40)
if1 = Amplifier("if1", gain=14.5, saturation=25)
if2 = Amplifier("if2", gain=14.5, saturation=25)
if3 = Amplifier("if3", gain=14.5, saturation=25)
if4 = Amplifier("if1", gain=14.5, saturation=25)
if5 = Amplifier("if2", gain=15.0, saturation=20)
if6 = Amplifier("if3", gain=15.0, saturation=20)
atten1 = Attenuator("if_fixed_atten", gain=-1)
dac_atten = ProgrammableAttenuator("dac_atten", gain=-30)
adc_atten1 = ProgrammableAttenuator("adc_atten1", gain=-10)
adc_atten2 = ProgrammableAttenuator("adc_atten2", gain=0)
atten30 = Attenuator("4K atten", gain=-30)
device_atten = Device("mkid", gain=-3)
atten20 = Attenuator("cable atten", gain=-20)
atten10 = Attenuator("atten", gain=-10)

chain = ROChain([dac_atten, atten30, device_atten, paramp, hemt, if1, adc_atten1, if2, adc_atten2, if3, if4, if5, atten1, if6])


def calculate_adc_dac_atten(dac_output_power: float, device_power: float, chain: ROChain, adc_input_power: float)->tuple:
    dac_atten = -device_power + dac_output_power + chain.pre_device_gain()
    chain[chain.dac_atten_idx].gain = -dac_atten
    post_adc_attens = adc_input_power - chain.post_adc_atten2_gain()
    pre_adc_attens = device_power + chain.post_device_gain()
    total_adc_attenuation = pre_adc_attens - post_adc_attens

    amp_between_adc_attens = chain.ro_chain[chain.adc_atten1_idx+1]
    assert isinstance(amp_between_adc_attens, Amplifier), "adc attenuator 1 needs to be followed by amplifier"
    if pre_adc_attens > amp_between_adc_attens.max_input:
        chain.ro_chain[chain.adc_atten1_idx].gain = pre_adc_attens-amp_between_adc_attens.max_input
        chain.ro_chain[chain.adc_atten2_idx].gain = -total_adc_attenuation + chain.ro_chain[adc_atten1].gain
    else:
        chain.ro_chain[chain.adc_atten1_idx].gain = 0
        chain.ro_chain[chain.adc_atten2_idx].gain = -total_adc_attenuation

    chain.validate(dac_output_power)

    return chain.ro_chain[chain.dac_atten_idx].gain, (chain.ro_chain[chain.adc_atten1_idx].gain, chain.ro_chain[chain.adc_atten2_idx].gain)



lo_sweep_freqs = compute_lo_steps(center=0, resolution=7.14e3, bandwidth=2e6) # TODO what is the right resolution
attenuations = compute_power_sweep_attenuations(0,30,0.25)
if_board_cfgs = [IFConfig(lo=x, adc_attn=50, dac_attn=50) for x in lo_sweep_freqs]


j = CaptureJob(cr)
j.submit(True, True)

print('hi')
print('hi')
