import copy

from mkidgen3.server.feedline_config import *
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.opfb import opfb_bin_number

if1 = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=None)
if1a = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=3.0)
if2 = IFConfig(lo=3.0, adc_attn=(3.0, None), dac_attn=3.0)
if3 = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=None)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top.bit', ignore_version=True)
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)
if_board = IFConfig(lo=6000, adc_attn=10, dac_attn=50)
waveform_vals = WaveformFactory(frequencies=[100e6])
waveform = WaveformConfig(waveform=waveform_vals)

waveform_vals2 = WaveformFactory(frequencies=[150e6])
waveform2 = WaveformConfig(waveform=waveform_vals2)
freqs = waveform.waveform.freqs

# Bin2Res Config
bins = np.zeros(2048, dtype=int)
bins[:freqs.size] = opfb_bin_number(freqs, ssr_raw_order=True)
chan = ChannelConfig(bins=bins)

# DDC Config
ddc_tones = np.zeros(2048)
ddc_tones[:freqs.size]=freqs
ddc = DDCConfig(tones=ddc_tones)

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)
fc2 = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform2, chan=chan, ddc=ddc)


assert fc2.compatible_with(fc2.hashed_form)

# test hashed comparison
fc3 = copy.deepcopy(fc2)
fc3.if_board = IFConfig(lo=3000, dac_attn=50, adc_attn=50)

m = FeedlineConfigManager()

assert if1 < if1a and if1.compatible_with(if1a)
assert if1 < if2 and not if1.compatible_with(if2) and not if1a.compatible_with(if2)
assert if3.hashed_form == if1 and if3.hashed_form.compatible_with(if1)
assert if3.hashed_form != if1a and not if3.hashed_form.compatible_with(if1a)

assert fc3.compatible_with(fc3.hashed_form)


m.learn(fc)
assert m.unlearned_hashes(fc) == set()


update = m.add('1', fc)
assert update == fc
assert m.required() == fc
assert m.pop('1') == True

assert m.unlearned_hashes(fc2) == set()
#assert m.unlearned_hashes(x1a.hashed_form) == set([hash(if1a)])
#m.add('1', x1)
change = m.add('1a', fc2)
print(change)
