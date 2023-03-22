
from mkidgen3.feedline_objects import *

if1 = IFConfig(lo=6.0, adc_attn=(3.0,None), dac_attn=None)
if1a = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=3.0)
assert if1 == if1a
if2 = IFConfig(lo=3.0, adc_attn=(3.0, None), dac_attn=3.0)
assert if1!=if2 and if1a!=if2

if3 = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=None)

assert if3.hashed_form==if1

assert if3.hashed_form!=if1a  #TODO we might actually like this to be true

cc1 = ChannelConfig()
ddc1 = DDCConfig()
fc1 = FilterConfig()
tc1 = TriggerConfig()
pp1=PhotonPipeConfig(cc1, ddc1, fc1, tc1)

adc1 = ADCconfig()
dac1 = DACConfig(n_uniform_tones=10)
x = FeedlineConfig(if1, dac1, pp1, adc_config=adc1)

m = FeedlineConfigManager()
m.learn(x)
assert m.unlearned_hashes(x)==set()
update = m.add('bob',x)
assert update==x
assert m.effective()==x
assert m.pop('bob')==False  #TODO ugh should this actually be true?
