from mkidgen3.server.feedline_objects import *

if1 = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=None)
if1a = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=3.0)
if2 = IFConfig(lo=3.0, adc_attn=(3.0, None), dac_attn=3.0)
if3 = IFConfig(lo=6.0, adc_attn=(3.0, None), dac_attn=None)
cc1 = ChannelConfig()
ddc1 = DDCConfig()
fc1 = FilterConfig()
tc1 = TriggerConfig()
pp1 = PhotonPipeConfig(cc1, ddc1, fc1, tc1)
adc1 = ADCConfig()
dac1 = DACConfig(n_uniform_tones=512)
x1 = FeedlineConfig(if1, dac1, pp1, adc_config=adc1)
x1a = FeedlineConfig(if1a, dac1, pp1, adc_config=adc1)

m = FeedlineConfigManager()

assert if1 < if1a and if1.compatible_with(if1a)
assert if1 < if2 and not if1.compatible_with(if2) and not if1a.compatible_with(if2)
assert if3.hashed_form == if1 and if3.hashed_form.compatible_with(if1)
assert if3.hashed_form != if1a and not if3.hashed_form.compatible_with(if1a)


m.learn(x1)
assert m.unlearned_hashes(x1) == set()
update = m.add('1', x1)
assert update == x1
assert m.required() == x1
assert m.pop('1') == True

assert m.unlearned_hashes(x1a) == set()
assert m.unlearned_hashes(x1a.hashed_form) == set([hash(if1a)])
m.add('1', x1)
change = m.add('1a', x1a)
print(change)
