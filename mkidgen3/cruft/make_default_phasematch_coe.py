def make_default_phase_filter(file):
    x=[[0]*29+[-1]]*512
    x=[z for y in x for z in y]
    x=' '.join(map(str,x))
    with open(file, 'w') as f:
      f.write('radix=10;\ncoefdata={};'.format(x))
