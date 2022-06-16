import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from collections import defaultdict
file = '/Users/one/result.txt'

def load_data(file):
    # tone_id = (j * 8 + k) % 128 - 64;
    # inc = tone_id * 4.096e9 / 262144.0 / 1e6;
    # -17, -0.265625, (0.364471435546875, 0.34228515625), (0.7289686274214116, -0.6845471059286886), (
    # 0.4999985552182034, 1.727414099803037e-005), (-0.78314208984375, 0.53240966796875)

    def parse_line(l):
        tone, _, other = l.partition(',')
        inc, _, other = other.partition(',')
        iq_in, dds_v, ddcd, iq_out = other.split('),(')
        vals=[]
        for v in (iq_in, dds_v, ddcd, iq_out):
            r,i=v.replace('(','').replace(')','').split(',')
            vals.append(float(r.strip())+float(i.strip())*1j)
        return int(tone.strip()),float(inc.strip()), vals
    d = defaultdict(lambda:defaultdict(list))
    with open(file) as f:
        for l in f.readlines():
            if l.startswith('#'):
                continue
            tone,inc, vals=parse_line(l)
            d[tone]['increment']=inc
            d[tone]['in'].append(vals[0])
            d[tone]['dds_gold'].append(vals[1])
            d[tone]['out_gold'].append(vals[2])
            d[tone]['out'].append(vals[3])
    for v in d.values():
        for k in ('in', 'out_gold', 'out', 'dds_gold'):
            v[k]=np.asarray(v[k])
    return d


def piq(d, k):
    x=np.asarray(d[k])
    plt.plot(x.real,label=k)
    plt.plot(x.imag)

f='/Volumes/BOOTCAMP/Users/one/xilinx_projects/hls/resonator-dds/resonator-dds-2021_1/bwfix5/csim/build/result16_8_17.txt'
f='/Volumes/BOOTCAMP/Users/one/xilinx_projects/hls/resonator-dds/result16_8_17.txt'
data = load_data(f)
plt.close('all')
f,axes = plt.subplots(3,4, figsize=(13,6))
for ax,id in zip(axes.T,(0,33,128, 250)):
    plt.sca(ax[0])
    tdata=data[id]
    tone=tdata['increment']*1e3
    plt.plot(tdata['in'].real,label='IQ')
    plt.plot(tdata['in'].imag)
    plt.title(f'In, {tone:.3f} kHz')

    plt.sca(ax[1])
    plt.plot(tdata['out_gold'].real,'C0--', label='Gold')
    plt.plot(tdata['out_gold'].imag,'C1--')
    plt.plot(tdata['out'].real,'C0', label='HLS')
    plt.plot(tdata['out'].imag,'C1')
    x=tdata['dds_gold']*tdata['in']
    plt.plot(x.real, '.', label='Goldpy')
    plt.plot(x.imag,'.')
    plt.title('Out')
    plt.legend()

    plt.sca(ax[2])
    plt.plot(tdata['dds_gold'].real,label='DDS')
    plt.plot(tdata['dds_gold'].imag)

from mkidgen3.fixedpoint import FP16_15, FP18_17


def line2iq(line):
    try:
        i, q = line.strip().replace('(','').replace(')','').split(',')[-2:]
        return float(i) + 1j * float(q)
    except ValueError:
        print(line)
        raise


def plot_psd(file, NRES=128):
    with open(file) as f:
        lines = f.readlines()
    outiq = np.array([line2iq(l) for l in lines if l and not l.startswith('#')])
    outiq = outiq.reshape((outiq.size // NRES, NRES))

    phase = np.arctan2(outiq.imag, outiq.real)
    mag = np.abs(outiq)

    phase_psd = sps.periodogram(phase, fs=2e6, axis=0)
    mag_psd = sps.periodogram(mag, fs=2e6, axis=0)
    plt.figure(figsize=(11,7))
    plt.subplot(221)
    plt.imshow(10*np.log10(phase_psd[1]), interpolation=None,
               extent=(-1, 1, phase_psd[0].min()/1e6, phase_psd[0].max()/1e6),origin='lower')
    plt.ylabel('PSD Freq. (MHz)')
    plt.xlabel('Tone Offset (MHz)')
    plt.colorbar()
    plt.title('Phase')

    plt.subplot(222)
    plt.imshow(10*np.log10(mag_psd[1]), interpolation=None,
               extent=(-1, 1, mag_psd[0].min()/1e6, mag_psd[0].max()/1e6), origin='lower')
    plt.ylabel('PSD Freq. (MHz)')
    plt.xlabel('Tone Offset (MHz)')
    plt.colorbar()
    plt.title('Mag')

    plt.subplot(223)
    plt.plot(phase_psd[0] / 1e6, 10 * np.log10(phase_psd[1].mean(1)))
    plt.xlabel('PSD Freq. (MHz)')
    plt.ylabel('Mean Phase PSD (dBc)')

    plt.subplot(224)
    plt.plot(mag_psd[0] / 1e6, 10 * np.log10(mag_psd[1].mean(1)))
    plt.xlabel('PSD Freq. (MHz)')
    plt.ylabel('Mean Mag PSD (dBc)')

    plt.subplots_adjust(top=0.979, bottom=0.083, left=0.075, right=0.985, hspace=0.071, wspace=0.179)

    print(f'Phase error (mean/max/std): {np.abs(phase).mean():.2g}/{np.abs(phase).max():.2g}/{np.abs(phase).std():.2g}')
    mage=mag-.5
    print(f'Mag error (mean/max/std): {np.abs(mage).mean():.2g}/{np.abs(mage).max():.2g}/{np.abs(mage).std():.2g}')

plot_psd('/Users/one/result.txt')
plt.suptitle('21 P0, 21 Tone, 21 accumulator bits')
plot_psd('/Users/one/result16_8_21.txt')
plt.suptitle('16 P0, 8 Tone, 21 accumulator bits')
plot_psd('/Users/one/result16_8_17.txt')
plt.suptitle('16 P0, 8 Tone, 17 accumulator bits')
