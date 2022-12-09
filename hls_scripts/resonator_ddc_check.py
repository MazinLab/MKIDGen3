import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from collections import defaultdict
file = '/Users/one/result.txt'

def load_data(file):
    # res:<< ":" << samp << ":" << inc << ":" << phase << ":" << phasev[ k].to_double() << ":"
    # << bin_iq << ":" << dds_val << ":" << centerv << ":" << ddcd << ":" << out.data[k]
    keys = ['id', 'i', 'increment','phase0', 'phase_tb', 'phase_core', 'iq_tb', 'dds_tb', 'center', 'result_tb', 'out']

    def parse_line(l):
        vals = l.split(':')
        ret=[]
        for v in vals:
            if ',' in v:
                r, i = v.replace('(', '').replace(')', '').split(',')
                v=float(r.strip()) + float(i.strip()) * 1j
            else:
                v=float(v)
            ret.append(v)
        return ret

    lines = []
    with open(file) as f:
        for l in f.readlines():
            if l.startswith('#'):
                continue
            lines.append(parse_line(l))

    dat = np.array(lines)

    nres = len(set(dat[:, 0].astype(int)))
    nsamp = len(set(dat[:, 1].astype(int)))

    keys=['increment', 'phase0', 'phase_tb', 'phase_core', 'iq_tb', 'dds_tb', 'center', 'result_tb', 'out']
    complex_keys = ['iq_tb', 'dds_tb', 'center', 'result_tb', 'out']

    ret = {}
    for i,k in enumerate(keys):
        x = dat[:, 2+i].reshape((nsamp, nres))
        x = x.astype(float) if k not in complex_keys else x
        ret[k]=x

    return ret


def piq(d, k):
    x=np.asarray(d[k])
    plt.plot(x.real,label=k)
    plt.plot(x.imag)


f='/Users/one/result16_8_17.txt'
data = load_data(f)

pyres = data['dds_tb']*data['iq_tb']-data['center']
plt.close('all')
f,axes = plt.subplots(4, 4, figsize=(13, 6))
for ax, id in zip(axes.T, (0, 12, 33, 54)):
    plt.sca(ax[0])
    tone = data['increment'][:, id][0]
    p0 = data['phase0'][:, id][0]

    t=np.linspace(0, data['increment'][:, id].size, num=data['increment'][:, id].size*10)
    pyphase = -t*tone-p0
    pydds = np.cos(np.pi*pyphase)+1j*np.sin(np.pi*pyphase)
    pyres = data['iq_tb'][:, id]*pydds[::10]-data['center'][:, id]

    plt.plot(data['iq_tb'][:, id].real, label='IQ')
    plt.plot(data['iq_tb'][:, id].imag)
    plt.title(f'In, {tone*1e3:.3f} kHz')
    plt.ylim(-.2,.2)

    plt.sca(ax[1])
    plt.plot(data['result_tb'][:, id].real, 'C0--', label='TB')
    plt.plot(data['result_tb'][:, id].imag, 'C1--')
    plt.plot(data['out'][:, id].real, 'C0', label='HLS')
    plt.plot(data['out'][:, id].imag, 'C1')
    plt.plot(pyres.real, 'C0.', label='Py')
    plt.plot(pyres.imag, 'C1.')
    plt.title('Out')
    plt.ylim(-1.1,1.1)
    plt.legend()

    plt.sca(ax[2])
    plt.plot(data['dds_tb'][:, id].real, 'C0.', label='TB')
    plt.plot(data['dds_tb'][:, id].imag, 'C1.')
    plt.plot(t, pydds.real, 'C0', label='Py')
    plt.plot(t, pydds.imag, 'C1')
    plt.title('DDS')
    plt.legend()
    plt.ylim(-1,1)

    plt.sca(ax[3])
    pyphase = (pyphase*np.pi + np.pi) % (2 * np.pi) - np.pi
    plt.plot(data['phase_tb'][:, id]*np.pi, '.', label='TB')
    plt.plot(data['phase_core'][:, id] * np.pi,'x', label='Core')
    plt.plot(t, pyphase, label='Py')
    plt.title('Phase')
    plt.legend()
    # plt.ylim(-1,1)

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
