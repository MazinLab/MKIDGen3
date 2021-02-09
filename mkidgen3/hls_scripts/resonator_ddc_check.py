import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

file = '/Users/one/result.txt'



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