import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def calculate_psd(data, fs=1e6, fres=1e3, fit_level = True, plot=True):
    plt.figure()
    f, psd = welch(data, fs=fs, nperseg=fs/fres)

    if fit_level:
        pass

    if plot:
        fig, ax = plt.subplots()
        ax.semilogx(f, 10 * np.log10(psd))
        ax.set_xlabel(f'Frequency [Hz] ({fres * 1e-3:g} kHz resolution)')
        ax.set_ylabel('dBc/Hz')
        ax.grid()
        ax.title('Power Spectral Density')

    return f, psd
