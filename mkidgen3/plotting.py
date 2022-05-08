import matplotlib.pyplot as plt
import numpy as np

def plt_adc_timeseries(data, start, stop, fs=4.096e9, ax=None, **kwargs):
    """
    data: np array of captured adc data
    start: first sample # in plot
    stop: last sample # in plot
    fs = ADC sample rate [Hz]

    Returns:
        Plot of ADC timeseries.
    """
    
    if ax is None:
        ax = plt.gca()

    n = data.shape[0] # total samples
    tvec = np.linspace(0, n/fs, n)*1e9 # time vector [nano seconds]
    sample_sl =  slice(start, stop) # plt slice

    plt.figure(figsize=(10,5))
    plt.plot(tvec[sample_sl], data.real[sample_sl])
    plt.plot(tvec[sample_sl], data.real[sample_sl],"o")
    plt.grid(True)
    plt.xlabel("time(ns)", position=(0.5,1))
    plt.ylabel("signal(V)", position=(0,0.5))
    ax.set_xlim(tvec[start], tvec[stop])
    plt.title('Time Series')
    
    
