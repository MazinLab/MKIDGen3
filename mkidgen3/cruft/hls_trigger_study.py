import numpy as np
from glob import glob
import os.path
import matplotlib.pyplot as plt


PULSE_DECAY_TIME=15e-6


def pulse(t, decay=PULSE_DECAY_TIME):
    heavy_e=-np.e**(-t/decay)*np.heaviside(t,1)/2
    return heavy_e/2 + heavy_e*1j


def phase(iq):
    return np.arctan2(iq.imag, iq.real)


def gen_data(cps=2500, pulse_height=.5, first_pulse_us=600, max_t_ms=10,
        dir='./', base_dir='/Users/one/Box Sync/ucsb/gen3/baseline_samples_MEC20200128'):

    ncycles=max_t_ms*1000
    phases=np.zeros(ncycles)

    bfiles = glob(os.path.join(base_dir, '*.npz'))
    bline = np.load(bfiles[0])['arr_0']

    base=bline[:ncycles]
    base=base-base.mean()

    assert bline.size>ncycles

    t=np.arange(phases.size)/1e6

    t0s=np.arange(first_pulse_us/1e6, t.max(), 1/cps)

    for t0 in t0s:
        phases+=pulse_height*(phase(pulse(t-t0)+1))




    plt.plot(phases, label='Pulses')
    plt.plot(base, label='Base')


    phases+=base

    plt.plot(phases, label='Phasedata')
    plt.legend()

    # phases+=.9*(1-phases.max())

    # towrap=phases<-1
    # phases[towrap]=2-phases[towrap]
    # towrap=phases>=1
    # phases[towrap]=-2+phases[towrap]
    # if not np.all((phases>=-1) & (phases<=1)):
    #     print('Phases <-1 found')

    np.savetxt(os.path.join(dir, 'phase_input.txt'), phases)

    return phases, bline[:ncycles]

#
# def plot_res(file):
#     simdat = np.genfromtxt(file)
#
# cps = 2500
# pulse_height = .5
# first_pulse_us = 600
# max_t_ms = 10
# dir = './'
# base_dir = '/Users/one/Box Sync/ucsb/gen3/baseline_samples_MEC20200128'
#
# ncycles=max_t_ms*1000
# phases=np.ones(ncycles)
#
# bfiles = glob(os.path.join(base_dir, '*.npz'))
# bline = np.load(bfiles[0])['arr_0']
#
# assert bline.size>ncycles
#
# t=np.arange(phases.size)/1e6
#
# t0s=np.arange(first_pulse_us/1e6, t.max(), 1/cps)
#
# for t0 in t0s:
#     phases+=pulse_height*phase(pulse(t-t0))


def parse_simlog(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    base = np.array([parse_base(l) for l in filter(lambda l: l.startswith('BL '), lines)])
    trig = np.array([parse_trig(l) for l in filter(lambda l: l.startswith('TR '), lines)])
    pulse = np.array([parse_pulse(l) for l in filter(lambda l: l.startswith('PU '), lines)])
    return base, trig, pulse


def parse_base(line):
    _, phase, baseline, diff, accum,  win, n_in, n_out, grow, shrink = line.split()
    return (float(phase), float(baseline), float(diff), float(accum), float(win),
            int(n_in), int(n_out), int(grow), int(shrink))


def parse_trig(line):
    _, triggered = line.split()
    return bool(int(triggered)),


def parse_pulse(line):
    _, baseline,peak,start,peak_time,stop,have_pulse = line.split()
    return float(baseline), float(peak), int(start), int(peak_time), int(stop), bool(have_pulse)


logfile='/Volumes/[C] Boot Camp.hidden/Users/one/xilinx_projects/hls/photon-trigger/photon-trigger/solution1/csim/report/photon_trigger_csim.log'
base, t, pulse =parse_simlog(logfile)
plt.figure()
plt.plot(base[:,0]-base[:,1], label='Corrected')
plt.plot(base[:,0],'-.', label='Phase', linewidth=.5)
plt.plot(base[:,1], label='BL', linewidth=.5)
plt.plot(base[:,4], label='Win', linewidth=.5)
plt.plot(base[:,1]+base[:,4], 'k--', linewidth=.5)
plt.plot(base[:,1]-base[:,4], 'k--', linewidth=.5)
plt.plot(t+1.1,label='Trigger')
plt.legend()