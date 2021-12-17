
"""
Generate a file containing tone to bin center offsets for 2048 resonators and a file containing IQ values
for the resonators over some number of cycles

IQ values are complex numbers on the unit circle
"""
import numpy as np
from mkidgen3.daccomb import SweepFile


class Testdata:
    def __init__(self, iq=None, offset_hz=None, bincount=None, freqs=None):
        self.iq=iq
        self.bincount=bincount
        self.offset_hz=offset_hz
        self.freqs=freqs


def test_data(ncycles=10, MIN_FREQ_HZ=4096e6,  MAX_FREQ_HZ =8191e6, NUM_RES = 2048, MAX_PER_BIN = 4):

    mec_freqfilea='/Users/one/Box Sync/ucsb/mec/psfreq/psfreqs_FL9a_clip_new_atten_plus_freq_shift.txt'
    mec_freqfileb='/Users/one/Box Sync/ucsb/mec/psfreq/psfreqs_FL9b_clip_new_atten_plus_freq_shift.txt'
    freqfilea = SweepFile(mec_freqfilea)
    freqfileb = SweepFile(mec_freqfileb)

    freqs=np.concatenate((freqfilea.freq,freqfileb.freq))
    freqs-=freqs.min()
    freqs+=MIN_FREQ_HZ

    bincount, binedges = np.histogram(np.round(freqs/1e6), bins=np.arange(4096,8192+1))

    print('Most resonators in a 1MHz bin in loaded files: {}'.format(max(bincount)))

    if freqs.size<NUM_RES:
        print("Inserting {} resonators at the end".format(NUM_RES-freqs.size))
        addable = (MAX_PER_BIN-bincount[::-1]).clip(0, MAX_PER_BIN)
        total_added = np.cumsum(addable)
        last_add_ndx = np.where(total_added >= NUM_RES-freqs.size)[0][0]
        addable[last_add_ndx] += NUM_RES-(total_added[last_add_ndx]+freqs.size)  # in case we would add a few too many
        addable[last_add_ndx+1:]=0
        bincount[::-1] += addable
        #TODO properly add frequencies
        freqs = np.concatenate((freqs, [MAX_FREQ_HZ]*(NUM_RES-freqs.size)))


    offset_hz = (freqs-1e6*np.round(freqs/1e6))  # on a 1MHz grid

    freqs_pm=freqs-MIN_FREQ_HZ-2048e6
    bin_ndx = np.round(freqs_pm / 1e6).astype(int) + 2048
    bin_f_center = np.arange(-2048, 2048) * 1e6
    offset_hz_2 = freqs_pm-bin_f_center[bin_ndx]

    np.random.seed(0)
    iq=np.random.uniform(low=0, high=2*np.pi, size=ncycles*NUM_RES)
    iq=np.array((np.sin(iq), np.cos(iq))).T

    return Testdata(iq=iq, freqs=freqs, offset_hz=offset_hz, bincount=bincount)

if __name__=='__main__':
    td=test_data()

    with open('/Users/one/Desktop/toneoffsets.dat','w') as f:
        f.writelines(['{}\n'.format(oset) for oset in td.offset_hz])

    with open('/Users/one/Desktop/resiqs.dat','w') as f:
        f.writelines(['{} {}\n'.format(i,q) for i,q in td.iq])

    with open('/Users/one/Desktop/res_in_bin.dat','w') as f:
        f.writelines(['{}\n'.format(n) for n in td.bincount])


