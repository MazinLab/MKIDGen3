from logging import getLogger
import numpy as np
import os

ISGOOD = 0b1
ISREVIEWED = 0b10
ISBAD = 0

MAX_ML_SCORE = 1
MAX_ATTEN = 100

LOCUT = 1e9

A_RANGE_CUTOFF = 6e9


def parse_lo(lofreq, frequencies=None, sample_rate=2.048e9):
    """  Sets the attribute LOFreq (in Hz) """
    lo = round(lofreq / (2.0 ** -16) / 1e6) * (2.0 ** -16) * 1e6

    try:
        delta = np.abs(frequencies - lo)
    except AttributeError:
        getLogger(__name__).warning('No frequency list yet loaded. Unable to check if LO is reasonable.')
        return lo

    tofar = delta > sample_rate / 2
    if tofar.all():
        getLogger(__name__).warning('All frequencies more than half a sample rate from '
                                    'the LO. LO: {} Delta min: {} Halfsamp: {} )'.format(lo, delta.min(),
                                                                                         sample_rate / 2))
        raise ValueError('LO out of bounds')
    elif tofar.any():
        getLogger(__name__).warning('Frequencies more than half a sample rate from the LO exist')
    return lo


class SweepFile(object):
    def __init__(self, file):
        self.file = file
        self.feedline = None
        self.resIDs = None
        self.wsfreq = None
        self.flag = None
        self.wsatten = None
        self.mlatten = None
        self.mlfreq = None
        self.ml_isgood_score = None
        self.ml_isbad_score = None
        self.phases = None
        self.iqRatios = None
        self.freq = None
        self.atten = None
        self._load()
        self._vet()

    @property
    def goodmlfreq(self):
        return self.mlfreq[self.flag & ISGOOD]

    def sort(self):
        s = np.argsort(self.resIDs)
        self.resIDs = self.resIDs[s]
        self.wsfreq = self.wsfreq[s]
        self.flag = self.flag[s]
        self.mlfreq = self.mlfreq[s]
        self.mlatten = self.mlatten[s]
        self.atten = self.atten[s]
        self.ml_isgood_score = self.ml_isgood_score[s]
        self.ml_isbad_score = self.ml_isbad_score[s]
        self.freq = self.freq[s]

    def toarray(self):
        return np.array([self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, self.freq,
                         self.atten, self.ml_isgood_score, self.ml_isbad_score, self.phases, self.iqRatios])

    def lomask(self, lo):
        return ((self.flag & ISGOOD) & (~np.isnan(self.freq)) & (np.abs(self.freq - lo) < LOCUT) & (
                    self.atten > 0)).astype(bool)

    def vet(self):
        if (np.abs(self.atten[~np.isnan(self.atten)]) > MAX_ATTEN).any():
            getLogger(__name__).warning('odd attens')
        if (np.abs(self.ml_isgood_score[~np.isnan(self.ml_isgood_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml good score')
        if (np.abs(self.ml_isbad_score[~np.isnan(self.ml_isbad_score)]) > MAX_ML_SCORE).any():
            getLogger(__name__).warning('bad ml bad scores')

        assert self.resIDs.size == np.unique(self.resIDs).size, "Resonator IDs must be unique."

        assert (self.resIDs.size == self.wsfreq.size == self.flag.size ==
                self.atten.size == self.mlfreq.size == self.ml_isgood_score.size ==
                self.ml_isbad_score.size)

    def genheader(self, useSBSup=False):
        if useSBSup:
            header = ('feedline={}\n'
                      'wsatten={}\n'
                      'rID\trFlag\twsFreq\tmlFreq\tmlatten\tfreq\tatten\tmlGood\tmlBad\tphases\tiqratios')
        else:
            header = ('feedline={}\n'
                      'wsatten={}\n'
                      'rID\trFlag\twsFreq\tmlFreq\tmlatten\tfreq\tatten\tmlGood\tmlBad')
        return header.format(self.feedline, self.wsatten)

    def save(self, file='', saveSBSupData=False):
        sf = file.format(feedline=self.feedline) if file else self.file.format(feedline=self.feedline)
        self.vet()
        if saveSBSupData:
            np.savetxt(sf, self.toarray().T, fmt="%8d %1u %16.7f %16.7f %5.1f %16.7f %5.1f %6.4f %6.4f %6.4f %6.4f",
                       header=self.genheader(True))
        else:
            np.savetxt(sf, self.toarray().T[:, :-2], fmt="%8d %1u %16.7f %16.7f %5.1f %16.7f %5.1f %6.4f %6.4f",
                       header=self.genheader(False))

    def _vet(self):

        assert (self.resIDs.size == self.wsfreq.size == self.flag.size == self.atten.size == self.freq.size ==
                self.mlatten.size == self.mlfreq.size == self.ml_isgood_score.size == self.ml_isbad_score.size)

        for x in (self.freq, self.mlfreq, self.wsfreq):
            use = ~np.isnan(x)
            if x[use].size != np.unique(x[use]).size:
                getLogger(__name__).warning("Found non-unique frequencies")

        self.flag = self.flag.astype(int)
        self.resIDs = self.resIDs.astype(int)
        self.feedline = int(self.resIDs[0] / 10000)

    def _load(self):
        d = np.loadtxt(self.file.format(feedline=self.feedline), unpack=True)
        if d.ndim == 1:  # allows files with single res
            d = np.expand_dims(d, axis=1)
        try:
            if d.shape[0] == 11:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                    self.freq, self.atten, self.ml_isgood_score, self.ml_isbad_score, self.phases, self.iqRatios = d
            if d.shape[0] == 9:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                    self.freq, self.atten, self.ml_isgood_score, self.ml_isbad_score = d
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
            elif d.shape[0] == 7:
                self.resIDs, self.flag, self.wsfreq, self.mlfreq, self.mlatten, \
                    self.ml_isgood_score, self.ml_isbad_score = d
                self.freq = self.mlfreq.copy()
                self.atten = self.mlatten.copy()
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
            elif d.shape[0] == 5:
                self.resIDs, self.freq, self.atten, self.phases, self.iqRatios = d
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
            else:
                self.resIDs, self.freq, self.atten = d
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
        except IndexError:
            raise ValueError('Unknown number of columns')

        self.freq[np.isnan(self.freq)] = self.mlfreq[np.isnan(self.freq)]
        self.freq[np.isnan(self.freq)] = self.wsfreq[np.isnan(self.freq)]
        self.atten[np.isnan(self.atten)] = self.mlatten[np.isnan(self.atten)]

        self.flag = self.flag.astype(int)
        self.mlfreq[self.flag & ISBAD] = self.wsfreq[self.flag & ISBAD]
        self.ml_isgood_score[self.flag & ISBAD] = 0
        self.ml_isbad_score[self.flag & ISBAD] = 1
        self._vet()


class PhaseFIRCoeffFile:

    def __init__(self, file):
        self.file=file
        self.coeffs=None
        self._load()

    def _load(self):
        # grab FIR coeff from file
        if os.path.splitext(self.file)[1] == ".npz":
            q
            # res_ids = npz['res_ids']
            self.coeffs = npz['filters']
            # out = np.zeros((len(self.resIDs), filters.shape[1]))
            # for index, resID in enumerate(self.resIDs):
            #     if resID not in resIDs:
            #         raise ValueError("Filter coefficients missing resID {}".format(resID))
            #     location = (resIDs == resID)
            #     if location.sum() > 1:
            #         raise ValueError("Filter coefficients contain more than one reference to resID {}".format(resID))
            #     firCoeffs[index, :] = filters[location, :]
        else:
            self.coeffs = np.loadtxt(self.file)