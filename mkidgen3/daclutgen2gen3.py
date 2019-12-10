import numpy as np
import scipy.special
from logging import getLogger
import logging

# flags
ISGOOD = 0b1
ISREVIEWED = 0b10
ISBAD = 0

MAX_ML_SCORE = 1
MAX_ATTEN = 100

LOCUT = 1e9

A_RANGE_CUTOFF = 6e9

nDacSamplesPerCycle = 8
nLutRowsToUse = 2 ** 15
dacSampleRate = 2.e9
nBitsPerSamplePair = 32
nChannels = 1024


def parse_lo(lofreq, freqList=None):
    """  Sets the attribute LOFreq (in Hz) """
    lo = round(lofreq / (2.0 ** -16) / 1e6) * (2.0 ** -16) * 1e6

    try:
        delta = np.abs(freqList - lo)
    except AttributeError:
        getLogger(__name__).warning('No frequency list yet loaded. Unable to check if LO is reasonable.')
        return lo

    tofar = delta > dacSampleRate / 2
    if tofar.all():
        getLogger(__name__).warning('All frequencies more than half a sample rate from '
                                    'the LO. LO: {} Delta min: {} Halfsamp: {} )'.format(lo, delta.min(),
                                                                                         dacSampleRate / 2))
        raise ValueError('LO out of bounds')
    elif tofar.any():
        getLogger(__name__).warning('Frequencies more than half a sample rate from the LO')
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
                # _, u = np.unique(self.freq, return_index=True)
                # self.resIDs = self.resIDs[u]
                # self.freq = self.freq[u]
                # self.atten = self.atten[u]
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
            else:
                self.resIDs, self.freq, self.atten = d
                # _, u = np.unique(self.freq, return_index=True)
                # self.resIDs = self.resIDs[u]
                # self.freq = self.freq[u]
                # self.atten = self.atten[u]
                self.wsfreq = self.freq.copy()
                self.mlfreq = self.freq.copy()
                self.mlatten = self.atten.copy()
                self.flag = np.full_like(self.resIDs, ISGOOD, dtype=int)
                self.ml_isgood_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.ml_isbad_score = np.full_like(self.resIDs, np.nan, dtype=float)
                self.phases = np.full_like(self.resIDs, 0, dtype=float)
                self.iqRatios = np.full_like(self.resIDs, 1, dtype=float)
        except:
            raise ValueError('Unknown number of columns')

        self.freq[np.isnan(self.freq)] = self.mlfreq[np.isnan(self.freq)]
        self.freq[np.isnan(self.freq)] = self.wsfreq[np.isnan(self.freq)]
        self.atten[np.isnan(self.atten)] = self.mlatten[np.isnan(self.atten)]

        self.flag = self.flag.astype(int)
        self.mlfreq[self.flag & ISBAD] = self.wsfreq[self.flag & ISBAD]
        self.ml_isgood_score[self.flag & ISBAD] = 0
        self.ml_isbad_score[self.flag & ISBAD] = 1
        self._vet()


def generateTones(freqList, nSamples, sampleRate, amplitudeList=None, phaseList=None, iqRatioList=None,
                  iqPhaseOffsList=None):
    """
    Generate a list of complex signals with amplitudes and phases specified and frequencies quantized

    INPUTS:
        freqList - list of resonator frequencies
        nSamples - Number of time samples
        sampleRate - Used to quantize the frequencies
        amplitudeList - list of amplitudes. If None, use 1.
        phaseList - list of phases. If None, use random phase

    OUTPUTS:
        dictionary with keywords
        I - each element is a list of I(t) values for specific freq
        Q - Q(t)
        quantizedFreqList - list of frequencies after digitial quantiziation
        phaseList - list of phases for each frequency
    """
    if amplitudeList is None:
        amplitudeList = np.asarray([1.] * len(freqList))
    if phaseList is None:
        phaseList = np.random.uniform(0., 2. * np.pi, len(freqList))
    if iqRatioList is None:
        iqRatioList = np.ones(len(freqList))
    if iqPhaseOffsList is None:
        iqPhaseOffsList = np.zeros(len(freqList))
    if len(freqList) != len(amplitudeList) or len(freqList) != len(phaseList) or len(freqList) != len(
            iqRatioList) or len(freqList) != len(iqPhaseOffsList):
        raise ValueError("Need exactly one phase, amplitude, and IQ correction value for each resonant frequency!")

    # Quantize the frequencies to their closest digital value
    freqResolution = sampleRate / nSamples
    quantizedFreqList = np.round(freqList / freqResolution) * freqResolution
    iqPhaseOffsRadList = np.deg2rad(iqPhaseOffsList)

    # generate each signal
    iValList = []
    qValList = []
    dt = 1. / sampleRate
    t = dt * np.arange(nSamples)
    for i in range(len(quantizedFreqList)):
        phi = 2. * np.pi * quantizedFreqList[i] * t
        expValues = amplitudeList[i] * np.exp(1.j * (phi + phaseList[i]))
        iScale = np.sqrt(2.) * iqRatioList[i] / np.sqrt(1. + iqRatioList[i] ** 2)
        qScale = np.sqrt(2.) / np.sqrt(1. + iqRatioList[i] ** 2)
        iValList.append(iScale * (np.cos(iqPhaseOffsRadList[i]) * np.real(expValues) +
                                  np.sin(iqPhaseOffsRadList[i]) * np.imag(expValues)))
        qValList.append(qScale * np.imag(expValues))

    return {'I': np.asarray(iValList), 'Q': np.asarray(qValList), 'quantizedFreqList': quantizedFreqList,
            'phaseList': phaseList}


def generateDacComb(freqList=None, resAttenList=None, phaseList=None, iqRatioList=None,
                    iqPhaseOffsList=None, avoidSpikes=True, globalDacAtten=None, lo=None):
    """
    Creates DAC frequency comb by adding many complex frequencies together with specified amplitudes and phases.

    The resAttenList holds the absolute attenuation for each resonantor signal coming out of the DAC.
    Zero attenuation means that the tone amplitude is set to the full dynamic range of the DAC and the
    DAC attenuator(s) are set to 0. Thus, all values in resAttenList must be larger than globalDacAtten.
    If you decrease the globalDacAtten, the amplitude in the DAC LUT decreases so that the total
    attenuation of the signal is the same.

    Note: The freqList need not be unique. If there are repeated values in the freqList then
    they are completely ignored when making the comb along with their corresponding attenuation, phase, etc...

    INPUTS:
        freqList - list of all resonator frequencies. If None, use self.freqList
        resAttenList - list of absolute attenuation values (dB) for each resonator.
        phaseList - list of phases for each complex signal. If None, generates random phases.
        iqRatioList -
        iqPhaseOffsList -
        avoidSpikes - If True, loop the generateTones() function with random phases to avoid a 90+ percentile spike in the comb

    OUTPUTS:
        dictionary with keywords
        I - I(t) values for frequency comb [signed 32-bit integers]
        Q - Q(t)
        quantizedFreqList - list of frequencies after digitial quantiziation
        dacAtten - The global dac hardware attenuation in dB that should be set

    Attributes:
        self.attenList - overwrites this if it already exists
        self.freqList - overwrites this if it already exists
        self.dacQuantizedFreqList - List of quantized freqs used in comb
        self.dacPhaseList - List of phases used to generate freq comb
        self.dacFreqComb - I(t) + j*Q(t)
    """

    if len(freqList) > nChannels:
        getLogger(__name__).warning("Too many freqs provided. Can only accommodate " + str(nChannels) + " resonators")
        freqList = freqList[:nChannels]

    if len(freqList) != len(resAttenList):
        raise ValueError("Need exactly one attenuation value for each resonant frequency!")

    if (phaseList is not None) and len(freqList) != len(phaseList):
        raise ValueError("Need exactly one phase value for each resonant frequency!")

    if iqRatioList is not None and  len(freqList) != len(iqRatioList):
        raise ValueError("Need exactly one iqRatio value for each resonant frequency!")

    if iqPhaseOffsList is not None and len(freqList) != len(iqPhaseOffsList):
        raise ValueError("Need exactly one iqPhaseOffs value for each resonant frequency!")

    getLogger(__name__).debug('Generating DAC comb...')

    if globalDacAtten is None:
        globalDacAtten = np.amin(resAttenList)
        autoDacAtten = True
    else:
        autoDacAtten = False

    # Calculate relative amplitudes for DAC LUT
    nBitsPerSampleComponent = nBitsPerSamplePair / 2
    maxAmp = int(np.round(2 ** (nBitsPerSampleComponent - 1) - 1))  # 1 bit for sign
    amplitudeList = maxAmp * 10 ** (-(resAttenList - globalDacAtten) / 20.)


    # Calculate nSamples and sampleRate
    nSamples = nDacSamplesPerCycle * nLutRowsToUse
    sampleRate = dacSampleRate

    # Calculate resonator frequencies for DAC
    LOFreq = parse_lo(lo, freqList=freqList)

    dacFreqList = freqList - LOFreq
    dacFreqList[dacFreqList < 0.] += dacSampleRate  # For +/- freq

    # Make sure dac tones are unique
    dacFreqList, args, args_inv = np.unique(dacFreqList, return_index=True, return_inverse=True)

    rstate = np.random.get_state()
    np.random.seed(0)
    toneParams = {
        'freqList': dacFreqList,
        'nSamples': nSamples,
        'sampleRate': sampleRate,
        'amplitudeList': amplitudeList[args]}
    if phaseList is not None:
        toneParams['phaseList'] = phaseList[args]
    if iqRatioList is not None:
        toneParams['iqRatioList'] = iqRatioList[args]
    if iqPhaseOffsList is not None:
        toneParams['iqPhaseOffsList'] = iqPhaseOffsList[args]

    # Generate and add up individual tone time series.
    # This part takes the longest
    toneDict = generateTones(**toneParams)
    iValues = toneDict['I'].sum(axis=0)
    qValues = toneDict['Q'].sum(axis=0)

    # check that we are utilizing the dynamic range of the DAC correctly
    sig_i = iValues.std()
    sig_q = qValues.std()
    # 10% of the time there should be a point this many sigmas higher than average
    expectedHighestVal_sig = scipy.special.erfinv((len(iValues) - 0.1) / len(iValues)) * np.sqrt(2)
    if avoidSpikes and sig_i > 0 and sig_q > 0:
        while max(np.abs(iValues).max() / sig_i, np.abs(qValues).max() / sig_q) >= expectedHighestVal_sig:
            getLogger(__name__).warning("The freq comb's relative phases may have added up sub-optimally. "
                                        "Calculating with new random phases")
            toneParams['phaseList'] = None  # If it was defined before it didn't work. So do random ones this time
            toneDict = generateTones(**toneParams)
            iValues = toneDict['I'].sum(axis=0)
            qValues = toneDict['Q'].sum(axis=0)

    np.random.set_state(rstate)

    dacQuantizedFreqList = (toneDict['quantizedFreqList'])[args_inv]
    dacPhaseList = (toneDict['phaseList'])[args_inv]

    if autoDacAtten:
        highestVal = np.max((np.abs(iValues).max(), np.abs(qValues).max()))
        dBexcess = 20. * np.log10(highestVal / maxAmp)
        dBexcess = np.ceil(4. * dBexcess) / 4.  # rounded up to nearest 1/4 dB
        # reduce to fit into DAC dynamic range and quantize to integer
        iValues_new = np.round(iValues / 10. ** (dBexcess / 20.)).astype(np.int)
        qValues_new = np.round(qValues / 10. ** (dBexcess / 20.)).astype(np.int)
        if np.max((np.abs(iValues).max(), np.abs(qValues).max())) > maxAmp:
            dBexcess += 0.25  # Since there's some rounding there's a small chance we need to decrease by another atten step
            iValues_new = np.round(iValues / 10. ** (dBexcess / 20.)).astype(np.int)
            qValues_new = np.round(qValues / 10. ** (dBexcess / 20.)).astype(np.int)

        globalDacAtten -= dBexcess
        if globalDacAtten > 31.75 * 2.:
            dB_reduce = globalDacAtten - 31.75 * 2.
            getLogger(__name__).warning("Unable to fully utilize DAC dynamic range by " + str(dB_reduce) + "dB")
            globalDacAtten -= dB_reduce
            dBexcess += dB_reduce
            iValues_new = np.round(iValues / 10. ** (dBexcess / 20.)).astype(np.int)
            qValues_new = np.round(qValues / 10. ** (dBexcess / 20.)).astype(np.int)

        iValues = iValues_new
        qValues = qValues_new

    else:
        iValues = np.round(iValues).astype(np.int)
        qValues = np.round(qValues).astype(np.int)

    dacFreqComb = iValues + 1j * qValues
    highestVal = np.max((np.abs(iValues).max(), np.abs(qValues).max()))

    msg = ('\tGlobal DAC atten: {} dB'.format(globalDacAtten) +
           '\tUsing {} percent of DAC dynamic range\n'.format(highestVal / maxAmp * 100.) +
           '\thighest: {} out of {}\n'.format(highestVal, maxAmp) +
           '\tsigma_I: {}  sigma_Q:{}\n'.format(np.std(iValues), np.std(qValues)) +
           '\tLargest val_I: {} sigma. '.format(1.0 * np.abs(iValues).max() / np.std(iValues)) +
           'val_Q: {} sigma.\n'.format(np.abs(qValues).max() / np.std(qValues)) +
           '\tExpected val: ' + str(expectedHighestVal_sig) + ' sigmas\n')
    getLogger(__name__).debug(msg)

    if globalDacAtten < 0.:
        raise ValueError("Desired resonator powers are unacheivable. "
                         "Increase resonator attens by " + str(-1 * globalDacAtten) + "dB")

    return {'I': iValues, 'Q': qValues, 'quantizedFreqList': dacQuantizedFreqList, 'dacAtten': globalDacAtten,
            'comb': dacFreqComb, 'dacPhaseList': dacPhaseList}


def get_gen2_dac_comb(mec_freqfile, lo):

    freqfile = SweepFile(mec_freqfile)

    combdata = generateDacComb(freqList=freqfile.freq, resAttenList=freqfile.atten, phaseList=freqfile.phases,
                               iqRatioList=freqfile.iqRatios, avoidSpikes=True, globalDacAtten=None, lo=lo)

    return combdata['comb']

if __name__ == '__main__':
    mec_freqfile='/Users/one/Desktop/untitled folder/psfreqs_FL8a_clip.txt'
    lo=4428029278.278099

    freqfile = SweepFile(mec_freqfile)
    logging.basicConfig()
    combdata = generateDacComb(freqList=freqfile.freq, resAttenList=freqfile.atten, phaseList=freqfile.phases,
                               iqRatioList=freqfile.iqRatios, avoidSpikes=True, globalDacAtten=None, lo=lo)

