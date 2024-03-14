import copy
import functools
from logging import getLogger

import zmq
from mkidgen3.server.feedline_config import IFConfig, FeedlineConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest,StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.server.feedline_config import FeedlineConfigManager
from mkidgen3.power_sweep_helpers import *
from mkidgen3.opfb import opfb_bin_number
import numpy as np
from mkidgen3.util import setup_logging
from typing import Iterable
setup_logging('feedlineclient')
import threading
# ctx = zmq.Context.instance()
# ctx.linger = 0
import pickle

frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)

# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz

# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)

# IF Config
if_board = IFConfig(lo=6000, adc_attn=50, dac_attn=50)

# DAC Config
with open('psweep_1024_offset_WaveformConfig.pkl', 'rb') as inp:
    waveform = pickle.load(inp)

# Bin2Res Config
chan = waveform.default_channel_config

# DDC Config
ddc = waveform.default_ddc_config

# Filter Config
filtercfg=FilterConfig(coefficients=f'unity{2048}')

trigconfig = TriggerConfig()

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform, chan=chan, ddc=ddc,
                    filter=filtercfg, trig=trigconfig)
#gsm = StatusListener(b'', frsb.status_url)
#cr = CaptureRequest(2**14, 'ddciq', fc, frsa)
#cr = CaptureRequest(1024**3//4//2048, 'iq', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/iq1024MiB.npz')
#cr = CaptureRequest(1024**3//2//2048, 'phase', fc, frsa, file='file:///home/xilinx/wheatley/jbtest/phase1024MiB.npz')
#cr = CaptureRequest(3024**3//2//2048, 'phase', fc, frsa)
#cr = CaptureRequest(2**19, 'adc', fc, frsa)


## DEFINE ANALOG SIGNAL CHAIN IN FRIDGE

paramp = Amplifier("paramp", gain=12, saturation=-40)
hemt = Amplifier("hemt", gain=40, saturation=-40)
if1 = Amplifier("if1", gain=14.5, saturation=25)
if2 = Amplifier("if2", gain=14.5, saturation=25)
if3 = Amplifier("if3", gain=14.5, saturation=25)
if4 = Amplifier("if1", gain=14.5, saturation=25)
if5 = Amplifier("if2", gain=15.0, saturation=20)
if6 = Amplifier("if3", gain=15.0, saturation=20)
attn1 = Attenuator("if_fixed_atten", gain=-1)
dac_attn = ProgrammableAttenuator("dac_atten", gain=-30)
adc_attn1 = ProgrammableAttenuator("adc_atten1", gain=-10)
adc_attn2 = ProgrammableAttenuator("adc_atten2", gain=0)
attn30 = Attenuator("mix", gain=-30)
device_attn = Device("mkid", gain=-3)
attn20 = Attenuator("4K", gain=-20)
attn10 = Attenuator("cable atten", gain=-10)

signal_chain = ROChain(dac_attn, attn20, attn10, attn30, device_attn, paramp, hemt, if1, adc_attn1, if2, adc_attn2, if3, if4, if5, attn1, if6)
dev_pow_start, dev_pow_stop = -95, -110
adc_input_pow = -1
dac_output_pow = -33 #  This depends on N tones

# STARTING VALUES FOR ADC DAC ATTEN
dac_attn_start, adc_attn_start = calculate_adc_dac_attn(dac_output_pow, dev_pow_start, signal_chain, adc_input_pow)
dac_attn_stop, _ = calculate_adc_dac_attn(dac_output_pow, dev_pow_stop, signal_chain, adc_input_pow)

# COMPUTE ATTEN AND LO LISTS
attns = compute_power_sweep_attenuations(dac_attn_start, adc_attn_start, dac_attn_stop)
lo_sweep_freqs = compute_lo_steps(center=0, resolution=7.14e3, bandwidth=2e6)


class PowerSweepJob:
    """
    This class is responsible for gathering power seep data.
    """
    def __init__(self, lo_sweep_freqs: Iterable[float | int], attns: list[tuple[float | int, float | int]], fc: FeedlineConfig, iq_avg: int = 1024, server=None, save_as=None):
        self.lo_sweep_freqs = lo_sweep_freqs
        self.attns = attns
        self.iq_avg = iq_avg
        self.server = server
        self.save_as = save_as
        self._fc = fc
        self._nchannels = len(fc.waveform.waveform.freqs)
        self._thread = None
        self.die = threading.Event()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('May only start once!')
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = False
        self.die.clear()
        self._thread.start()

    def _run(self):
        self._complete = False
        result = np.zeros((self._nchannels, len(self.lo_sweep_freqs), len(self.attns)), dtype=np.complex64)
        chan = np.arange(self._nchannels)
        for atten_i, attn in enumerate(self.attns):
            for lo_i, lo in enumerate(self.lo_sweep_freqs):
                self._fc.if_board = IFConfig(lo=lo, dac_attn=attn[0], adc_attn=attn[1])
                j = CaptureJob(CaptureRequest(self.iq_avg, 'ddciq', self._fc, self.server,
                                              channels=chan, numpy_metric='mean'))
                try:
                    print('submitting job')
                    j.submit(True, True)
                    print('submitted job')
                    result[:, lo_i, atten_i] = j.data(timeout=60).data  # block until data is received
                    print('received data')
                except KeyboardInterrupt:
                    getLogger(__name__).error(f'Keyboard Interrupt, aborting and shutting down')
                    j.cancel()
                    break
                if self.die.is_set():
                    print('die is set')
                    break
                if not lo_i:
                    self._fc = self._fc.hashed_form
        self._complete = True
        print('completed')
        self.result = result

lo_sweep_freqs = [6001,6000]
attns=[(50,50), (40,40)]
psj = PowerSweepJob(lo_sweep_freqs, attns, fc, server=frsa)
j_norm = CaptureJob(CaptureRequest(512, 'ddciq', fc, frsb,  channels=np.arange(psj._nchannels), numpy_metric='mean', file=None))
#
psj.start()
j_norm.submit(True, True)
print('hi')