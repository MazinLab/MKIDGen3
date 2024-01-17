import zmq
from mkidgen3.server.feedline_config import IFConfig, BitstreamConfig, RFDCClockingConfig, RFDCConfig, WaveformConfig, ChannelConfig, DDCConfig, FeedlineConfig, FilterConfig, TriggerConfig
from mkidgen3.server.captures import CaptureJob, FRSClient, CaptureRequest,StatusListener
from mkidgen3.server.waveform import WaveformFactory
from mkidgen3.opfb import opfb_bin_number
import numpy as np
from mkidgen3.util import setup_logging
setup_logging('feedlineclient')
# ctx = zmq.Context.instance()
# ctx.linger = 0


send_wave = ''

frsa = FRSClient(url='mkidrfsoc4x2.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)
frsb = FRSClient(url='rfsoc4x2b.physics.ucsb.edu', command_port=8888, data_port=8889, status_port=8890)

frsu=frsb

# Bitstream Config
bitstream = BitstreamConfig(bitstream='/home/xilinx/gen3_top_final.bit', ignore_version=True)

# RFDC Clocking Config
rfdc_clk = RFDCClockingConfig(programming_key='4.096GSPS_MTS_dualloop', clock_source=None)  # clock source should default to external 10 MHz

# RDFC Config
rfdc = RFDCConfig(dac_mts=True, adc_mts=False, adc_gains=None, dac_gains=None)

# IF Config
if_board = IFConfig(lo=6000, adc_attn=10, dac_attn=50)

# DAC Config
waveform_vals = WaveformFactory(frequencies=[100e6])
waveform = WaveformConfig(waveform=waveform_vals)
waveform2 = WaveformConfig(waveform=WaveformFactory(frequencies=[150e6]))
freqs = waveform.waveform.freqs

if send_wave == 'hash':
    waveform = waveform.hashed_form
    waveform2 = waveform2.hashed_form
elif send_wave == 'computed':
    waveform.waveform.output_waveform # trigger waveform computation
    waveform2.waveform.output_waveform


# Bin2Res Config
bins = np.zeros(2048, dtype=int)
bins[:freqs.size] = opfb_bin_number(freqs, ssr_raw_order=True)
chan = ChannelConfig(bins=bins)

# DDC Config
ddc_tones = np.zeros(2048)
ddc_tones[:freqs.size]=freqs
ddc = DDCConfig(tones=ddc_tones)

# Feedline Config
fc = FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform, chan=chan, ddc=ddc)
fc2= FeedlineConfig(bitstream=bitstream, rfdc_clk=rfdc_clk, rfdc=rfdc,
                    if_board=if_board, waveform=waveform2, chan=chan, ddc=ddc,
                    filter=FilterConfig(coefficients='unity20'),
                    trig=TriggerConfig(holdoffs=[20]*2048, thresholds=[0]*2048))
# gsm = StatusListener(b'', frsb.status_url)
cr = CaptureRequest(3*1024**3//4, 'adc', fc, frsu, file='file:///home/xilinx/wheatley/jbtest/adc3096MiB.npz')
cr = CaptureRequest(1024**3//4//2048, 'iq', fc, frsu, file='file:///home/xilinx/wheatley/jbtest/iq1024MiB.npz')
cr = CaptureRequest(1024**3//2//2048, 'phase', fc, frsu, file='file:///home/xilinx/wheatley/jbtest/phase1024MiB.npz')
cr = CaptureRequest(3024**3//2//2048, 'phase', fc, frsu)
cr = CaptureRequest(2**19, 'adc', fc, frsu)
cr = CaptureRequest(3*1024**3//4, 'adc', fc, frsu)

j = CaptureJob(cr)
j.submit(True, True)
#
# cr2 = CaptureRequest(2**19, 'adc', fc2, frsu)
#
# j2 = CaptureJob(cr2)
# j2.submit(True, True)

# cr3 = CaptureRequest(100, 'postage', fc2, frsu, channels=[0,1,2])
# j3 = CaptureJob(cr3)
# j3.submit(True, True)


cr4 = CaptureRequest(100, 'photon', fc2, frsu)
j4 = CaptureJob(cr4)
j4.submit(True, True)


# import asyncio
# async def foo():
#     self.register_map.IP_IER.CHAN0_INT_EN = 0
#     self.register_map.IP_ISR = 1
#     coro=self.interrupt.wait()
#     self.register_map.IP_IER.CHAN0_INT_EN = 1
#     self.register_map.IP_ISR = 1
#     await coro
#     self.register_map.IP_ISR = 1
#
# loop = asyncio.new_event_loop()
# task = loop.create_task(foo())
# loop.run_until_complete(task)


raise RuntimeError
