# UCSB Gen3 Design
Disclaimer: This document is still very much in flux and many resources calculations are only partially done, especially
for BRAM usage. Some of the linked projects do not yet exist and the names and links are suggestive placeholders. In
many cases there are alternative/backup approaches that are not described.

[Gen3 Python Package](https://github.com/MazinLab/MKIDGen3)

[Top Level Vivado Project](https://github.com/MazinLab/gen3-vivado-top)

## RF Data Converter Block
Detailed configuration of this block is TBD.

## Frequency Comb Generation
The comb is computed in python using repackaged code from gen2 on a per-feedline basis. (TODO Modify the gen2 code
so it uses sane amounts of ram.) The comb, as a 2x8x2^15 element LUT, is loaded as signed shorts into a BRAM that
drives the DAC via gen3.firmware.combgenerator.lut.write(lut). Assuming the DAC has been properly configured, replay is
controlled by setting/clearing gen3.firmware.combgenerator.run.
   
### dac-replay-ip
https://github.com/MazinLab/dac-replay-ip

This IP Subsystem consists of a mr_table block and associated control blocks necessary to feed a FL's DAC with a LUT
-based waveform. An alternate replay block could be created in HLS.

## Readout
The readout subsystem is significantly more complicated than the comb generator. It consists of an oversampled
polyphase filter bank (OPFB), followed by resonator channel selector, direct digital down conversion, low-pass
filtering, photon detection, and event packaging.

### Overlapped Polyphase Filter
Inputs:
- I 256bit AXI4S of 16 bit words
- Q 256bit AXI4S of 16 bit words

Outputs:
- 1 576bit AXI4S of 16 18 bit complex equipped with TLAST

Utilization:
- 256 (FIR) + 144 (FFT) DSP48

The [OPFB subsystem](https://github.com/MazinLab/gen3-opfb) is composed of several blocks: I and Q streams from the
 ADCs are ingested by the 
[adc-to-opfb](https://github.com/MazinLab/adc-to-opfb) HLS block that, the output is fed into a bank of 16 Xilinx FIR
Compiler blocks that are controlled by a small HLS coefficient select block 
([opfb-fir-cfg](https://github.com/MazinLab/opfb-fir-cfg)). The output of each of these blocks is fed through 
a [opfb-fir-to-fft](https://github.com/MazinLab/opfb-fir-to-fft) HLS block (16 total) and then these 16 streams are
fed into a 16x super sample rate 4096 point FFT block from the . The output of the FFT is packaged appropriately by
TODO and output. 

- NB Framing options: (TODO)

#### adc-to-opfb
Inputs: 
- I 128bit AXI4S of 16 bit words
- Q 128bit AXI4S of 16 bit words

Outputs:
- 16x IQ 32bit AXI4S of 16bit complex words, streams equipped with TLAST.

This block takes the incoming 8 I & Q samples and bundles them into complex shorts for internal operations. It then
breaks the 8 IQ into 16 by applying feeding the even and odd lanes with every other sample (i.e. sample 0, 16, 32 -> 
lane0; 8, 24, 40 -> lane1; sample 1, 17, 33 -> lane2; etc). Each lane keeps a delay line of 128 samples and uses the
delay line to drive the output in 'off' cycles, thereby resulting in output that looks like 0, -127, 1, -126, ... 
for a given lane. The resulting pattern may be seen in full, along with the intended FIR coefficient set (TODO) for
multiplication in the simulation output. TLAST (TODO) is set on the 256th cycle of each lane. 

#### FIR Bank & Control
The 16 FIRs are configured for 512 channel 2 parallel path 16 bit input, 18 bit output with 256 coefficient sets.
Configuration is by channel with the channel HLS block configuring coefficient set order as 0, 0, 1, 1, 2, 2, ....
Coefficient reload is disabled. The FIRs are not presently planned to do anything with axi4s side channel
information aside from TLAST passthrough. TODO: Revisit side channel handling for datastream framing as OPFB progresses.

#### opfb-fir-to-fft
Inputs: 
- IQ 36bit AXI4S of 18 bit complex equipped with TLAST

Output: 
- IQ 36bit AXI4S of 18 bit complex equipped with TLAST

This block takes an IQ stream from a lane's FIR and reorderes them as needed to feed the FFT. It breaks the incoming
stream into even (group A) and odd samples, with the odd samples grouped into a group of the first 128 (group B) and the
second 128 (group C). Samples are stored in internal FIFOs and sent on in order from group A, then C, then B, thereby
achieving the alternating cycle reorder required by the OPFB. The results of the are available from the C simulation of
the core.

#### Vector FFT
Inputs: 
- 16x IQ 36bit AXI4S of 18 bit complex equipped with TLAST

Output: 
- 1 576bit AXI4S of 16 18 bit complex equipped with TLAST

The FFT is done by the Xilinx Vector FFT from the SSR Blockset for System Generator. The inputs into the FFT must be
connected in an interleaved fashion to ensure the expected natural bin order. The first 8 inputs coming from the even
lanes and the second 8 inputs from the odd lanes. TLAST is asserted on the cycle containing the last 16 bins of the FFT.

The HLS SSR FFT is not used as of HLx 2019.2 as:
- There are windows/linux compile issues
- Initial tests show it may not make timing

### Resonator Channelization 
Inputs: 
- 2 288bit AXI4S of 16 18 bit numbers equipped with TLAST
- AXI-Lite resonator map port, in the  (TODO, Need to describe)

Outputs:
- 1 288bit AXI4S of 8 18 bit complex equipped with TLAST and possibly (TODO) TKEEP

This IP subsystem [gen3-reschan](https://github.com/MazinLab/gen3-reschan)takes the raw bin IQ streams, selects the
values corresponding to resonators, digitally down-converts each with the 
[resonator-dds](https://github.com/MazinLab/resonator-dds) HLS block, and then the samples are then run through an
FIR Compiler core to low-pass and decimate them.

#### opfb-bin-to-res Selection
Inputs: 
- 2 288bit AXI4S of 16 18 bit numbers (I and Q streams), both equipped with TLAST
- AXI-Lite resonator map port (256x8 ap_uint<12> array).

Outputs:
- 1 288bit AXI4S of 8 18 bit complex equipped with TLAST and possibly (TODO) TKEEP

The resonator selection [HLS block](https://github.com/MazinLab/opfb-bin-to-res) ingests the IQ bins output by the
OPFB subsystem and extracts and outputs the IQ values corresponding to resonators. This is done by caching the bin
spectrum (updating the next 16 bins each clock) and fetching 8 bins containing resonators each clock. This is
implemented as 8 banks of ~147kbit BRAM in order to have sufficient memory ports. The bins used are stored in a bin
to resonator LUT (2048 x 12 bits) loaded in from python via AXI Lite. 

####  resonator-dds
Inputs: 
- 1 288bit AXI4S of 8 18 bit complex equipped with TLAST and possibly (TODO) TKEEP or TUSER
- Quarter wave SIN/COS LUT of length... This is implemented in ROM and sized to support 8 simultaneous reads.
- Resonator frequency table stored as a phase increment LUT.
- Resonator phase shift table stored as a phase zero.

Outputs:
- 1 288bit AXI4S of 8 18 bit complex equipped with TLAST and possibly (TODO) TKEEP

Utilization:
- 24 DSP48

The HLS block uses the TLAST on the inbound stream to keep track of the cycle and query phase and phase increment
LUTs for the received group of resonators. These are used increment/maintain an accumulated phase for each resonator
which is used to query a quarter-wave sin/cos LUT (stored in ROM). Phase offsets are used to apply a per-resonator 
phase shift (rotate loops in gen2 parlance). Presently the result is neither dithered nor Taylor corrected, however
there is example Xilinx HLS code for how to implement those features as well. The resulting sin/cos values are fed 
into a Xilinx complex multiplier (3 DSP slices per IQ, 24 in total) and the results passed out of the block.

#### FIR Core
Utilization:
- 320 (or 240) 

The fir is configured for 256 channel, 16 parallel channel (8I, 8Q) operation with 20 taps and 18 bit IO. It is
configured for a decimation of 2, dropping the sample rate to 1 MHZ. This takes ~320 DSP slices, though the core
itself should be able to run at ~820MHz, so with a clock crossing we could run with 8 to 6 to 8 lanes for a total of
240 DSP slices.

### Resonator Processing
Inputs: 
- 288bit AXI4S of 8 18 bit complex equipped with TLAST and possibly (TODO) TKEEP or TUSER
- Timestamp

Outputs:
- 88bit AXI4S photon stream (18bit phase, 18bit baseline, 32bit time, 16bit resID)

IO:
- AXI4M Port
    - Matched filter configuration (in)
    - Per-resonator dropped photon counter (out)
    - FL ID (to convert resonance to resonatorID) (in)
    - Trigger configuration (in)

The [resonator processor subsystem](https://github.com/MazinLab/gen3-resproc) takes in the stream from the 
channelization and breaks it into separate lanes of 256 resonators which are fed into 8 instances of the resonator
[lane subsystem](https://github.com/MazinLab/resonator-stream-lane). It breaks the stream via an HLS block and
implements an Xilinx MCDMA module to route the required configuration streams to the lane subsystems. The lane
subsystems yield triggered photon packets on an XXX (TODO) output, which are gathered by XXX, and output as a stream
of photons. NB that 2500 photons/s * 2048 resonators/FL is ~5e6, so this stream can run quite slowly. 

#### iq-stream-split
This HLS block takes an 8 wide IQ stream and breaks it into 8 streams, attaching the full resonator ID as TUSER.

#### lane-subsystem
Inputs:
- AXI4S 36bit IQ stream with 16bit TUSER
- AXI4S FIR Reload
- ??? Trigger config
- time register

Outputs:
- ??? Photon events

Utilization:
- 200 DSP slices, 138 if FIR at 768MHz (assuming 50/51 taps)

The lane subsystem routes the incoming IQ stream into a Xilinx cordic block configured for arctangent operation, the
resulting stream is then sent to and FIR Compiler core for matched filtering, finally the resulting filtered stream
is routed to to the HLS [photon-trigger](https://github.com/MazinLab/photon-trigger) block which outputs a stream of
photon events.

##### FIR block and HLS channel config
- 200 DSP slices, 138 if FIR at 768MHz (assuming 50/51 taps)

The FIR is configured for 256 channel operation with 51 taps for 256 reloadable coefficient sets and 18 bit IO
It is configured to operate at 768 MHz and for by channel coefficient operation. This takes 17 DSP slices. The
channel configuration is streamed in 0,1,2,3... from a simple VHDL block. The reload channel is an input to the
subsystem. TUSER is passed through.

##### photon-detect
This HLS block takes in the matched phase stream and performs a baseline filter and trigger. It outputs photons on a
fifo photon stream (18bit phase, 18bit baseline, 32bit time, 16bit resID)

#### Photon Packaging
This TBD block combines photons output by the eight lanes and combines them into a photon stream for the feedline. At
2500 photons per second on a full feedline this would be a ~450Mbit stream ~90 bits wide. 

### Photon Processing
This subsystem takes in a (multiple? TDB) photon stream(s) and ships the events off to the PS via AXI4M. The PS
system will then in turn write them them to the instrument disk via an NFS mount. We may also want this block to
maintain some statistics about the photon stream. This statistics would be operating on a ~0.5 - 2 Gbit stream.