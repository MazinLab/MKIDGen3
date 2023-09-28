# MKIDGEN3 Software Design Description

## Overview
The MKIDGEN3 software facilitates using multiple hardware subsystems (RFSoC board, IF board) to set up and read out and MKID array. 
The core functionality revolves around capturing data at various points in the FPGA DSP pipeline.
Captures are facilitated by `capture requests`.

## Capture Requests


## Array Setup Steps
1. Run sweeps (power and freq.) to find res and drive power
2. Process
3. Rerun 1&2 with fixed freq to finialize optimal drive power
4. Run IQ sweeps to find loop centers
5. Process
6. capture Optimal filter phase data
7. Process
8. capture phase data for thresholding
9. Process
10. ready to observe

## Definitions
- active feedlines: a feedline which is intended to be used for observing and has the requisite calibration/setup information
- observing: recording photon data on all active feedlines 
- capture request: contains an id which is the hash of the capture settings

## Main Programs, and their objects:
- Feedline Readout Server
  - FeedlineReadoutServer
    - FeedlineHardware
    - TapThread
- Readout Director
  - PhotonDataAggregator
- Feedline Redis Server
  - Contains all calibration data for one feedline
    - dac table, IF settings, res frequencies, optimal filters, etc.
- Global Redis Server
  - contains all calibration data for all feedlines
  - can be edited to change individual calibration settings manually, updated settings only get applied when observing is started



Recovery Procedure:
- If a feedline goes down mid observing there are two choices:
  - Restart feedline with exact same settings, observing continues uninterrupted, photon capture IDs are the same
  - Recalibrate one or more feedlines: observing stops 
