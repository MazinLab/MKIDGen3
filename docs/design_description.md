# MKIDGEN3 Software Design Description

## Overview
The MKIDGEN3 software facilitates using multiple hardware subsystems (RFSoC board, IF board) to set up and read out an MKID array. 
The core functionality revolves around capturing data at various points in the FPGA DSP pipeline.
Captures are facilitated by `capture requests`. The `Director` progam runs on the client machine and facilitates
generating capture requests to fufill array setup steps. Once all calibration settings have been established, it facilitates
a standing capture request to record photons. The `GUI` can be clicked to call director functions and facilitate
array setup steps in a graphical way. 

## Key Components
### Capture Requests
Capture requests can target three possible locations on the FPGA:
1. Setup / Engineering
   2. ADC Capture
   3. IQ Capture (Post-DDC, Pre-optimal filter)
   4. Phase Capture (Post-optimal filter)
2. Postage stamp (post optimal filter IQ timestreams for 8 pixels)
3. Photon capture

All three locations can run captures concurrently but the setup / engineering capture only supports
one sub-location at a time. Every capture request is tracked via a `capture_id` which is a hash consisting
of all the settings related to the capture. Identical capture settings will produce the same capture ID.
The capture ID(s) can be used by a subscriber machine to filter capture requests published by the FRS. 
*TODO: Consider including the capture location / type in the zmq header such that it can be used by a subscriber
to filter. Jenny thinks this is a good idea.* A capture request may fail and/or abort at any time. Upon 
completion (including failure) a single null data byte will be published. 

### FPGA Redis Server
The FPGA Redis Server runs on the ARM core and is responsible for two main tasks:
1. Keep updated information on the current status of any capture requests (queued, in-progress, completed)
*Will every single capture request be recorded in the Redis Server? What counts as "current"?
Jenny thinks this needs to be flushed out / discussed a little more.*
2. Keep updated information of the status of the FPGA programming including the DAC table settings, 
optimal filter taps, bin2res settings, etc.

*Is there a seperate utility for tracking setup steps / what's potentially out of date or is that 
also a function of this server? Who is responsible for storing MKID feedline/array information like 
resonator frequncies and powers? However these are stored needs to be convenient to hand off to different
programs, view, associate with data, etc.*

*TODO: Where/how are logging messages stored or published?*



### Feedline Readout Server
The Feedline Readout Server (FRS) facilitates programming and interacting with the FPGA.

The FRS accepts capture requests on the capture request port and processes them in a loose priority order, executing requestes as they are 
compatible with those previously received and running. 

Anyone is able to subscribe to published data and they are able to filter by capture ID (and capture type if we inplement that). 
Only the computer that generated the capture request necessarily knows the capture ID(s) for the 
requests they submitted.





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
- Feedline Readout Server (FRS)
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

## Usage Scenarios 

Full array setup sans-gui
1. Start global redis server
2. start FL redis servers (optional)
3. Start all FRSs
4. Create some sort of PowerSweepEngine
  - needs board to use (pull from redis by default or by explicit list)
  - needs the power sweep settings
  - needs processing method and config settings
5. Tell engine to go
6. Engine generates and submits capture jobs harvesting and storing the data
  - handles hiccups with resume ability
  - stores what settings it used into redis: state:last_powersweep:...
7. once all data is recieved it processes the data per its config and stores the result 
  - in redis: state:last_powersweep:result:...
  - in a file at a location specified by the current configuration
8. Create some sort of rotateloopsengine

## Redis command, control, status schema
state:fl#:....
config:fl#...
