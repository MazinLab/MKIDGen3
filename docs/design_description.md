# MKIDGEN3 Software Design Description

## Definitions
- active feedlines: feedlines for which the feedline-specific redis server has enough calibration metadata filled in to record photon data
- Observing: recording photon data on all active feedlines

Main Programs, and their objects:
- Feedline Readout Server
  - FeedlineReadoutServer
    - FeedlineHardware
    - TapThread
- Readout Director
  - PhotonDataAggregator
- 




Recovery Procedure:
- If a feedline goes down mid observing there are two choices:
  - Restart feedline with exact same settings, observing continues uninterrupted, photon capture IDs are the same
  - Recalibrate one or more feedlines: observing stops 
