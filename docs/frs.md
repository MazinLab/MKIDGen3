#Overview
Client sends capture requests to feedline readout server. Nominally this would be via some 
sort of feedline api to handle connecting, request creation, submission, and other feedline controls.


Client Requests result in a capture object with methods for checking on the state of the capture and some other opeations.
Future that will even


Feedline readout server (FRS) accepts requests for specific actions:
- capture({ADCCapReq, IQCapReq, PhaseCapReq, PhotonCapReq, PostageCapReq})
  - requests are always accepted, if completed the included settings will be used
  - capture requests include:
    - FeedlineSetup to be used for the capture
- reset()
  - Purpose is to reset the FPGA to a know initial state 
  - cancels any running or pending capture requests
  - resets the PL by reloading the bitstream
- rfkill(dac=True, ifboard=True)
  - The purpose of this command is to ensure that a feedline is not emitting any RF
  - cancels any running or pending capture requests
  - capture requests will not start while rf is killed
- cancel(id=None)
  - cancels specified or all running and pending captures
  - individual captures may be canceled by calling .cancel() on the client side
- status()
  - returns the FeedlineStatus
- configure()
  - applies a FeedlineSetup without triggering a capture
  - results in an error if captures are in progress or pending 


The FRS program has a main thread and worker capture threads. Worker threads may be aborted from the main thread or
a connection they have to their capture client. 

Compatible simultaneous captures:
- Photon + Postage + ADC|IQ|PHASE
- Captures' FeedlineSetups must be compatible to be simultaneous

The FRS main thread accepts requests on a control socket (REQ/REP). Captures are each processed in their own threads 
that update on the status of the capture, handle canceling the capture, and getting data shipped off to the destination. 
Capture threads must be able to be terminated from the capture requester OR the FRS control thread.
- Capture requests are placed into an internal queue of some sort for all requests
- Capture manager thread
  - Checks for running capture threads, on thread finish updates the set of needed settings
  - if an abort request comes in look through full queue and pop request if it's for something running signal that thread to die
  - if needed settings have changed since everything in queue was inspected pulls and checks for runnability
  - places at back of queue if not runnable 
  - applies needed settings and starts a capture thread for it
  - publish status of capture to capture status destination
Capture threads:
  - listen for aborts from manager
  - update buffers, fetch data, and send off
  - publish data on global capturedata socket with capture id
  
Anyway you slice it this implies we need an aggregate set FeedlineSetup object of what is running


Assume Capture Client is online and running.
Create a CaptureJob
  - something needs to be subscribed to capture data stream for capture id in a receiver thread
  - creates a CaptureStatusListener for itself
  - send capture request to FRS via control socket
  - send abort request via FRS control socket
  - kills listener and data threads (if needed they don't autodie) on abort 

CaptureJob
- CaptureRequest
  - FeedlineSetup
  - tap
  - dest
  - ....
  - id
- _CaptureStatusListener
- _CaptureDataSaver (if not going to a file)
- status() -> PENDING|RUNNING|COMPLETED|ABORTED|FAILED
- cancel()
- data() -> Future of some sort
- submit()

CaptureStatusListener
  - Subscribes to capture status socket (SUB) for id
  - maintains lists of status updates for id
  - updates() returns the values of the status updates for id
  - del kills listener thread

 


## CaptureRequests data and control:
Capture requests need to include some sort of control port for aborting and possibly requesting status.
We have two broad options for data: 
1) General or specific destinations for the data.
2) PUB or asynchronous REQ/REP (?, maybe thats actually DEALER) for socket tpyes.

Proposal 1: Capture data 
- Data is published to a capture port by capture ID and message type (status | data). Capture threads connect to the 
same publish socket and sends multipart messages [id, messagetype, data].
- Abort and  requests come in over a REQ/REP socket
- main thread abort requests come in over some sort of local socket (zpair?)
- this implies that there is an XPUB proxy running somewhere (i think)



Feedline Readout Server Status:
Returns: 
1. capture running / pending 
    * current active capture request
    * requests in queue?
2. if board status
    * if.status() reports LO setting, error, atten setting, locked?
3. overlay downloaded?
    * bitfile name
4. Board PLLs programmed?
5. Dac status
    * programmed / outputting?
    * latest dac output spec
    * current output
6. Channel status
    * is bin to res programmed?
    * what is the channel mapping
7. DDC status
    * is the ddc programmed
    * what are the tone increments, centers, offsets
8. Optimal Filter status
    * are the filters programmed
    * is it a unity filter
9. Trigger status
    * is the trigger running?
    * what are the thresholds
    * what are the holdoffs
10. Setup Engine Progress?
    * what setup steps have been completed? power sweep, rotate loops, optimal filter, thresholding
11. Are we taking photon data?
