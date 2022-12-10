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
    * what is the channel
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
