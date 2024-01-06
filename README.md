# MKIDGen3 - The Mazinlab Third Generation MKID Readout 

This is the top-level repository for the third generation RF multiplexed readout of microwave kinetic inductance detectors (TODO cite). We generally refer to the overall system as "gen3". The readout is under  active development around the Xilinx ZCU111 Eval board.

This document, package, and the firmware are still in high flux. APIs, resource calculations, and drivers are only partially done. Various subsystems are in myriad states of test. Liberal communication with the authors is both welcomed and advised. The remainder of this readme focuses on the python side of things. For more information about the gateware and Vitis HLS blocks see the readme and design documents in the `firmware` subdirectory. 

### Download the Repo
`git clone --recurse-submodules https://github.com/MazinLab/MKIDGen3.git`

### Build the Bitstream
```
cd firmware
source <path/to/vivado/2022.1>
make gen3_top
```

### PYNQ setup

SSH into the ZCU.

    cd ~
    mkdir -p ~/src/mkidgen3/
    git clone https://github.com/MazinLab/MKIDGen3.git ~/src/mkidgen3/
    cd ~/src/mkidgen3
    git checkout develop
    cd ~/src
    git submodule init arduino/ifduino
    git submodule update arduino/ifduino
    sudo cp udev/gen3.rules /etc/udev/rules.d/  #If using the gen2 ifboard
    source /etc/profile.d/pynq_venv.sh
    sudo pip3 install fpbinary pyserial
    sudo pip3 install -e ~/src/mkidgen3

To use the interpreter within PyCharm use /usr/local/share/pynq-venv/bin/python as the remote interpreter, execute with sudo, and ensure that the following envirnoment variables are set:

    BOARD=ZCU111
    XILINX_XRT=/usr
    
### Some basic usage:

    import logging
    import pynq, xrfclk, xrfdc
    import numpy as np
    import matplotlib.pyplot as plt
    import mkidgen3 as g3
    from mkidgen3.util import buf2complex
    from mkidgen3.ifboard import IFBoard

    logging.basicConfig()
    logging.getLogger('mkidgen3').setLevel('INFO')

    ol = g3.configure('iqtest.bit', clocks=True, external_10mhz=False, ignore_version=True)

    tones = np.array([100e6])
    amplitudes = np.array([1])

    dactable = g3.set_waveform(tones, amplitudes, fpgen=lambda x: (x*2**15).astype(np.uint16))

    raw_data=buf2complex(ol.capture.capture_adc(2**19))

    opfb_data = g3.capture_opfb(n=256)

    g3.set_channels(tones)  #NB we may not yet have the frequency -> bin conversion correct
    g3.configure_ddc(tones)  # Same caveat applies here

    ol.photon_pipe.reschan.bin_to_res.bins = [0,2,3]  # Some list of bins to capture
    iq_data=buf2complex(ol.capture.capture_iq(256, 'all', tap_location='iq'))

The iqtest and OPFB_test notebooks have more examples (as do some others) though these notebooks need significant cleanup.  
