# MKIDGen3 - The Mazinlab Third Generation MKID Readout 

This is the top-level repository for the third generation RF multiplexed readout of microwave kinetic inductance detectors (TODO cite). We generally refer to the overall system as "gen3". The readout is under active development around the RFSoC4x2 Academic Eval board.

This document, package, and the firmware are still in high flux. APIs, resource calculations, and drivers are only partially done. Various subsystems are in myriad states of test. Liberal communication with the authors is both welcomed and advised. The remainder of this readme focuses on the python side of things. For more information about the gateware and Vitis HLS blocks see the readme and design documents in the `firmware` subdirectory. 

### Download the Repo
`git clone --recursive https://github.com/MazinLab/MKIDGen3.git`

To install the client version of the repo for interacting with the board:

```
cd MKIDGen3
pip install -e '.[client,plotting]'
```

### Build the Bitstream
See the instructions in [the firmware repo](https://github.com/MazinLab/gen3-vivado-top/blob/main/README.md).

### PYNQ setup

Login to the 4x2 and apply the changes described in [the pynq deviations document](https://github.com/MazinLab/MKIDGen3/blob/develop/docs/pynq_deviations.md) to enable multi-tile synchronization, make the PL DDR4 available, and enable the ifboard, then install this package with:

    cd ~
    mkdir -p ~/src/mkidgen3/
    git clone https://github.com/MazinLab/MKIDGen3.git ~/src/mkidgen3/
    cd ~/src/mkidgen3
    pip install -e '.[server,plotting]'

You will likely want to do this in the `pynq-venv` (`source /etc/profile.d/pynq_venv.sh` as root) otherwise you will manually have to install the `xrfclk` package and the (patched) `xrfdc` package as they are not published on pypi or conda-forge

To use the interpreter within PyCharm use /usr/local/share/pynq-venv/bin/python as the remote interpreter, execute with sudo, and ensure that the following envirnoment variables are set:

    BOARD=RFSoC4x2
    XILINX_XRT=/usr
