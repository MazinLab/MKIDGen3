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
See the instructions in [https://github.com/MazinLab/gen3-vivado-top/blob/main/README.md](the firmware repo)

### Board Install

These install instructions require the RFSoC to be connected to the internet. See instructions [in the PYNQ documentation](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z1_setup.html) setup for ethernet connection.

First clone the source code
```
mkdir -p ~/src/mkidgen3/
git clone https://github.com/MazinLab/MKIDGen3.git ~/src/mkidgen3/
git clone https://github.com/Xilinx/RFSoC-MTS.git ~/src
```
Next, patch the `xrfdc` driver using the install script in the `RFSoC-MTS` repo.
```
cd RFSoC-MTS
./install.sh
```
Now install the MKIDGen3 code that runs on the RFSoC
```
cd ~/src/mkidgen3
source /etc/profile.d/pynq_venv.sh
sudo pip install -e '.[server,plotting]'
```

To use the interpreter within PyCharm use /usr/local/share/pynq-venv/bin/python as the remote interpreter, execute with sudo, and ensure that the following envirnoment variables are set:

    BOARD=RFSoC4x2
    XILINX_XRT=/usr
