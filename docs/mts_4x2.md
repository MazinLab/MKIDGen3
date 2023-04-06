# Enabling MTS on RFSoC4x2
## Patch XRFDC package
In a terminal in on 4x2:
```angular2html
git clone https://github.com/Xilinx/RFSoC-MTS.git (commit 0e339c3)
cd RFSoC-MTS
git clone https://github.com/xilinx/PYNQ (commit de6b6fc)
cd PYNQ
git apply ../boards/patches/xrfdc_mts.patch
su
pushd sdbuild/packages/xrfdc
. pre.sh
. qemu.sh
```
restart python interpreter