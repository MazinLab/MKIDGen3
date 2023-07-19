# MTS Overview

The Multi-Tile Synchronization proceadure allows a designer to synchronize the operation of converters across multiple tiles. On the 4x2 we synchronize ADC tiles 224 and 226, and DAC tiles 228 and 230. In broad strokes sychronization proceeds in 3 steps:

1. We reset all tiles and initlize the MTS engines in those tiles
2. The phase of the sample clock is adjusted until the edge of the SYSREF signal occurs in roughly middle of a sample clock period in each tile
4. The SYSREF is sampled through all of the converters cross-clock FIFOs and the latencies through those FIFOs are adjusted so that the latency is the same between tiles

For a slightly more comprehensive overview see the relevant xilinx [appnote]([url](https://docs.xilinx.com/r/en-US/xapp1349-rfdc-subsystems/Summary)).

# MTS Setup

At present (July 18th, PYNQ 3.0.1) the `xrfdc` package requires patches which can be obtained from [xilinx](https://github.com/Xilinx/RFSoC-MTS) in order to enable MTS:

## Patch XRFDC package
In a terminal in on 4x2:
```bash
git clone https://github.com/Xilinx/RFSoC-MTS.git # (commit 0e339c3)
cd RFSoC-MTS
git clone https://github.com/xilinx/PYNQ # (commit de6b6fc)
cd PYNQ
git apply ../boards/patches/xrfdc_mts.patch
su
pushd sdbuild/packages/xrfdc
. pre.sh
. qemu.sh
```
Once this is done restart the interpeter and you should be ready to use MTS

## Enable MTS Debug Info

Enabling debug output from MTS requires two roundabout patches:

```patch
diff --git a/sdbuild/packages/xrfdc/package/src/libmetal_stubs.c b/sdbuild/packages/xrfdc/package/src/libmetal_stubs.c
index 7d11ab8f..64330520 100644
--- a/sdbuild/packages/xrfdc/package/src/libmetal_stubs.c
+++ b/sdbuild/packages/xrfdc/package/src/libmetal_stubs.c
@@ -60,6 +60,6 @@ unsigned int metal_register_generic_device(void* h) { return 0; }
 unsigned int metal_device_open(void* h) { return 0; }
 
 __attribute__((constructor)) void foo(void) {
-    _metal.common.log_level = METAL_LOG_WARNING;
+    _metal.common.log_level = METAL_LOG_DEBUG;
     _metal.common.log_handler = metal_default_log_handler;
 }
```

```patch
diff --git a/sdbuild/packages/xrfdc/package/xrfdc/__init__.py b/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
index edddc3e2..8e706a72 100644
--- a/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
+++ b/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
@@ -21,6 +18,8 @@ _ffi.cdef(header_text)
 
 _lib = _ffi.dlopen(os.path.join(_THIS_DIR, 'libxrfdc.so'))
 
+DEBUG_MODE = False
+
 
 # Next stage is a simple wrapper function which checks the existance of the
 # function in the library and the return code and throws an exception if either
@@ -40,6 +39,13 @@ def _safe_wrapper(name, *args, **kwargs):
         if stderr:
             message += f"\nstderr: {stderr}"
         raise RuntimeError(message)
+    elif DEBUG_MODE:
+        stdout = c1.read()
+        stderr = c2.read()
+        if stdout:
+            print(stdout)
+        if stderr:
+            print(stderr)
 
 
 # To reduce the amount of typing we define the properties we want for each
```

You can then get debug output by setting `xrfdc.DEBUG_MODE` to true

# `mkidgen3` MTS API

At a top level MTS can be enabled by passing the argument `mts = True` to in the argument to [`mkidgen3.configure`](https://github.com/MazinLab/MKIDGen3/blob/3bb12604067f6d7ba6beea005e6137ab4ab1f301/mkidgen3/overlay_helpers.py#L130) with a valid clock configuration key like `'4.096GSPS_MTS'`.

For finer control you can set a target latency with the `dac_target`/`adc_target` argument to the overlay method [`rfdc.sync_tiles`](https://github.com/MazinLab/MKIDGen3/blob/3bb12604067f6d7ba6beea005e6137ab4ab1f301/mkidgen3/drivers/rfdc.py#L158) it is required that this latency be larger than the latency the board started up with, and you must call [`rfdc.enable_mts()`](https://github.com/MazinLab/MKIDGen3/blob/3bb12604067f6d7ba6beea005e6137ab4ab1f301/mkidgen3/drivers/rfdc.py#L101) first. Startup latencies larger than 112 have not been observed on boards we have tested on so far.
