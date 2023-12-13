# PYNQ Image Patches

## MTS Enablement

This patch can be applied by using the proceadure described in [`mts_4x2.md`](https://github.com/MazinLab/MKIDGen3/blob/develop/docs/mts_4x2.md) the patches to enable debug info are not required but are applied on rfsoc4x2b.

```diff
diff --git a/sdbuild/packages/xrfdc/package/Makefile b/sdbuild/packages/xrfdc/package/Makefile
index 2fc69230..9cb219f5 100644
--- a/sdbuild/packages/xrfdc/package/Makefile
+++ b/sdbuild/packages/xrfdc/package/Makefile
@@ -25,7 +25,7 @@ $(LIBRARY): $(EMBEDDEDSW_DIR) $(LIB_METAL_INC)
 
 install:
 	cp $(LIBRARY) $(PACKAGE)/
-	pip3 install .
+	pip3 install . --upgrade
 
 $(EMBEDDEDSW_DIR):
 	git clone https://github.com/Xilinx/embeddedsw \
diff --git a/sdbuild/packages/xrfdc/package/setup.py b/sdbuild/packages/xrfdc/package/setup.py
index 620fff69..9dadb4d8 100644
--- a/sdbuild/packages/xrfdc/package/setup.py
+++ b/sdbuild/packages/xrfdc/package/setup.py
@@ -12,14 +12,12 @@ long_description = (''.join(readme_lines))
 
 setup(
     name="xrfdc",
-    version='1.0',
+    version='2.0',
     description="Driver for the RFSoC RF Data Converter IP",
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/Xilinx/PYNQ/tree/master/sdbuild/packages/xrfdc',
     license='BSD 3-Clause',
-    author="Craig Ramsay",
-    author_email="cramsay01@gmail.com",
     packages=['xrfdc'],
     package_data={
         '': ['*.py', '*.so', '*.c'],
diff --git a/sdbuild/packages/xrfdc/package/xrfdc/__init__.py b/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
index edddc3e2..83f66bb4 100644
--- a/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
+++ b/sdbuild/packages/xrfdc/package/xrfdc/__init__.py
@@ -1,7 +1,6 @@
 #   Copyright (c) 2018, Xilinx, Inc.
 #   SPDX-License-Identifier: BSD-3-Clause
 
-
 import cffi
 import os
 import pynq
@@ -9,8 +8,6 @@ import warnings
 from wurlitzer import pipes
 
 
-
-
 _THIS_DIR = os.path.dirname(__file__)
 
 with open(os.path.join(_THIS_DIR, 'xrfdc_functions.c'), 'r') as f:
@@ -349,15 +346,6 @@ class RFdcAdcTile(RFdcTile):
 
 
 class RFdc(pynq.DefaultIP):
-    """The class RFdc is bound to the IP xilinx.com:ip:usp_rf_data_converter:2.3,
-    xilinx.com:ip:usp_rf_data_converter:2.4 or xilinx.com:ip:usp_rf_data_converter:2.6.
-    Once the overlay is loaded, the data converter IP will be allocated the driver
-    code implemented in this class.
-
-    For a complete list of wrapped functions see:
-    https://github.com/Xilinx/PYNQ/tree/master/sdbuild/packages/xrfdc/package
-    """
-    
     bindto = ["xilinx.com:ip:usp_rf_data_converter:2.6",
               "xilinx.com:ip:usp_rf_data_converter:2.4", 
               "xilinx.com:ip:usp_rf_data_converter:2.3"]
@@ -378,10 +366,22 @@ class RFdc(pynq.DefaultIP):
         _lib.XRFdc_CfgInitialize(self._instance, self._config)
         self.adc_tiles = [RFdcAdcTile(self, i) for i in range(4)]
         self.dac_tiles = [RFdcDacTile(self, i) for i in range(4)]
+        self.mts_adc_config = _ffi.new('XRFdc_MultiConverter_Sync_Config*')
+        self.mts_dac_config = _ffi.new('XRFdc_MultiConverter_Sync_Config*')
+        _safe_wrapper("XRFdc_MultiConverter_Init", self.mts_adc_config, cffi.FFI.NULL, cffi.FFI.NULL)
+        _safe_wrapper("XRFdc_MultiConverter_Init", self.mts_dac_config, cffi.FFI.NULL, cffi.FFI.NULL)
+
 
     def _call_function(self, name, *args):
         _safe_wrapper(f"XRFdc_{name}", self._instance, *args)
 
+    def mts_adc(self):
+        return _safe_wrapper("XRFdc_MultiConverter_Sync", self._instance, 0, self.mts_adc_config)
+        
+    def mts_dac(self):
+        return _safe_wrapper("XRFdc_MultiConverter_Sync", self._instance, 1, self.mts_dac_config)
+        
+
 
 # Finally we can add our data-driven properties to each class in the hierarchy
 
@@ -448,6 +448,3 @@ TRSHD_OFF                  = 0x0
 TRSHD_STICKY_OVER          = 0x1
 TRSHD_STICKY_UNDER         = 0x2
 TRSHD_HYSTERISIS           = 0x3
-
-
-
diff --git a/sdbuild/packages/xrfdc/package/xrfdc/xrfdc_functions.c b/sdbuild/packages/xrfdc/package/xrfdc/xrfdc_functions.c
index ef80b9ff..8761e936 100644
--- a/sdbuild/packages/xrfdc/package/xrfdc/xrfdc_functions.c
+++ b/sdbuild/packages/xrfdc/package/xrfdc/xrfdc_functions.c
@@ -80,6 +80,44 @@ typedef struct {
 	XRFdc_Distribution DistributionStatus[8];
 } XRFdc_Distribution_Settings;
 
+/**
+ * MTS DTC Settings.
+ */
+typedef struct {
+	u32 RefTile;
+	u32 IsPLL;
+	int Target[4];
+	int Scan_Mode;
+	int DTC_Code[4];
+	int Num_Windows[4];
+	int Max_Gap[4];
+	int Min_Gap[4];
+	int Max_Overlap[4];
+} XRFdc_MTS_DTC_Settings;
+
+/**
+ * MTS Sync Settings.
+ */
+typedef struct {
+	u32 RefTile;
+	u32 Tiles;
+	int Target_Latency;
+	int Offset[4];
+	int Latency[4];
+	int Marker_Delay;
+	int SysRef_Enable;
+	XRFdc_MTS_DTC_Settings DTC_Set_PLL;
+	XRFdc_MTS_DTC_Settings DTC_Set_T1;
+} XRFdc_MultiConverter_Sync_Config;
+
+/**
+ * MTS Marker Struct.
+ */
+typedef struct {
+	u32 Count[4];
+	u32 Loc[4];
+} XRFdc_MTS_Marker;
+
 /**
  * ADC Signal Detect Settings.
  */
@@ -412,6 +450,8 @@ typedef struct {
 } XRFdc;
 
 
+
+
 /***************** Macros (Inline Functions) Definitions *********************/
 #define XRFDC_ADC_TILE 0U
 #define XRFDC_DAC_TILE 1U
@@ -517,4 +557,6 @@ u32 XRFdc_GetDSA(XRFdc *InstancePtr, u32 Tile_Id, u32 Block_Id, XRFdc_DSA_Settin
 u32 XRFdc_SetPwrMode(XRFdc *InstancePtr, u32 Type, u32 Tile_Id, u32 Block_Id, XRFdc_Pwr_Mode_Settings *SettingsPtr);
 u32 XRFdc_GetPwrMode(XRFdc *InstancePtr, u32 Type, u32 Tile_Id, u32 Block_Id, XRFdc_Pwr_Mode_Settings *SettingsPtr);
 u32 XRFdc_ResetInternalFIFOWidth(XRFdc *InstancePtr, u32 Type, u32 Tile_Id, u32 Block_Id);
-u32 XRFdc_ResetInternalFIFOWidthObs(XRFdc *InstancePtr, u32 Tile_Id, u32 Block_Id);
\ No newline at end of file
+u32 XRFdc_ResetInternalFIFOWidthObs(XRFdc *InstancePtr, u32 Tile_Id, u32 Block_Id);
+u32 XRFdc_MultiConverter_Sync(XRFdc *InstancePtr, u32 Type, XRFdc_MultiConverter_Sync_Config *ConfigPtr);
+void XRFdc_MultiConverter_Init(XRFdc_MultiConverter_Sync_Config *ConfigPtr, int *PLL_CodesPtr, int *T1_CodesPtr);
\ No newline at end of file
diff --git a/sdbuild/packages/xrfdc/qemu.sh b/sdbuild/packages/xrfdc/qemu.sh
index bbb35c29..a6fa79aa 100755
--- a/sdbuild/packages/xrfdc/qemu.sh
+++ b/sdbuild/packages/xrfdc/qemu.sh
@@ -9,7 +9,6 @@ set -e
 for f in /etc/profile.d/*.sh; do source $f; done
 
 export HOME=/root
-export BOARD=${PYNQ_BOARD}
 
 cd /root/xrfdc_build
 
```

## Device Tree Modifications

These changes reserve 2G of PS DRAM for the photon fountain (should probably be reduced, must happen on boot), and reserves the window in the memory address space where the PL DRAM lives (may happen on bitstream load but honestly this would be messy).

```diff
diff --git a/boards/RFSoC4x2/petalinux_bsp/meta-user/recipes-bsp/device-tree/files/system-user.dtsi b/boards/RFSoC4x2/petalinux_bsp/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
index 29b5621..f08083e 100644
--- a/boards/RFSoC4x2/petalinux_bsp/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
+++ b/boards/RFSoC4x2/petalinux_bsp/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
@@ -19,6 +19,34 @@
                 clock-frequency = <26000000>;
        };
 
+       reserved-memory {
+               #address-cells = <2>;
+               #size-cells = <2>;
+
+               ranges;
+
+               bufferreserved: psbuffer@0 {
+                       no-map;
+
+                       reg = <0x08 0x00000000 0x00 0x40000000>;
+               };
+
+               plreserved: plbank@1 {
+                       no-map;
+
+                       reg = <0x05 0x00000000 0x01 0x00000000>;
+               };
+       };
+
+       zocl@0 {
+               compatible = "xlnx,zocl";
+               memory-region = <&bufferreserved>;
+       };
+
+       zocl@1 {
+               compatible = "xlnx,zocl";
+               memory-region = <&plreserved>;
+       };
 };
 
 &gem1 {
@@ -157,4 +185,3 @@
                shunt-resistor = <10000>;
        };
 };
-
```

## PYNQ Python Package

This enables changing the default PS DDR contiguous memory allocation region, there is an open PR to upstream this into pynq

```diff
diff --git a/PYNQ/pynq/pl_server/embedded_device.py b/../pynq/pl_server/embedded_device.py
index 0caa457..02fd48b 100644
--- a/PYNQ/pynq/pl_server/embedded_device.py
+++ b/../pynq/pl_server/embedded_device.py
@@ -227,7 +227,7 @@ class BitstreamHandler:
         else:
             raise CacheMetadataError(f"No cached metadata present")
 
-    def get_parser(self, partial:bool=False):
+    def get_parser(self, partial:bool=False, ps_region = (0, 256*1024)):
         """Returns a parser object for the bitstream
 
         The returned object contains all of the data that
@@ -256,7 +256,7 @@ class BitstreamHandler:
                     raise RuntimeError(f"Unable to parse metadata")
 
             if xclbin_data is None:
-                xclbin_data = _create_xclbin(parser.mem_dict)
+                xclbin_data = _create_xclbin(parser.mem_dict, ps_region)
             xclbin_parser = XclBin(xclbin_data=xclbin_data)
             _unify_dictionaries(parser, xclbin_parser)
 
@@ -267,7 +267,7 @@ class BitstreamHandler:
         elif is_xsa:
             parser = RuntimeMetadataParser(Metadata(input=self._filepath))
             if xclbin_data is None:
-                xclbin_data = _create_xclbin(parser.mem_dict)
+                xclbin_data = _create_xclbin(parser.mem_dict, ps_region)
             xclbin_parser = XclBin(xclbin_data=xclbin_data)
             _unify_dictionaries(parser, xclbin_parser)
             parser.refresh_hierarchy_dict()
@@ -379,15 +379,15 @@ BLANK_METADATA = r"""<?xml version="1.0" encoding="UTF-8"?>
 """
 
 
-def _ip_to_topology(mem_dict):
+def _ip_to_topology(mem_dict, ps_region):
     topology = {
         "m_mem_data": [
             {
                 "m_type": "MEM_DDR4",
                 "m_used": 1,
-                "m_sizeKB": 256 * 1024,
+                "m_sizeKB": ps_region[1],
                 "m_tag": "PSDDR",
-                "m_base_address": 0,
+                "m_base_address": ps_region[0],
             }
         ]
     }
@@ -413,7 +413,7 @@ def _as_str(obj):
     return obj
 
 
-def _create_xclbin(mem_dict):
+def _create_xclbin(mem_dict, ps_region):
     """Create an XCLBIN file containing the specified memories"""
     import json
     import subprocess
@@ -422,7 +422,7 @@ def _create_xclbin(mem_dict):
     with tempfile.TemporaryDirectory() as td:
         td = Path(td)
         (td / "metadata.xml").write_text(BLANK_METADATA)
-        (td / "mem.json").write_text(json.dumps(_ip_to_topology(mem_dict)))
+        (td / "mem.json").write_text(json.dumps(_ip_to_topology(mem_dict, ps_region)))
         completion = subprocess.run(
             [
                 "xclbinutil",
@@ -538,6 +538,7 @@ class EmbeddedDevice(XrtDevice):
     BS_FPGA_MAN_FLAGS = "/sys/class/fpga_manager/fpga0/flags"
 
     _probe_priority_ = 50
+    _ps_region = (0, 256*1024)
 
     @classmethod
     def _probe_(cls):
@@ -560,7 +561,7 @@ class EmbeddedDevice(XrtDevice):
 
         for k, v in mem_dict.items():
             if "base_address" in v:
-                if v["base_address"] == 0:
+                if v["base_address"] == self._ps_region[0]:
                     return self.get_memory(v)
         raise RuntimeError("XRT design does not contain PS memory")
 
@@ -634,6 +635,35 @@ class EmbeddedDevice(XrtDevice):
                         f = ZU_AXIFM_REG[para][reg_name]["field"]
                         Register(addr)[f[0] : f[1]] = ZU_AXIFM_VALUE[width]
 
+    def set_psddr_region(self, address, size):
+        """Sets the default PSDDR allocation area to a reserved region
+
+        This allows you to use any reserved region defined in the device
+        tree as the default allocation target for `pynq.allocate`. You
+        should be careful to only use reserved regions that actually
+        lie within the PSDDR address range.
+
+        `address` should be a byte address and `size` should be the
+        requested size in bytes, and the region must be page alligned.
+
+        This should be called before the bitstream is downloaded eg with:
+
+        ```python
+        pynq.Device.active_device
+        ```
+        """
+        import subprocess
+        completion = subprocess.run(
+            ["getconf", "PAGESIZE"],
+            stdout=subprocess.PIPE,
+        )
+        page_size = int(completion.stdout.strip())
+
+        if (address % page_size) or (size % page_size) or (size % 1024):
+            raise RuntimeError("Memory region must be page aligned")
+
+        self._ps_region = (address, size // 1024)
+
     def gen_cache(self, bitstream, parser=None):
         """ Generates the cache of the metadata even if no download occurred """
         if not hasattr(parser, "_from_cache"):
@@ -687,7 +717,7 @@ class EmbeddedDevice(XrtDevice):
         super().post_download(bitstream, parser, self.name)
 
     def get_bitfile_metadata(self, bitfile_name:str, partial:bool=False):
-        parser = _get_bitstream_handler(bitfile_name).get_parser(partial=partial)
+        parser = _get_bitstream_handler(bitfile_name).get_parser(partial=partial, ps_region = self._ps_region)
         if parser is None:
             raise RuntimeError("Unable to find metadata for bitstream")
         return parser
```

## Miscellaneous

### Additional Packages (RFSOCB, A has a subset)

```
Commandline: apt install zsh
Commandline: apt install nfs-client
Commandline: apt install fuse
Commandline: apt install sshfs
Commandline: apt install dos2unix
Commandline: apt-get install -y device-tree-compiler
Commandline: apt install python3-serial
Commandline: apt install screen
```

Login shell was changed to zsh

### UDEV Rules

`/etc/udev/99-ifboard.rules`

```udev
# IFBoard udev rule
SUBSYSTEM=="tty", ATTRS{idVendor}=="239a", ATTRS{idProduct}=="800d", SYMLINK+="ifboard", OWNER="root", MODE="0666"
```
