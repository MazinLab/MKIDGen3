Board Setup
=============

Hardware Setup:
*******************
Required Items:
* RFSoC
* IF Board
* RF Cables
* Power Cables


Download Overlay, connect and program IF board
**************************************************
.. code-block:: python

    """This example demonstrates how to download the overlay and interact with the FPGA through Python.

    """

    import mkidgen3 as g3
    bitstream='/home/xilinx/bit/cordic_16_15_fir_22_0.bit'
    ol = g3.configure(bitstream, ignore_version=True, clocks=True, external_10mhz=False, download=True)
