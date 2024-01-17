import pynq


class Quirk:
    """A class to unify working around hardware/pynq quirks in different versions

    Indicate quirk presence with .quirkname properties, provide fixes with .do_quirkname(cls, ..)
    classmethods, and indicate fix completion with .done_quirkname class properties

    For groups of quirks use .pre_eventname() and .post_eventname() classmethods to manage them
    """


class MTS(Quirk):
    """Unclear why this is required but it has been in every bistream I've used, report upstream"""

    double_sync = True


class Overlay(Quirk):

    def __init__(self, overlay:pynq.Overlay):
        self._ol = overlay

    @property
    def interrupt_mangled(self):
        """Work around an inconsistency in name mangling between the HWH parser and the pynq
        interrupt logic

        Report upstream to Xilinx
        """
        for n in self._ol.ip_dict.keys():
            if "axi_intc" in n and "/" in n:
                return True
        return False

    def do_interrupt_mangled(self):
        parser_intcs = self._ol.device.parser.interrupt_controllers
        top_intcs = self._ol.interrupt_controllers
        for n in self._ol.ip_dict.keys():
            if "axi_intc" in n and "/" in n:
                parser_intcs[n] = parser_intcs[n.replace("/", "_")]
                top_intcs[n] = top_intcs[n.replace("/", "_")]

    @property
    def threepart_ddc(self):
        """The three part DDC uses an AXI BRAM controller which is picked up as a memory
        instead of an IP by PYNQ3, this instantiates the 3 part ddc driver so that to the
        user it still looks like an IP
        """
        for n in self._ol.mem_dict.keys():
            if "reschan" in n and "axi_bram_ctrl" in n:
                return True
        return False

    def do_threepart_ddc(self):
        import mkidgen3

        for n in self._ol.mem_dict.keys():
            if "reschan" in n and "axi_bram_ctrl" in n:
                from mkidgen3.drivers.ddc import ThreepartDDC

                num = int(n.split("_")[-1])
                memname = "".join(n.split("/"))
                mmio = getattr(self._ol, memname).mmio
                hier = self._ol
                for h in n.split("/")[:-1]:
                    hier = getattr(hier, h)
                setattr(hier, "ddccontrol_{:d}".format(num), ThreepartDDC(mmio))

    def post_configure(self):
        if not hasattr(self._ol, 'dac_table') and hasattr(self._ol, 'dactable'):
            self._ol.dac_table = self._ol.dactable
        if self.interrupt_mangled:
            self.do_interrupt_mangled()
        if self.threepart_ddc:
            self.do_threepart_ddc()
