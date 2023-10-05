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
    @classmethod
    @property
    def interrupt_mangled(cls):
        """Work around an inconsistency in name mangling between the HWH parser and the pynq
        interrupt logic

        Report upstream to Xilinx
        """
        import mkidgen3

        for n in mkidgen3._gen3_overlay.ip_dict.keys():
            if "axi_intc" in n and "/" in n:
                return True
        return False

    @classmethod
    def do_interrupt_mangled(cls):
        import mkidgen3

        parser_intcs = mkidgen3._gen3_overlay.device.parser.interrupt_controllers
        top_intcs = mkidgen3._gen3_overlay.interrupt_controllers
        for n in mkidgen3._gen3_overlay.ip_dict.keys():
            if "axi_intc" in n and "/" in n:
                parser_intcs[n] = parser_intcs[n.replace("/", "_")]
                top_intcs[n] = top_intcs[n.replace("/", "_")]

    @classmethod
    @property
    def threepart_ddc(cls):
        """The three part DDC uses an AXI BRAM controller which is picked up as a memory
        instead of an IP by PYNQ3, this instantiates the 3 part ddc driver so that to the
        user it still looks like an IP
        """
        import mkidgen3

        for n in mkidgen3._gen3_overlay.mem_dict.keys():
            if "reschan" in n and "axi_bram_ctrl" in n:
                return True
        return False

    @classmethod
    def do_threepart_ddc(cls):
        import mkidgen3

        for n in mkidgen3._gen3_overlay.mem_dict.keys():
            if "reschan" in n and "axi_bram_ctrl" in n:
                from mkidgen3.drivers.ddc import ThreepartDDC

                num = int(n.split("_")[-1])
                memname = "".join(n.split("/"))
                mmio = getattr(mkidgen3._gen3_overlay, memname).mmio
                hier = mkidgen3._gen3_overlay
                for h in n.split("/")[:-1]:
                    hier = getattr(hier, h)
                setattr(hier, "ddccontrol_{:d}".format(num), ThreepartDDC(mmio))

    @classmethod
    def post_configure(cls):
        if cls.interrupt_mangled:
            cls.do_interrupt_mangled()
        if cls.threepart_ddc:
            cls.do_threepart_ddc()
