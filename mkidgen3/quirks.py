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
    def post_configure(cls):
        if cls.interrupt_mangled:
            cls.do_interrupt_mangled()
