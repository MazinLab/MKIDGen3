from pynq import DefaultIP


class IQGen(DefaultIP):
    bindto = ['mazinlab:mkidgen3:iq_gen:0.5']

    def __init__(self, description):
        super().__init__(description=description)

    def generate(self, n=2**20-1):
        """ Tell the block to generate n multiples of 4096 of samples. """
        self.register_map.max = n
        self.register_map.run = True

    def stop(self):
        self.register_map.run=False
