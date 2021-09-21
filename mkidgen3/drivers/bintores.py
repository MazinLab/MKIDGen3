class BinToResIP(DefaultIP):
    resmap_addr = 0x1000

    def __init__(self, description):
        """
        0x1fff : Memory 'data_V' (256 * 96b)
        Word 4n   : bit [31:0] - data_V[n][31: 0]
        Word 4n+1 : bit [31:0] - data_V[n][63:32]
        Word 4n+2 : bit [31:0] - data_V[n][95:64]
        Word 4n+3 : bit [31:0] - reserved
        """
        super().__init__(description=description)

    bindto = ['MazinLab:mkidgen3:bin_to_res:0.6']

    @staticmethod
    def _checkgroup(group_ndx):
        if group_ndx < 0 or group_ndx > 255:
            raise ValueError('group_ndx must be in [0,255]')

    def read_group(self, group_ndx):
        self._checkgroup(group_ndx)
        g = 0
        vals = [self.read(self.resmap_addr + 16 * group_ndx + 4 * i) for i in range(3)]
        for i, v in enumerate(vals):
            # print(format(v,'032b'))
            g |= v << (32 * i)
        # print('H-'+format(g,'096b')+'-L')
        return [((g >> (12 * j)) & 0xfff) for j in range(8)]

    def write_group(self, group_ndx, group):
        self._checkgroup(group_ndx)
        if len(group) != 8:
            raise ValueError('len(group)!=8')
        bits = 0
        for i, g in enumerate(group):
            bits |= (int(g) & 0xfff) << (12 * i)
        data = bits.to_bytes(12, 'little', signed=False)
        self.write(self.resmap_addr + 16 * group_ndx, data)

    def bin(self, res):
        """ The mapping for resonator i is 12 bits and will require reading 1 or 2 32 bit word
        n=i//8 j=(i%8)*12//32
        """
        return self.read_group(res // 8)[res % 8]

    @property
    def bins(self):
        return [v for g in range(256) for v in self.read_group(g)]

    @bins.setter
    def bins(self, bins):
        if len(bins) != 2048:
            raise ValueError('len(bins)!=2048')
        if min(bins) < 0 or max(bins) > 4095:
            raise ValueError('Bin values must be in [0,4095]')
        for i in range(256):
            self.write_group(i, bins[i * 8:i * 8 + 8])