import mkidgen3 as g3
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # plt.switch_backend('QT5cairo')
    # plt.plot([1,2,3])
    # plt.show(block=False)

    ol = g3.configure('/home/xilinx/jupyter_notebooks/Power_Test/bxbfft_unpipelined.bit', clocks=True, external_10mhz=True, ignore_version=True)
    print(ol.is_loaded())



    print('hi')
    print('bye')