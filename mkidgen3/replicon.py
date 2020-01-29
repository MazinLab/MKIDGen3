from hlsinputgen_dds import test_data
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class Replibin:
    def __init__(self,iq=0,n=0):
        self.n=n
        self.iq=iq

    def __repr__(self):
        return f'Replibin({self.iq},{self.n})'

    def __str__(self):
        return f'{self.iq}:{self.n}'


def process_stateless_column(input: list):
    above_shifted_down = False
    above_free = False
    out = [Replibin() for i in range(len(input)+1)]

    for i in range(0, len(input)):
        # //Higher indices are down!
        this_out = i
        below_out = i + 1
        above_out = i -1

        # //Move the sample down if we must
        if above_shifted_down:
            above_free = False
            if input[i].n > 0:
                out[below_out] = input[i]
            else:
                above_shifted_down = False
        else:
            # //Replicate/shift the sample up if possible and necessary
            if above_free and input[i].n > 0:
                out[above_out].iq = input[i].iq
                out[above_out].n = 1
            elif above_free:
                out[above_out].n = 0
                out[above_out].iq = 0
            # //Propagate the sample if needed
            if input[i].n > 1 and above_free:
                out[this_out].iq = input[i].iq
                out[this_out].n = 1
                above_free = False
            elif input[i].n > 0 and not above_free:
                out[this_out].iq = input[i].iq
                out[this_out].n = 1
                above_free = False
            else:
                above_free = True
            # //Replicate the sample down if needed
            if input[i].n > 2 and above_free:
                out[below_out].iq = input[i].iq
                out[below_out].n = input[i].n - 2
                above_shifted_down = True
            elif input[i].n > 1 and not above_free:
                out[below_out].iq = input[i].iq
                out[below_out].n = input[i].n - 1
                above_shifted_down = True
            else:
                above_shifted_down = False
    return out


class Crossbar:
    def __init__(self, size=16,initial_stages=1):
        self.size = size
        self.interstage = []
        self.out = []
        for i in range(initial_stages+1):
            self.new_stage()

    def new_stage(self):
        self.interstage.append([Replibin() for i in range(self.size+len(self.interstage))])

    def run_crossbar(self, input):

        i = 0
        self.interstage[0] = right = input
        # Duplicate
        while i < len(self.interstage)-1:
            right = process_stateless_column(right)
            self.interstage[i+1] = right
            i += 1

            done = all([v.n <= 1 for v in right])

            if done:
                print(f'Attained output at stage {i}. Out: {right}')

            if i == len(self.interstage)-1:
                if not done:
                    self.new_stage()
                    print(f'Adding stage {i}. Right: {right}')


        return self.interstage[-1]

    def draw(self):

        plt.rcParams.update({'font.size': 8})
        for c, stage in enumerate(self.interstage):
            for r, state in enumerate(stage):
                plt.text(c, r-c, f'{state}' if state.n>0 else '-')

        plt.axvline(.95, color='k')
        plt.axvline(len(self.interstage)-1.05, color='k')
        plt.xlim(-2, len(self.interstage)+1)
        plt.ylim(-1, len(self.interstage)+self.size)
        plt.rcParams.update({'font.size': 10})


if __name__ == '__main__':
    td = test_data()
    counts = td.bincount
    print('Running wth {} bins, max of {} per bin'.format(len(counts), max(counts)))

    res_groups = [list(zip(range(i, j), counts[i:j])) for i, j in zip(range(0, 4096, 16), range(16, 4097, 16))]
    series_compare = [(res[0], 1) for res_group in res_groups for res in res_group for i in range(res[1])]

    np.histogram(td.freqs, bins=np.linspace(td.freqs.min(), td.freqs.max(), 17))

    res_groups_256MHZ = [list(zip(range(i, j), counts[i:j])) for i, j in zip(range(0, 4096, 17), range(16, 4097, 16))]

    cb = Crossbar(16, initial_stages=4)

    print('Resonators per lane taken in groups of 16: {}'.format(np.array(res_groups).sum(0)[:,1]))
    print('Resonators per lane taken in groups of 16 for each 256MHz window: {}'.format(np.histogram(td.freqs, bins=np.linspace(td.freqs.min(), td.freqs.max(), 17))[0]))

    run = False


    # i+=1
    # #if i=4 activate breakpoint
    # g,n=res_groups[i:i+2]
    # cb.run_crossbar(g)
    # plt.figure()
    # cb.draw(n, cb.out[-1])
    # plt.title('Cycle {}'.format(i))
    # #

    # with PdfPages('/Users/one/Desktop/crossbar.pdf') as pdf:
    plt.figure()
    plt.clf()
    prow = 1
    plt.subplot(2, 1, prow)
    cb.draw()
    plt.title('Cycle 0')
    plt.ylabel('Row')
    # plt.xlabel('Stage')
    # plt.tight_layout()
    # plt.gcf().set_size_inches([10, 3.5])
    # pdf.savefig(plt.gcf(), orientation='portrait')
    # plt.close(plt.gcf())
if run:

    for in_cycle, g in enumerate(res_groups[:5]):
        g = [Replibin(*x) for x in g]
        cb.run_crossbar(g)

        if 50 > in_cycle > -1:  # Visually validated through 29
            if prow == 2:
                plt.xlabel('Stage')
                plt.tight_layout()
                plt.gcf().set_size_inches([11, 8.5])
                # pdf.savefig(plt.gcf(), orientation='portrait')
                # plt.close(plt.gcf())
                plt.figure()
                # plt.subplots_adjust(.07,.12,.98,.93,.20,.23)
                plt.clf()
                prow = 1
            else:
                prow += 1
            plt.subplot(2, 1, prow)
            cb.draw()
            plt.title('Cycle {}'.format(in_cycle + 1))
            plt.ylabel('Row')

    series = [o for l in cb.out for o in l if o[1] > 0]
    print(series == series_compare[:len(series)])
    # series_compare = [o for l in res_groups for o in l if o[1]>0][:10]
    #
