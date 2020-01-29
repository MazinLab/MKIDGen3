from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from hlsinputgen_dds import test_data


def propagate(v):
    return v[1] > 0

def duplicate(v):
    return v[1] > 1


def condense_down(in_left: list):
        nrow = len(in_left)
        out_right = [(None, 0)] * nrow
        row_free = [False] * nrow

        # Row N-1
        if propagate(in_left[nrow - 1]):
            row_free[nrow - 1] = False
            last_consecutive_unused_ndx = nrow - 2
            out_right[nrow - 1] = in_left[nrow - 1]
        else:
            out_right[nrow - 1] = in_left[nrow - 1]  # propagating only for visualization
            last_consecutive_unused_ndx = nrow - 1  # just assume all rows are free
            row_free[nrow - 1] = True

        # Rows 0 to n-2
        for i in range(nrow - 2, -1, -1):
            if row_free[i + 1]: # row below free
                if propagate(in_left[i]):
                    last_consecutive_unused_ndx = i
                    out_right[i + 1] = in_left[i]
                else:
                    assert last_consecutive_unused_ndx > i
                    out_right[i + 1] = in_left[i]  # propagating only for visualization
                row_free[i] = True

            else:
                if propagate(in_left[i]):
                    out_right[i] = in_left[i]
                    row_free[i] = False
                    last_consecutive_unused_ndx = i-1  # assume row before free for now
                else:
                    assert last_consecutive_unused_ndx == i  # no change
                    out_right[i] = in_left[i]  # propagating only for visualization
                    row_free[i] = True

        return out_right, last_consecutive_unused_ndx


def condense_and_delay(in_left: list, state: list, buffer: list):
    condensed, _ = condense_down(state)
    count = len([v for v in condensed if v[1]])
    if count < 16:
        if buffer:
            # prepend buffer
            condensed[0] = buffer.pop(0)
        elif propagate(condensed[-1]):
            # Take the bottom value and put it in the buffer
            buffer.append(condensed[-1])

            # need to to this as part of the condensation in HLS
            condensed[1:] = condensed[:-1]
            condensed[0] = (None, 0)
    elif count == 16:
        if buffer:
            # if there is data in the buffer it must go out first
            condensed.insert(0, buffer.pop(0))
            buffer.append(condensed.pop())
    state[:] = in_left[:]
    return condensed


def condense_and_delay_fast(in_left: list, state: list, buffer: list):
    condensed, last_consecutive_unused_ndx = condense_down(state)
    count = len([v for v in condensed if v[1]])
    if count < 16:
        if buffer:
            # prepend buffer
            condensed[last_consecutive_unused_ndx] = buffer.pop(0)
        elif propagate(condensed[-1]):
            # Take the bottom value and put it in the buffer
            buffer.append(condensed[-1])

            # need to to this as part of the condensation in HLS
            condensed[1:] = condensed[:-1]
            condensed[0] = (None, 0)
    elif count == 16:
        if buffer:
            # if there is data in the buffer it must go out first
            condensed.insert(0, buffer.pop(0))
            buffer.append(condensed.pop())
    state[:] = in_left[:]
    return condensed



def condense_column_v2(in_left: list, fifo_out=None, fifo_in=None):
    #shift non-zero rows up by one if row above is free

    nrow = len(in_left)
    out_right = [(None, 0)]*nrow
    row_free = [False] * nrow

    #Row 0
    if fifo_out is not None:
        if propagate(in_left[0]):
            fifo_out.append(in_left[0])
        row_free[0] = True
    else:
        row_free[0] = not propagate(in_left[0])
        out_right[0] = in_left[0]

    # Rows 1 to n-2
    for i in range(1, nrow-1):
        if row_free[i-1]:
            out_right[i-1] = in_left[i] if propagate(in_left[i]) else in_left[i]
            row_free[i] = True
        else:
            out_right[i] = in_left[i] if propagate(in_left[i]) else in_left[i]
            row_free[i] = not propagate(in_left[i])

    # Last row
    i = nrow-1
    if row_free[i - 1]:
        out_right[i - 1] = in_left[i] if propagate(in_left[i]) else in_left[i]
        row_free[i] = not bool(fifo_in)  # Not actually used
        out_right[i] = fifo_in.pop(0) if fifo_in else (None, 0)
    else:
        print("SHOULD NOT BE POSSIBLE DUE TO FIFO OUT")
        raise RuntimeError("Impossible code path")
        out_right[i] = in_left[i] if propagate(in_left[i]) else in_left[i]
        row_free[i] = not propagate(in_left[i])

    return out_right


def condense_column(in_left: list):
    #shift non-zero rows up by one if row above is free

    nrow = len(in_left)
    out_right = [(None, 0)]*nrow
    row_free = [False] * nrow

    #Row 0
    if propagate(in_left[0]):
        row_free[0] = False
        out_right[0] = in_left[0]
    else:
        row_free[0] = True
        out_right[0] = (None,0)

    # Rows 1 to n-1
    for i in range(1, nrow):
        if row_free[i-1]:
            out_right[i-1] = in_left[i] if propagate(in_left[i]) else (None,0)
            row_free[i] = True
        else:
            out_right[i] = in_left[i] if propagate(in_left[i]) else (None,0)
            row_free[i] = not propagate(in_left[i])

    return out_right


def process_column_v2(in_left: list, state: list, fifoin: list, fifoout: list):
    nrow = len(in_left)
    out_right = [(None, 0)]*nrow
    newstate = [(None, 0)]*nrow
    shift_down = [False]*nrow

    # Row 0
    fifov = fifoin.pop(0) if fifoin else (None, 0)
    if not propagate(fifov):
        # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
        out_right[0] = (state[0][0], 1) if propagate(state[0]) else (None, 0)
        # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
        out_right[1] = (state[0][0], state[0][1] - 1) if duplicate(state[0]) else (None, 0)
        shift_down[0] = duplicate(state[0])
    else:
        out_right[0] = fifov
        # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
        out_right[1] = state[0] if propagate(state[0]) else (None, 0)
        shift_down[0] = propagate(state[0])
    newstate[0] = in_left[0]

    # Rows 1 to n-2
    for i in range(1, nrow-1):
        if not shift_down[i-1]:
            # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
            out_right[i] = (state[i][0], 1) if propagate(state[i]) else (None, 0)
            # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
            out_right[i+1] = (state[i][0], state[i][1]-1) if duplicate(state[i]) else (None, 0)
            shift_down[i] = duplicate(state[i])
        else:
            # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
            shift_down[i] = propagate(state[i])
            out_right[i + 1] = state[i] if propagate(state[i]) else (None, 0)
        newstate[i] = in_left[i]

    # Row N-1
    i=nrow-1
    if not shift_down[i-1]:
        # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
        out_right[i] = (state[i][0], 1) if propagate(state[i]) else (None, 0)
        # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
        if duplicate(state[i]):
            fifoout.append((state[i][0], state[i][1] - 1))
        shift_down[i] = duplicate(state[i])  # Not actually used
    else:
        # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
        shift_down[i] = propagate(state[i])  # Not actually used
        if propagate(state[i]):
            fifoout.append(state[i])

    newstate[i] = in_left[i]

    for i in range(len(newstate)):
        state[:] = newstate[:]

    return out_right


def process_column_v2_wcondense(in_left: list, state: list, fifoin: list, fifoout: list):
    nrow = len(in_left)
    out_right = [(None, 0)]*nrow
    newstate = [(None, 0)]*nrow
    shift_down = [False]*nrow

    # Row 0
    fifov = fifoin.pop(0) if fifoin else (None, 0)
    if not propagate(fifov):
        # Two cases (shift down does/doesn't affect row i+1 propagation)
        if propagate(state[1]):
            # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
            out_right[0] = (state[0][0], 1) if propagate(state[0]) else (None, 0)
            # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
            out_right[1] = (state[0][0], state[0][1] - 1) if duplicate(state[0]) else (None, 0)
            shift_down[0] = duplicate(state[0])
        else:
            # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
            out_right[0] = (state[0][0], 1) if duplicate(state[0]) else (None, 0)
            # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
            out_right[1] = (state[0][0], state[0][1] - 1) if duplicate(state[0]) else ((state[0][0], 1) if propagate(state[0]) else (None, 0))
            shift_down[0] = True
    else:
        out_right[0] = fifov
        # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
        out_right[1] = state[0] if propagate(state[0]) else (None, 0)
        shift_down[0] = propagate(state[0])
    newstate[0] = in_left[0]

    # Rows 1 to n-2
    for i in range(1, nrow-1):
        if not shift_down[i-1]:
            if propagate(state[i+1]):
                # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
                out_right[i] = (state[i][0], 1) if propagate(state[i]) else (None, 0)
                # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
                out_right[i+1] = (state[i][0], state[i][1]-1) if duplicate(state[i]) else (None, 0)
                shift_down[i] = duplicate(state[i])
            else:
                # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
                out_right[i] = (state[i][0], 1) if duplicate(state[i]) else (None, 0)
                # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
                out_right[i+1] = (state[i][0], state[i][1] - 1) if duplicate(state[i]) else (
                    (state[i][0], 1) if propagate(state[i]) else (None, 0))
                shift_down[i] = True
        else:
            # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
            shift_down[i] = propagate(state[i])
            out_right[i + 1] = state[i] if propagate(state[i]) else (None, 0)
        newstate[i] = in_left[i]

    # Row N-1
    i=nrow-1
    if not shift_down[i-1]:
        # state[i] moves right with count 1 if it needs propagating and state[i-1] didn't shift down
        out_right[i] = (state[i][0], 1) if propagate(state[i]) else (None, 0)
        # state[i] moves down right with count n-1 if it needs duplicating and state[i-1] didn't shift down
        if duplicate(state[i]):
            fifoout.append((state[i][0], state[i][1] - 1))
        shift_down[i] = duplicate(state[i])  # Not actually used
    else:
        # state[i] moves down right with count n if it needs propagating and state[i-1] shifted down
        shift_down[i] = propagate(state[i])  # Not actually used
        if propagate(state[i]):
            fifoout.append(state[i])

    newstate[i] = in_left[i]

    for i in range(len(newstate)):
        state[:] = newstate[:]

    return out_right


class Crossbar:
    def __init__(self, size=16, condenser_stages=48, initial_stages=1):
        self.size = size
        self.stages = []
        self.fifos = []
        self.out = []
        self.max_fifo_lengths = []
        self.condense_stages = [[(None, 0)] * self.size for i in range(condenser_stages)]
        self.condense_fifos = [[] for i in range(condenser_stages)]
        for i in range(initial_stages):
            self.new_stage()

    def new_stage(self):
        self.stages.append([(None,0)] * self.size)
        self.fifos.append([])
        self.max_fifo_lengths.append(0)

    def run_crossbar(self, input):

        right = input

        # Duplicate
        i = 0
        while i < len(self.stages):

            right = process_column_v2(right, self.stages[i], self.fifos[i], self.fifos[i])
            i += 1
            if i == len(self.stages):
                if any([v[1] > 1 for v in right]):
                    self.new_stage()
                    print(f'Adding stage {i}. Right: {right}')
                else:
                    print(f'Attained output at stage {i}. Out: {right}')

            # Track max fifo length
            self.max_fifo_lengths = [max(len(fifo), ml) for fifo, ml in zip(self.fifos, self.max_fifo_lengths)]

        # Condense
        i=0
        while i < len(self.condense_stages):
            right = condense_and_delay_fast(right, self.condense_stages[i], self.condense_fifos[i])
            i += 1

        self.out.append(right)
        return self.out[-1]

    def draw(self, input, output):
        ncol = len(self.stages)

        plt.rcParams.update({'font.size': 8})
        for r, i in enumerate(input):
            plt.text(-1, r, '{}:{}'.format(*i) if i[0] is not None else '-')

        plt.axvline(0)

        for c, (stage, fifo) in enumerate(zip(self.stages, self.fifos)):
            for r, state in enumerate(stage):
                plt.text(c, r, '{}:{}'.format(*state) if state[0] is not None else '-')
            for r, f in enumerate(fifo[::-1]):
                plt.text(c, -(r + 1), '{}:{}'.format(*f))

        plt.axvline(ncol, color='k')

        for c, (stage, fifo) in enumerate(zip(self.condense_stages, self.condense_fifos)):
            for r, state in enumerate(stage):
                plt.text(c+ncol, r, '{}:{}'.format(*state) if state[0] is not None else '-')
            for r, f in enumerate(fifo[::-1]):
                plt.text(c+ncol, -(r + 1), '{}:{}'.format(*f))

        plt.axvline(ncol+len(self.condense_stages))

        for r, o in enumerate(output):
            plt.text(ncol+len(self.condense_stages), r, '{}:{}'.format(*o) if o[0] is not None else '-')

        plt.axhline(0, color='k')
        plt.xlim(-2, ncol+len(self.condense_stages)+1)
        plt.ylim(-2, 17)
        plt.rcParams.update({'font.size': 10})


if __name__ == '__main__':
    counts = test_data().bincounts
    print('Running wth {} bins, max of {} per bin'.format(len(counts), max(counts)))
    res_groups = [list(zip(range(i, j), counts[i:j])) for i, j in zip(range(0,4096,16), range(16,4097,16))]
    series_compare = [(res[0],1) for res_group in res_groups for res in res_group for i in range(res[1])]

run=True

if run:
    cb = Crossbar(16, initial_stages=4, condenser_stages=20)

    # i+=1
    # #if i=4 activate breakpoint
    # g,n=res_groups[i:i+2]
    # cb.run_crossbar(g)
    # plt.figure()
    # cb.draw(n, cb.out[-1])
    # plt.title('Cycle {}'.format(i))
    # #



    with PdfPages('/Users/one/Desktop/crossbar.pdf') as pdf:
        plt.figure()
        plt.clf()
        prow=1
        plt.subplot(2,1,prow)
        cb.draw(res_groups[0], cb.condense_stages[0])
        plt.title('Cycle {}'.format(0))
        plt.ylabel('Row')
        # plt.xlabel('Stage')
        # plt.tight_layout()
        # plt.gcf().set_size_inches([10, 3.5])
        # pdf.savefig(plt.gcf(), orientation='portrait')
        # plt.close(plt.gcf())

        for in_cycle, g in enumerate(res_groups[:50]):

            cb.run_crossbar(g)

            if 50>in_cycle>-1:  # Visually validated through 29
                if prow==2:
                    plt.xlabel('Stage')
                    plt.tight_layout()
                    plt.gcf().set_size_inches([11, 8.5])
                    pdf.savefig(plt.gcf(), orientation='portrait')
                    plt.close(plt.gcf())
                    plt.figure()
                    # plt.subplots_adjust(.07,.12,.98,.93,.20,.23)
                    plt.clf()
                    prow=1
                else:
                    prow+=1
                plt.subplot(2, 1, prow)
                cb.draw(res_groups[in_cycle+1], cb.out[-1])
                plt.title('Cycle {}'.format(in_cycle+1))
                plt.ylabel('Row')


    series = [o for l in cb.out for o in l if o[1] > 0]
    print(series == series_compare[:len(series)])
    # series_compare = [o for l in res_groups for o in l if o[1]>0][:10]
    #


