import numpy as np

def est_loop_centers(iq):
    """
    Finds the (I,Q) centers of the loops via percentile math
    iq - array[n_loops, n_samples]
    returns centers[iq.shape[0]]

    see mkidgen2.roach2controls.fitLoopCenters for history
    """
    ictr = (np.percentile(iq.real, 95, axis=1) + np.percentile(iq.real, 5, axis=1)) / 2
    qctr = (np.percentile(iq.imag, 95, axis=1) + np.percentile(iq.imag, 5, axis=1)) / 2

    return ictr + qctr * 1j
