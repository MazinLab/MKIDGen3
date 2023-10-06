import numpy as np


class PixelMap:
    """ A stub for the mapping of feedline and channel to pixel position

    map=PixelMap()
    map[fl, channel] -> array of xy for specified fl, channel
    unplaced channels shall be non-finite
    """
    def __init__(self, x):
        """
        Args:
            x: an n_feedline x n_channels x 2 array where the last dimension is x and y
            x[1,300] would then be the x,y of feedline 1 channel 300
            finite elements must be unique (np.unique(x[np.isfinite(x)]).size==np.isfinite(x).sum())
        """
        assert np.unique(x[np.isfinite(x)]).size==np.isfinite(x).sum()
        self._map = x

    def __getitem__(self, fl, channel):
        return self._map[fl, channel]


def example_map(n_feedlines, n_channels)->PixelMap:
    """

    Args:
        n_feedlines: The total number of feed lines present
        n_channels:

    Returns: An example map

    """
    return PixelMap(np.arange(n_feedlines*n_channels, dtype=int).reshape((n_feedlines*n_channels)))
