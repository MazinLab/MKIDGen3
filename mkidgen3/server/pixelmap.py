import numpy as np

#
# class PixelMap:
#     """ A stub for the mapping of feedline and channel to pixel position
#
#     map=PixelMap()
#     map[fl, channel] -> array of xy for specified fl, channel
#     unplaced channels shall be non-finite
#     """
#     def __init__(self, x):
#         """
#         Args:
#             x: an n_feedline x n_channels x 2 array where the last dimension is x and y
#             x[1,300] would then be the x,y of feedline 1 channel 300
#             finite elements must be unique (np.unique(x[np.isfinite(x)]).size==np.isfinite(x).sum())
#         """
#         assert np.unique(x[np.isfinite(x)]).size==np.isfinite(x).sum()
#         self._map = x
#
#     def __getitem__(self, fl, channel):
#         return self._map[fl, channel]


class PixelMap:
    """ The attribute map_bins is a tuple of the pixel bin edges i.e. for histograms"""

    def __init__(self, map, nx, ny=0):
        """
        A Pixel to channel mapping

        nx: the number of x pixels
        ny: the number of y pixels, 0 if 1d
        map: is a 1 or 2d array channel id to pixel coordinate. 1d is for linear arrays or
        single feedlines. Will be coerced to integer.

        """
        self.map = map.astype(int)
        self.nx = max(nx, 1)
        self.ny = max(ny, 0)
        assert (ny > 0) == (map.ndim == 2)
        # TODO assertion that no pixel is mapped to more than one channel
        #  set(zip(self.map[:,0], self.map[:,1]))

        maxn = self.map.max(axis=-1)
        ok = maxn[0] < self.nx and maxn[1] < self.ny if ny else maxn < self.nx
        assert ok, "Mapped pixels out of bounds"

        if not ny:
            self.map_bins = (np.arange(self.nx + 1, dtype=int),)
        else:
            self.map_bins = np.arange(self.nx + 1, dtype=int), np.arange(self.ny + 1, dtype=int)


def example_map(n_feedlines, n_channels)->PixelMap:
    """

    Args:
        n_feedlines: The total number of feed lines present
        n_channels:

    Returns: An example map

    """
    return PixelMap(np.arange(n_feedlines*n_channels, dtype=int).reshape((n_feedlines, n_channels)),
                    n_feedlines, n_channels)
