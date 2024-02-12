from .captures import *
from zmq import *
from .pixelmap import PixelMap
from mkidgen3.mkidpynq import unpack_photons

MAX_PHOTON_PER_CR_SEND = 5000
INSTRUMENT_PHOTON_TYPE = (('time', 'u32'), ('x', 'u32'), ('y', 'u32'), ('phase', 'u16'))

class Gen3TableSaver:
    def __init__(self, file, size_hint=None):
        self._file=file

    def grow(self, photons):
        pass


class Gen3LiveImage:
    def __init__(self, destination, exposure_time, photon_rate=True):
        self.exposure_time = exposure_time
        self.photon_rate = photon_rate

    def add_photons(self, photons):
        pass


class PhotonAggregator:
    """ A stub for combining photon capture data from multiple feedlines into a cohesive entity suitable for further
    use. Photon CaptureJobs result in photon data being published by the various feedline readout servers under the
    associated capture ID. To gather data from a full array we need to connect to each stream receive the data,
    uncompress it, convert the timestamps to full unix times, convert the feedline channels to full readout channels (
    FL+FLchannel), and then, depending, do things like convert to a rate, array image, save send on, or so forth

    Captures can die or be killed
    """
    def __init__(self, time_offset, jobs: List[CaptureJob] or None, pixel_map:PixelMap):
        self._time_offset = time_offset
        self._new_jobs = jobs
        self._map = pixel_map
        self._dead_jobs = []
        self._jobs = []
        self._hdf_saver = None

    def add_capture(self, jobs: List[CaptureJob]):
        self._new_jobs.extend(jobs)

    def run(self):
        poller = zmq.Poller()
        poller.register(self._pipe[1], flags=zmq.POLLIN)

        getLogger(__name__).debug(f'Listening for data')
        self._pipe[1].send(b'')

        while True:

            for job in list(self._new_jobs):
                assert not job.datasink.is_alive()
                poller.register(job.datasink.establish())  #TODO this isn't thread safe if the di
                self._jobs.append(job)
                self._new_jobs.remove(job)

            avail = dict(poller.poll())
            if self._pipe[1] in avail:
                getLogger(__name__).debug(f'Received shutdown order, terminating {self}')
                break

            nnew = 0
            photons_buf = np.recarray(len(avail)*MAX_PHOTON_PER_CR_SEND, dtype=INSTRUMENT_PHOTON_TYPE)

            for job in (j for j in list(self._jobs) if j.datasink.socket in avail):
                raw_phot_data = job.datasink.receive()

                if raw_phot_data is None:
                    getLogger(__name__).debug(f'Photon data stream over for {job}')
                    poller.unregister(job.datasing.socket)
                    self._jobs.remove(job)
                    self._dead_jobs.append(job)
                    continue

                sl_out = slice(nnew, nnew+raw_phot_data.size)
                photons_buf['time'][sl_out] = raw_phot_data['time'] + self._time_offset
                photons_buf['phase'][sl_out] = raw_phot_data['phase']
                xy = self._map[job.fl_ndx, raw_phot_data['id']]  #todo
                photons_buf['x'][sl_out] = xy[0]
                photons_buf['y'][sl_out] = xy[1]
                nnew+=raw_phot_data.size

            if self._live_image is not None:
                self._image_image.add_photons(photons_buf[:nnew])

            if self._hdf_saver is not None:
                self._hdf_saver.grow(photons_buf[:nnew])

            if self._republisher:
                self._republisher.send(photons_buf[:nnew])

        for job in self._jobs:
            job.datasink.destablish()


def make_image(map, photons, timestep_us=None, rate=False):
    """ Build an image out of the current photon data, specify a timestep (in us) to make a timecube"""

    first, last = photons['time'][[0, -1]]
    duration_s = (last - first) / 1e6
    if first >= last:
        raise ValueError('Last photon time is not after the first!')

    if timestep_us:
        timebins = (np.arange(first, last + timestep_us, timestep_us, dtype=type(timestep_us)),)
    else:
        timebins = tuple()
    imbins = map.map_bins

    val = map.map[photons['id']]
    val = (val,) if map.map.ndim == 1 else (val[0], val[1])
    coord = val + (photons['time'],) if timestep_us else val

    hist, _ = np.histogramdd(coord, bins=imbins + timebins, density=rate)
    if rate:
        hist *= photons.size
        hist /= duration_s if timestep_us is None else 1e6

    return hist

from mkidgen3.mkidpynq import PHOTON_DTYPE
class PhotonAccumulator:
    def __init__(self, max_ram=100, pixelmap: PixelMap = None, n_channels=None):
        """
        A tool to gather photons together without running out of ram and ease access
        to timeseries, images, and other helpful metrics.

        max_ram: how much space to allow for photon storage
        pixelmap: and optional channel id to x or xy map, should be a nChan x 1 or 2 array positions
        n_channels: optional shortcut if pixelmap not specified
        """
        assert bool(n_channels) ^ bool(pixelmap), 'Must specify n_channels xor pixelmap'
        self.ram_limit = max_ram  # MiB
        self.map = pixelmap if pixelmap else PixelMap(np.arange(n_channels, dtype=int), n_channels)
        self._data = np.zeros(self.ram_limit * 1024 ** 2 // PHOTON_DTYPE.itemsize, dtype=PHOTON_DTYPE)
        self.n = 0
        self.accumulations = 0

    def __getitem__(self, n):
        """Get photons for channel n"""
        return self.data[self.data['id'] == n]

    def accumulate(self, buffer, n=None):
        """Add some packed photons to the pot"""
        self.accumulations += 1
        if n is None:
            d = np.array(buffer)
            n = buffer.size
        else:
            d = np.array(buffer[:n])

        drop = max(n - (self._data.size - self.n), 0)
        if drop:
            getLogger(__name__).debug(f'Memory used up, dropping {drop} old photons.')
            self._data[:self._data.size - drop] = self._data[drop:]
        unpack_photons(buffer[:n], out=self._data, n=self.n)
        self.n += n - drop

    @property
    def data(self):
        return self._data[:self.n]

    def image(self, timestep_us=None, rate=False):
        """ Build an image out of the current photon data, specify a timestep (in us) to make a timecube"""
        return make_image(self.map, self.data[:self.n], timestep_us=timestep_us, rate=rate)



from mkidgen3.server import pixelmap
if __name__=='__main__':

    h5file = 'test.h5'
    live_image = ('live.tcp', 'live.bmp')
    repub = ('photons.tcp')

    #Need to manually spin up feedline servers

    N_FEEDLINES, N_RESONATORS = 2, 2048
    frs1 = FRSClient(url1)
    frs2 = FRSClient(url2)
    cfg = FeedlineConfig()

    servers = [frs1, frs2]
    map = pixelmap.example_map(N_FEEDLINES, N_RESONATORS)
    jobs = [CaptureJob(CaptureRequest(.5, 'photons', cfg, frs),
                       submit=(False, False)) for frs in servers]

    aggregator = PhotonAggregator(int(time.time()), jobs, map)
