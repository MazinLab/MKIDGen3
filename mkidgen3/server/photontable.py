import numpy as np
from logging import getLogger
import tables
# import os
# import time
import multiprocessing as mp
import functools

# from scipy.interpolate import InterpolatedUnivariateSpline
# from ruamel.yaml.comments import CommentedSeq
#
# import mkidcore.metadata
# from mkidcore.binfile.mkidbin import PhotonCType, PhotonNumpyType
# from mkidcore.corelog import getLogger
# from mkidcore.pixelflags import FlagSet
# import mkidcore.pixelflags as pixelflags
# from mkidcore.instruments import compute_wcs_ref_pixel
# import mkidpipeline.utils.memory as pipeline_ram
# from mkidcore.metadata import INSTRUMENT_KEY_MAP
# import SharedArray
#
# import tables
import tables.parameters
import tables.file
#
# import astropy.constants
# import astropy.units as u
# from astropy.io import fits
#
# import warnings


# These are little better than blind guesses and don't seem to impact performaace, but still need benchmarking
# tables.parameters.CHUNK_CACHE_SIZE = 2 * 1024 * 1024 * 1024
# tables.parameters.TABLE_MAX_SIZE = 2 * 1024 * 1024 * 1024  # default 1MB
# This governs the chunk cache that will store table data, if a row is
# 20 bytes (as is our present state)
# nslots = TABLE_MAX_SIZE / (chunksize * rowsize_bytes)
# so number of rows in the cache is ~ TABLE_MAX_SIZE/rowsize_bytes
# one 1s of 20k pix data @ 2500c/s is 0.95GB

# These are all used by the c code backing pytables and it isn't clear their importance yet
# tables.parameters.CHUNK_CACHE_NELMTS *= 10
# tables.parameters.SORTEDLR_MAX_SIZE = 1 *1024*1024*1024 # default 8MB
# tables.parameters.SORTED_MAX_SIZE = 1 *1024*1024*1024 # default 1MB
# tables.parameters.LIMBOUNDS_MAX_SIZE = 1*1024*1024*1024
# tables.parameters.SORTEDLR_MAX_SLOTS *= 10

# tables.parameters.CHUNK_CACHE_SIZE *=10
# tables.parameters.TABLE_MAX_SIZE *=10
# tables.parameters.CHUNK_CACHE_NELMTS *= 10
# tables.parameters.SORTEDLR_MAX_SIZE *=10
# tables.parameters.SORTED_MAX_SIZE *=10
# tables.parameters.LIMBOUNDS_MAX_SIZE *=10
# tables.parameters.SORTEDLR_MAX_SLOTS *= 10

# tables.parameters.CHUNK_CACHE_SIZE = 32*1024**2
# tables.parameters.MAX_BLOSC_THREADS = 8
# tables.parameters.MAX_NUMEXPR_THREADS = 8
# tables.parameters.IO_BUFFER_SIZE = 32*1024**2   # Don't exceed L3 cache size (16MiB)


class ThreadsafeFileRegistry(tables.file._FileRegistry):
    lock = mp.RLock()

    @property
    def handlers(self):
        return self._handlers.copy()

    def add(self, handler):
        with self.lock:
            return super().add(handler)

    def remove(self, handler):
        with self.lock:
            return super().remove(handler)

    def close_all(self):
        with self.lock:
            return super().close_all()


class ThreadsafeFile(tables.file.File):
    def __init__(self, *args, **kargs):
        with ThreadsafeFileRegistry.lock:
            super().__init__(*args, **kargs)

    def close(self):
        with ThreadsafeFileRegistry.lock:
            super().close()


@functools.wraps(tables.open_file)
def synchronized_open_file(*args, **kwargs):
    with ThreadsafeFileRegistry.lock:
        return tables.file._original_open_file(*args, **kwargs)


# monkey patch the tables package
tables.file._original_open_file = tables.file.open_file
tables.file.open_file = synchronized_open_file
tables.open_file = synchronized_open_file
tables.file._original_File = tables.File
tables.file.File = ThreadsafeFile
tables.File = ThreadsafeFile
tables.file._open_files = ThreadsafeFileRegistry()


class SharedTable(object):
    """multiprocessingsafe shared photon table, readonly!!!!"""

    def __init__(self, shape):
        self._shape = shape
        self._X = mp.RawArray(PhotonCType, int(np.prod(shape)))
        self.data = np.frombuffer(self._X, dtype=PhotonNumpyType).reshape(self._shape)

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop('data', None)
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = np.frombuffer(self._X, dtype=PhotonNumpyType).reshape(self._shape)


def load_shareable_photonlist(file, query=None):
    tic = time.time()
    f = Photontable(file)

    if query is not None:
        d = f.query(**query)
        table = SharedTable(d.shape)
        table.data = d
    else:
        table = SharedTable(f.photonTable.shape)
        f.photonTable.read(out=table.data)
    ram = table.data.size * table.data.itemsize / 1024 / 1024 / 1024.0
    msg = 'Created a shared table with {size} rows from {file} in {time:.2f} s, using {ram:.2f} GB'
    getLogger(__name__).info(msg.format(file=file, size=f.photonTable.shape[0], time=time.time() - tic, ram=ram))
    return table


class SharedPhotonList(object):
    """multiprocessingsafe shared photon table, readonly!!!!"""

    def __init__(self, file):

        file = os.path.abspath(file)
        self._primary = True
        self._name = file.replace(os.path.sep, '_')
        try:
            SharedArray.delete("shm://{}".format(self._name))
            getLogger(__name__).debug("Deleted existing shared memory store {}".format(self._name))
        except FileNotFoundError:
            pass
        f = Photontable(file)
        tic = time.time()
        self.data = np.frombuffer(SharedArray.create("shm://{}".format(self._name), f.photonTable.shape,
                                                     dtype=PhotonNumpyType), dtype=PhotonNumpyType)
        f.photonTable.read(out=self.data)
        toc = time.time()
        ram = self.data.size * self.data.itemsize / 1024 / 1024 / 1024.0
        msg = 'Created a shared table with {size} rows from {file} in {time:.2f} s, using {ram:.2f} GB'
        getLogger(__name__).debug(msg.format(file=file, size=f.photonTable.shape[0], time=toc - tic, ram=ram))

    def __getstate__(self):
        d = dict(__dict__)
        d.pop('data', None)
        d.pop('raw')
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._primary = False
        getLogger(__name__).debug('Attaching to ' + self._name)
        self.data = np.frombuffer(SharedArray.attach("shm://{}".format(self._name), dtype=PhotonNumpyType))

    def __del__(self):
        if self._primary:
            getLogger(__name__).debug('Deleting ' + self._name)
            SharedArray.delete("shm://{}".format(self._name))


_METADATA_BLOCK_BYTES = 4 * 1024 * 1024
_KEY_BYTES = 256
_VALUE_BYTES = 8192


class Photontable:
    TICKS_PER_SEC = int(1.0 / 1e-6)  # each integer value is 1 microsecond

    class PhotonDescription(tables.IsDescription):
        resID = tables.UInt32Col(pos=0)
        time = tables.UInt32Col(pos=1)
        wavelength = tables.Float32Col(pos=2)
        weight = tables.Float32Col(pos=3)

    def __init__(self, file_name, mode='read', verbose=False, in_memory=False):
        """
        Create Photontable object and load in specified HDF5 file.

        Parameters
        ----------
            file_name: String
                Path to HDF5 File
            mode: String
                'read' or 'write'. File should be opened in 'read' mode
                unless you are applying a calibration.
            verbose: bool
                Prints debug messages if True
        Returns
        -------
            Photontable instance

        """
        self.mode = 'write' if mode.lower() in ('write', 'w', 'a', 'append') else 'read'
        self.verbose = verbose
        self.file = None
        self.table = None
        self.filename = file_name
        self.nominal_wavelength_bins = None
        self.pixel_map = None
        self._flagArray = None
        self._mdcache = None
        self.in_memory = in_memory
        self._load_file()

    def __del__(self):
        """ Closes the obs file """
        if self.file is None:
            return
        try:
            self.file.close()
            del self.file
            self.file = None
        except:
            getLogger(__name__).warning('Error on file close', exc_info=True)

    def __str__(self):
        return 'Photontable: ' + self.filename

    def nyquist_wavelengths(self, min=None, max=None):
        """Return a wavelegnth array based on the wavecal resolution and the minimum and maximum wavelengths.

        min and max may be specified to set alternative minimum and maximum sampling points
        """
        try:
            resdata = self.query_header('wavecal.resolution')
        except KeyError:
            getLogger(__name__).debug('No wavecal, returning nominal values')
            return self.wavelength_bins(width=self.query_header('energy_resolution'),
                                        start=self.query_header('min_wavelength'),
                                        stop=self.query_header('max_wavelength'))

        dl = InterpolatedUnivariateSpline(resdata['wave'], resdata['wave'] / resdata['r'], w=1 / resdata['r_err'])
        if min is None:
            min = self.query_header('min_wavelength')
            # min = resdata['wave'][0] - dl(resdata['wave'][0]) / 2
        if max is None:
            max = self.query_header('max_wavelength')
            # max = resdata['wave'][-1] + dl(resdata['wave'][-1]) / 2
        wave = [min - dl(min) / 2]
        while wave[-1] <= max:
            wave.append(wave[-1] + dl(wave[-1]) / 2)
        return np.array(wave)

    def _load_file(self):
        """ Opens file and loads obs file attributes and beammap """
        getLogger(__name__).debug("Loading {} in {} mode.".format(self.filename, self.mode))
        try:
            kwargs = {'driver': 'H5FD_CORE'} if self.in_memory else {}
            self.file = tables.open_file(self.filename, mode='a' if self.mode == 'write' else 'r', **kwargs)
        except (IOError, OSError):
            raise

        # get important cal params
        self.nominal_wavelength_bins = self.nyquist_wavelengths()

        # get the beam image
        self.beamImage = self.file.get_node('/beammap/map').read()
        self._flagArray = self.file.get_node('/beammap/flag')  # The absence of .read() here is correct
        self.nXPix, self.nYPix = self.beamImage.shape

        # get the photontable
        self.photonTable = self.file.get_node('/photons/photontable')

    def _parse_query_range_info(self, startw=None, stopw=None, start=None, stop=None, intt=None):
        """ return a dict with info about the data returned by query with a particular set of args

        None says don't place this limit.

        intt takes precedence over stop

        start times beyond the duration of the file are interpreted as absolute start times. Start times that would
        still be beyond the end of the file are treated as None

        stop times beyond the end of the file are treated as absolute times.

        startw and stopw are overridden with None if they are more than an order of magnitude beyond the edges of
        nominal_wavelength_bins. if not astropy.units.Qunatity, nm is assumed
        """
        try:
            if start > self.duration:
                start = start
                relstart = start - self.start_time
            else:
                relstart = start
                start = self.start_time + start

            qstart = int(relstart * self.TICKS_PER_SEC)

            if start <= 0 or relstart > self.duration:
                raise TypeError

        except TypeError:
            start = self.start_time
            relstart = 0
            qstart = None

        if intt is not None:
            stop = relstart + intt

        try:
            if stop >= self.duration:
                stop -= self.start_time
                if stop < 0 or stop > self.duration:
                    raise TypeError
            relstop = stop
            stop += self.start_time
            qstop = int(relstop * self.TICKS_PER_SEC)

        except TypeError:
            stop = self.duration + self.start_time
            relstop = self.duration
            qstop = None

        if startw is None:
            qminw = None
        else:
            startw = u.Quantity(startw, u.nm).value
            qminw = None if startw <= self.nominal_wavelength_bins[0] / 10 else startw

        if stopw is None:
            qmaxw = None
        else:
            stopw = u.Quantity(stopw, u.nm).value
            qmaxw = None if stopw >= self.nominal_wavelength_bins[-1] * 10 else stopw

        return dict(start=start, stop=stop, relstart=relstart, relstop=relstop, duration=relstop - relstart,
                    qstart=qstart, qstop=qstop, minw=startw, maxw=stopw, qminw=qminw, qmaxw=qmaxw)

    @property
    def duration(self):
        return self.query_header('EXPTIME')

    @property
    def start_time(self):
        return self.query_header('UNIXSTR')

    @property
    def stop_time(self):
        return self.query_header('UNIXEND')

    @property
    def bad_pixel_mask(self):
        """A boolean image with true where pixel data has problems """
        from mkidpipeline.pipeline import PROBLEM_FLAGS  # This must be here to prevent a circular import!
        return self.flagged(PROBLEM_FLAGS)

    @property
    def wavelength_calibrated(self):
        return bool(self.query_header('wavecal'))

    @property
    def flags(self):
        """
        The flags associated with the with the file.

        Changing this once it is initialized is at your own peril!
        """
        from mkidpipeline.pipeline import PIPELINE_FLAGS  # This must be here to prevent a circular import!

        names = self.query_header('flags')
        if not names:
            getLogger(__name__).warning('Flag names were not attached at time of H5 creation. '
                                        'If beammap flags have changed since then things WILL break. '
                                        'You must recreate the H5 file.')
            names = PIPELINE_FLAGS.names
            self.enablewrite()
            self.update_header('flags', names)
            self.disablewrite()

        f = FlagSet.define(*[(n, i, PIPELINE_FLAGS.flags[n].description if n in PIPELINE_FLAGS.flags else '')
                             for i, n in enumerate(names)])
        return f

    def multiply_column_weight(self, resid, weights, column, flush=True):
        """
        Applies a weight calibration to the column specified by colName.

        Parameters
        ----------
        resid: int
            resID of desired pixel
        weights: array of floats
            Array of cal weights. Multiplied into the specified column column.
        column: string
            Name of weight column. Should be 'weight'
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        if column != 'weight':
            raise ValueError(f"{column} is not 'weight'")

        indices = self.photonTable.get_where_list('resID==resid')

        if not (np.diff(indices) == 1).all():
            raise NotImplementedError('Table is not sorted by Res ID!')

        if len(indices) != len(weights):
            raise ValueError('weights length does not match length of photon list for resID!')

        new = self.query(resid=resid, column=column) * np.asarray(weights)
        self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=new, colname=column)
        if flush:
            self.photonTable.flush()

    def enablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'write':
            return
        self.file.close()
        self.mode = 'write'
        self._load_file()

    def disablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'read':
            return
        self.photonTable.flush()
        self.file.close()
        self.mode = 'read'
        self._load_file()

    def attach_new_table(self, group, group_descr, table, table_descr, header, data, force=False):
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")
        try:
            g = self.file.get_node(f'/{group}')
        except tables.NoSuchNodeError:
            g = self.file.create_group("/", group, group_descr)
        try:
            t = self.file.get_node(f'/{group}/{table}')
            if force:
                t._f_remove()
            else:
                raise ValueError('Table already exists')
        except tables.exceptions.NoSuchNodeError:
            pass

        filt = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=True, bitshuffle=False, fletcher32=False)
        t = self.file.create_table(g, name=table, description=table_descr, title=header, expectedrows=len(data),
                                   filters=filt, chunkshape=None)
        t.append(data)
        t.flush()

    def detailed_str(self):
        t = self.photonTable.read()
        tinfo = repr(self.photonTable).replace('\n', '\n\t\t')
        if np.all(t['time'][:-1] <= t['time'][1:]):
            sort = 'time '
        elif np.all(t['resID'][:-1] <= t['resID'][1:]):
            sort = 'resID '
        else:
            sort = 'Un'

        msg = ('{file}:\n'
               '\t{nphot:.3g} photons, {sort}sorted\n'
               '\tT= {start} - {stop} ({dur} s)\n'
               '\tTable repr: {tbl}\n'
               '\t{dirty}\n'
               '\twave: {wave}\n'
               '\tflat: {flat}\n')

        dirty = ', '.join([n for n in self.photonTable.colnames if self.photonTable.cols._g_col(n).index is not None
                           and self.photonTable.cols._g_col(n).index.dirty])

        s = msg.format(file=self.filename, nphot=len(self.photonTable), sort=sort, tbl=tinfo,
                       start=t['time'].min(), stop=t['time'].max(), dur=self.duration,
                       dirty='Column(s) {} have dirty indices.'.format(dirty) if dirty else 'No columns dirty',
                       wave=self.query_header('wavecal'), flat=self.query_header('flatcal'))
        return s

    def xy(self, photons):
        """Return a tuple of two arrays corresponding to the x & y pixel positions of the given photons"""
        flatbeam = self.beamImage.flatten()
        beamsorted = np.argsort(flatbeam)
        ind = np.searchsorted(flatbeam[beamsorted], photons["resID"])
        return np.unravel_index(beamsorted[ind], self.beamImage.shape)

    def flagged(self, flags, pixel=(slice(None), slice(None)), allow_unknown_flags=True, all_flags=False,
                resid=None):
        """
        Test to see if a flag is set on a given pixel or set of pixels

        if resid is set it takes precedence over pixel
        :param pixel: (x,y) of pixel, 2d slice, list of (x,y), if not specified all pixels are used
        :param flags: if an empty set/None it the pixel(s) are considered unflagged
        :param allow_unknown_flags:
        :param all_flags: Require all specified flags to be set for the mask to be True
        :return:
        """

        if resid is not None:
            pixel = tuple(map(tuple, np.argwhere(resid == self.beamImage)))

        x, y = zip(*pixel) if isinstance(pixel[0], tuple) else pixel
        if not flags:
            return False if isinstance(x, int) else np.zeros_like(self._flagArray[x, y], dtype=bool)

        f = self.flags
        if len(set(f.names).difference(flags)) and not allow_unknown_flags:
            return False if isinstance(x, int) else np.zeros_like(self._flagArray[x, y], dtype=bool)

        bitmask = f.bitmask(flags, unknown='ignore')
        bits = self._flagArray[x, y] & bitmask
        return bits == bitmask if all_flags else bits.astype(bool)

    def flag(self, flag: int, pixel=(slice(None), slice(None))):
        """
        Applies a flag to the selected pixel on the BeamFlag array. Flag is a bitmask;
        new flag is bitwise OR between current flag and provided flag. Flag definitions
        can be found in mkidcore.pixelflags, flags extant when file was created are in self.flag_names

        Named flags must be converged to bitmask via self.flag_bitmask(flag names) first

        Parameters
        ----------
        pixel: 2-tuple of int/slice denoting x,y pixel location
        flag: int Flag to apply to pixel
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        self.flags.valid(flag, error=True)
        if not np.isscalar(flag) and self._flagArray[pixel].shape != flag.shape:
            raise ValueError('flag must be scalar or match the desired region selected by x & y coordinates')
        self._flagArray[pixel] |= flag
        self._flagArray.flush()

    def unflag(self, flag, pixel=(slice(None), slice(None))):
        """
        Resets the specified flag in the BeamFlag array to 0. Flag is a bitmask;
        only the bit(s) specified by 'flag' is/are reset.

        Named flags must be converged to bitmask via self.flags.flag_bitmask(flag names) first

        Parameters
        ----------
        pixel: 2-tuple of ints/slices
            xy-coordinate of pixel
        flag: int
            Flag to undo
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        self.flags.valid(flag, error=True)
        if not np.isscalar(flag) and self._flagArray[pixel].shape != flag.shape:
            raise ValueError('flag must be scalar or match the desired region selected by x & y coordinates')
        self._flagArray[pixel] &= ~flag
        self._flagArray.flush()

    def query(self, startw=None, stopw=None, start=None, stopt=None, resid=None, intt=None, pixel=None, column=None):
        """
        intt takes precedence, all none is the full file

        if a column is specified there is no need to do ['colname'] on the return

        :param start:
        :param pixel:
        :param stopt:
        :param intt:
        :param startw: number or none
        :param stopw: number or none
        :param resid: number, list/array or None

        pixel may be used and will be converted to the appropriate resid via the beamamp, resid takes precedence
        use caution with slices and large numbers of pixels!

        :return:
        """
        if pixel and not resid:
            resid = tuple(self.beamImage[pixel].ravel())

        query_nfo = self._parse_query_range_info(startw=startw, stopw=stopw, start=start, stop=stopt, intt=intt)
        start = query_nfo['qstart']
        stopt = query_nfo['qstop']
        startw = query_nfo['qminw']
        stopw = query_nfo['qmaxw']

        if resid is None:
            resid = tuple()

        try:
            iter(resid)
        except TypeError:
            resid = (resid,)

        if startw is None and stopw is None and start is None and stopt is None and not resid:
            return self.photonTable.read(field=column)  # we need it all!

        res = '|'.join(['(resID=={})'.format(r) for r in map(int, resid)])
        res = '(' + res + ')' if '|' in res and res else res
        tp = '(time < stopt)'
        tm = '(time >= start)'
        wm = '(wavelength >= startw)'
        wp = '(wavelength < stopw)'
        # should follow '{res} & ( ({time}) & ({wave}))'

        if startw is not None:
            if stopw is not None:
                wave = '({} & {})'.format(wm, wp)
            else:
                wave = wm
        elif stopw is not None:
            wave = wp
        else:
            wave = ''

        if start is not None:
            if stopt is not None:
                timestr = '({} & {})'.format(tm, tp)
            else:
                timestr = tm
        elif stopt is not None:
            timestr = tp
        else:
            timestr = ''

        query = res
        if res and (timestr or wave):
            query += '&'
        if res and timestr and wave:
            query += '('

        query += timestr
        if timestr and wave:
            query += '&'

        query += wave
        if res and timestr and wave:
            query += ')'

        if not query:
            getLogger(__name__).debug('Null Query, returning empty result')
            return np.array([], dtype=mkidcore.binfile.mkidbin.PhotonNumpyType)
        else:
            tic = time.time()
            try:
                q = self.photonTable.read_where(query, field=column)
            except SyntaxError:
                raise
            toc = time.time()
            msg = 'Fetched {}/{} rows in {:.3f}s using indices {} for query {} \n\t st:{} et:{} sw:{} ew:{}'
            getLogger(__name__).debug(msg.format(len(q), len(self.photonTable), toc - tic,
                                                 tuple(self.photonTable.will_query_use_indexing(query)), query,
                                                 *map(lambda x: '{:.2f}'.format(x) if x is not None else 'None',
                                                      (start, stopt, startw, stopw))))

            return q

    def filter_photons_by_flags(self, photons, allowed=(), disallowed=()):
        """
        Parameters
        ----------
        photons: structured array
            photon list to be filtered
        allowed: tuple
            collection of pixel flags to keep photons for
        disallowed: tuple
            collection of pixel flags to remove photons for

        Return
        ------
        photons filtered by flags
        """

        if len(allowed) > 0:
            raise NotImplementedError

        filtered = self.flagged(disallowed, self.xy(photons))
        return photons[np.invert(filtered)]

    def get_wcs(self, sample_times=None, derotate=True, wcs_timestep=None, cube_type=None, bins=None):
        """
        sample_times or wcs_timestep required, sample_times takes precedence. single_pa_time is a pa time to override
        all the timesteps (a single wcs will be returned). sample_times is in unixtime.

        bins are the cube bins and is required if cube_type is not None

        Parameters
        ----------
        derotate : bool
             True:  align each wcs solution to position angle = 0
             False:  no correction angle
        wcs_timestep : float
            Determines the time between each wcs solution (each new position angle)
        target_coordinates : SkyCoord, str
            Used to get parallatic angles. string is used to query simbad.
        cube_type : str
            None: wcs solution is calculated for ra/dec
            'wave':  wcs solution is calculated for ra/dec/wavelength
            'time':  wcs solution is calculated for ra/dec/time
        bins: array
            must be passed with the evenly spaced bins of the cube if cube_type is wave or time
        single_pa_time : float
            Time at which to orient all non-derotated frames, overrides derotate

        See instruments.compute_wcs_ref_pixel() for information on wcscal parameters

        Returns
        -------
        List of wcs headers at each position angle
        :param bins:
        """

        if not derotate and sample_times is None:
            sample_times = np.array([self.start_time])
        elif sample_times is None:
            sample_times = np.arange(self.start_time, self.stop_time, wcs_timestep)

        start_time = sample_times[0]
        ref_pixels = []
        try:
            for t in sample_times:
                md = self.metadata(t)
                instrument = md['INSTRUME'].lower()
                head = mkidcore.metadata.build_header(md, unknown_keys='ignore',
                                                      KEY_INFO=INSTRUMENT_KEY_MAP[instrument]['keys'],
                                                      TIME_KEYS=INSTRUMENT_KEY_MAP[instrument]['time'],
                                                      DEFAULT_CARDSET=INSTRUMENT_KEY_MAP[instrument]['card'],
                                                      TIME_KEY_BUILDER=INSTRUMENT_KEY_MAP[instrument]['builder'])
                ref_pixels.append(compute_wcs_ref_pixel((md['E_CONEXX'], md['E_CONEXY']),
                                                        reference_pixel=(head['E_PREFX'], head['E_PREFY']),
                                                        reference=(head['E_CXREFX'], head['E_CXREFY']),
                                                        conex_deltas=(md['E_DPDCX'], md['E_DPDCY'])))
        except KeyError:
            getLogger(__name__).warning('Insufficient data to build a WCS solution, conex info missing')
            return None

        cubeaxis = {}
        if cube_type in ('wave', 'time'):
            cubeaxis = {'CTYPE3': 'TIME' if cube_type == 'time' else 'WAVE',
                        'CUNIT3': 's' if cube_type == 'time' else 'nm',
                        'CDELT3': bins[1] - bins[0], 'CRPIX3': 1, 'CRVAL3': bins[0], 'NAXIS3': bins.size}

        # we want to inherit defaults for ref values
        header = mkidcore.metadata.build_header(self.metadata(start_time), unknown_keys='warn',
                                                KEY_INFO=INSTRUMENT_KEY_MAP[instrument]['keys'],
                                                TIME_KEYS=INSTRUMENT_KEY_MAP[instrument]['time'],
                                                DEFAULT_CARDSET=INSTRUMENT_KEY_MAP[instrument]['card'],
                                                TIME_KEY_BUILDER=INSTRUMENT_KEY_MAP[instrument]['builder'])
        wcs_solns = mkidcore.metadata.build_wcs(header, astropy.time.Time(val=sample_times, format='unix'), ref_pixels,
                                                self.beamImage.shape, subtract_parallactic=derotate, cubeaxis=cubeaxis)
        return wcs_solns

    def get_pixel_spectrum(self, pixel, start=None, duration=None, weight=False,
                           wave_start=None, wave_stop=None, bin_width=None, bin_edges=None, bin_type='energy'):
        """
        returns a spectral histogram of a given pixel integrated from start to start+duration,
        and an array giving the cutoff wavelengths used to bin the wavelength values

        Wavelength Bin Specification:
        Depends on parameters: wave_start, stop, wvlBinWidth, energyBinWidth, wvlBinEdges.
        Can only specify one of: wvlBinWidth, energyBinWidth, or wvlBinEdges. If none of these are specified,
        default wavelength bins are used. If flat calibration exists and is applied, flat cal wavelength bins
        must be used.

        Parameters
        ----------
        xCoord: int
            x-coordinate of desired pixel.
        yCoord: int
            y-coordinate index of desired pixel.
        start: float
            Start time of integration, in seconds relative to beginning of file
        duration: float
            Total integration time in seconds. If -1, everything after start is used
        weight: bool
            If True, weights counts by spectral/flat/linearity weight
        wave_start: float
            Start wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified (otherwise it's redundant). Defaults to self.wvlLowerLimit or 7000.
        wvlEnd: float
            End wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified. Defaults to self.wvlUpperLimit or 15000
        wvlBinWidth: float
            Width of histogram wavelength bin. Used for fixed wavelength bin widths.
        energyBinWidth: float
            Width of histogram energy bin. Used for fixed energy bin widths.
        wvlBinEdges: numpy array of floats
            Specifies histogram wavelength bins. wave_start and wvlEnd are ignored.
        timeSpacingCut: ????
            Legacy; unused


        Returns
        -------
        Dictionary with keys:
            'spectrum' - spectral histogram of given pixel.
            'wvlBinEdges' - edges of wavelength bins
            'effIntTime' - the effective integration time for the given pixel
                           after accounting for hot-pixel time-masking.
            'rawCounts' - The total number of photon triggers (including from
                            the noise tail) during the effective exposure.
                            :param weight:
                            :param wave_start:
                            :param wave_stop:
                            :param pixel:
                            :param start:
                            :param duration:
                            :param bin_width:
                            :param bin_edges:
                            :param bin_type:
        """

        photons = self.query(pixel=pixel, start=start, intt=duration, startw=wave_start, stopw=wave_stop)

        weights = photons['weight'] if weight else None

        if bin_edges and bin_width:
            getLogger(__name__).warning('Both bin_width and bin_edges provided. Using edges')
        elif not bin_edges and bin_width:
            bin_edges = self.wavelength_bins(width=bin_width, start=wave_start, stop=wave_stop,
                                             energy=bin_type == 'energy')
        elif not bin_edges and not bin_width:
            bin_edges = self.nominal_wavelength_bins

        spectrum, _ = np.histogram(photons['wavelength'], bins=bin_edges, weights=weights)

        return {'spectrum': spectrum, 'wavelengths': bin_edges, 'nphotons': len(photons)}

    def get_fits(self, start=None, duration=None, weight=False, wave_start=None,
                 wave_stop=None, rate=True, cube_type=None, bin_width=None, bin_edges=None, derotate=True,
                 bin_type='energy', exclude_flags=pixelflags.PROBLEM_FLAGS, **kwargs):
        """
        Return a fits hdul of the data.

        if cube_type is time or wave a spectral or temporal cube will be returned

        bin_type is for computing bins of bin_width: is width in energy or wavelength anything other than
        'energy' will be treated as wavelength requested.
        if cube_type is 'time' bin_width/edges are in seconds and one of the two is required. bin_type is ignored.

        see self.wavelength_bins() for how wavelegnth bins are computed from the arguments

        Temporal bin_edges will take precedence over start and duration

        :param wave_stop:
        :param rate:
        :param cube_type:
        :param bin_width:
        :param bin_edges:
        :param bin_type:
        :param start: float
            Photon list start time (seconds) relative or absolute. see _parse_query_range_info for details
        :param duration: float
            Photon list end time, in seconds relative to start.
            If None, goes to end of file
        :param weight: bool
            If True, applies the spectral/flat/linearity weight
        :param exclude_flags: int
            Specifies flags to exclude from the tallies
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py
        :param wave_start:
        """
        if type(bin_edges) == np.ndarray:
            bin_edges = bin_edges.copy()

        if cube_type:
            cube_type = cube_type.lower()

        bin_type = bin_type.lower()

        tic = time.time()

        ridbins = sorted(self.beamImage.ravel())
        ridbins = np.append(ridbins, ridbins[-1] + 1)

        time_nfo = self._parse_query_range_info(start=start, intt=duration, startw=wave_start, stopw=wave_stop)

        if cube_type == 'time':
            ycol = 'time'
            if bin_edges is None:
                try:
                    bin_edges = np.linspace(time_nfo['relstart'], time_nfo['relstart'] + time_nfo['duration'],
                                            int(time_nfo['duration'] / bin_width) + 1)
                except TypeError:
                    raise Warning('Either bin_width or bin_edges must be specified for get_fits')

            start = bin_edges[0]
            time_nfo['duration'] = duration = bin_edges[-1] - bin_edges[0]
            bin_edges *= 1e6

        elif cube_type == 'wave':
            ycol = 'wavelength'
            if bin_edges and bin_width:
                getLogger(__name__).warning('Both bin_width and bin_edges provided. Using edges')
            elif not bin_edges and bin_width:
                bin_edges = self.wavelength_bins(width=bin_width, start=wave_start, stop=wave_stop,
                                                 energy=bin_type == 'energy')
            elif not bin_edges and not bin_width:
                bin_edges = self.nominal_wavelength_bins
        else:
            ycol = None
            bin_edges = self.nominal_wavelength_bins[[0, -1]]
            duration = self.duration
        # Retrieval rate is about 2.27Mphot/s for queries in the 100-200M photon range
        photons = self.query(start=start, intt=duration, startw=wave_start, stopw=wave_stop)

        weights = photons['weight'] if weight else None
        if weights is not None and (weights == 0).all():
            getLogger(__name__).critical(f'weights=True and all photon weights are zero, are you sure?')
        if cube_type in ('time', 'wave'):
            data = np.zeros((self.nXPix, self.nYPix, bin_edges.size - 1))
            hist, xedg, yedg = np.histogram2d(photons['resID'], photons[ycol], bins=(ridbins, bin_edges),
                                              weights=weights)
        else:
            data = np.zeros((self.nXPix, self.nYPix))
            hist, xedg = np.histogram(photons['resID'], bins=ridbins, weights=weights)

        toc = time.time()
        xe = xedg[:-1]
        for pixel, resID in np.ndenumerate(self.beamImage):
            if self.flagged(exclude_flags, pixel):
                continue
            data[pixel] = hist[xe == resID]
        data = np.moveaxis(data, -1, 0)
        toc2 = time.time()
        getLogger(__name__).debug(f'Histogram completed in {toc2 - tic:.2f} s, reformatting in {toc2 - toc:.2f}')

        md = self.metadata(timestamp=time_nfo['start'])

        md['E_H5FILE'] = self.filename

        import mkidpipeline.config as pipeconfig
        md['E_GITHSH'] = pipeconfig.git_hash()
        md['E_CFGHSH'] = pipeconfig.config_hash()

        md['UNIXSTR'] = time_nfo['start']
        md['UNIXEND'] = time_nfo['stop']
        md['EXPTIME'] = time_nfo['duration']

        md.pop('data_path')
        flaglist = np.array(md.pop('flags'))

        excluded = self.flags.bitmask(exclude_flags, unknown='ignore')
        pixcal_hdu = [fits.ImageHDU(data=self._flagArray, name='FLAGS'),
                      fits.ImageHDU(data=(self._flagArray & excluded).astype(int), name='BAD'),
                      fits.TableHDU.from_columns(np.recarray(shape=flaglist.shape, buf=flaglist,
                                                             dtype=np.dtype([('flags', flaglist.dtype)])),
                                                 name='FLAG_NAMES')]

        # Deal with non Primary HDU keys
        ext_cards = [fits.Card('CRAYCAL', md.pop('cosmiccal'), comment='Cosmic ray data calculated'),
                     fits.Card('PIXCAL', md.pop('pixcal'), comment='Pixel masking step performed'),
                     fits.Card('LINCAL', md.pop('lincal'), comment='Linearity (dead time) corrected data'),
                     fits.Card('SPECCAL', md.pop('speccal'), comment='Speccal applied to data'),
                     fits.Card('WAVECAL', md.pop('wavecal'), comment='Wavecal applied to data'),
                     fits.Card('FLATCAL', md.pop('flatcal'), comment='Flatcal applied to data'),
                     fits.Card('H5MINWAV', md.pop('min_wavelength'), comment='Min wavelength in h5 file'),
                     fits.Card('H5MAXWAV', md.pop('max_wavelength'), comment='Max wavelength in h5 file'),
                     fits.Card('MINWAVE', time_nfo['minw'], comment='Lower wavelength cut'),
                     fits.Card('MAXWAVE', time_nfo['maxw'], comment='Upper wavelength cut'),
                     fits.Card('ERESOL', md.pop('energy_resolution'), comment='Nominal energy resolution'),
                     fits.Card('DEADTIME', md.pop('dead_time'), comment='Firmware dead-time (us)'),
                     fits.Card('UNIT', 'photons/s' if rate else 'photons', comment='Count unit'),
                     fits.Card('EXFLAG', excluded, comment='Bitmask of excluded flags')]

        # Build primary and image headers
        instrument = md['INSTRUME'].lower()
        header = mkidcore.metadata.build_header(md, unknown_keys='ignore',
                                                KEY_INFO=INSTRUMENT_KEY_MAP[instrument]['keys'],
                                                TIME_KEYS=INSTRUMENT_KEY_MAP[instrument]['time'],
                                                DEFAULT_CARDSET=INSTRUMENT_KEY_MAP[instrument]['card'],
                                                TIME_KEY_BUILDER=INSTRUMENT_KEY_MAP[instrument]['builder'])

        bins = None
        if cube_type is not None:
            bins = bin_edges / 1e6 if cube_type == 'time' else bin_edges
        #TODO set single_pa_time=None when generating an image|scube|tcube output
        wcs = self.get_wcs(cube_type=cube_type, bins=bins, derotate=derotate)
        if wcs:
            # TODO make sure there are no cases where this gives unintuitive behavior
            if isinstance(wcs, list):
                getLogger(__name__).info('More than one WCS solution found for FITS header - writing WCS solution for '
                                         'the first frame to the header')
                wcs = wcs[0]
            header.update(wcs.to_header())
        hdr = header.copy()
        hdr.extend(ext_cards, unique=True)

        # Build HDU List

        if cube_type is None:
            bin_hdu = []
        else:
            tmp = bin_edges / 1e6 if cube_type == 'time' else bin_edges
            bin_hdu = [fits.TableHDU.from_columns(np.recarray(shape=bin_edges.shape, buf=tmp,
                                                              dtype=np.dtype([('edges', bin_edges.dtype)])),
                                                  name='CUBE_EDGES')]
            bin_hdu[0].header.append(fits.Card('UNIT', 's' if cube_type == 'time' else 'nm', comment='Bin unit'))

        torate = 1 / duration if cube_type != 'time' else 1 / np.diff(bins)
        sci_data = data
        if rate:
            sci_data = data.copy()
            try:
                sci_data *= torate
            except:
                sci_data *= torate[:, None, None]  # TODO this probably doesn't work for 4D

        hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                             fits.ImageHDU(data=sci_data, header=hdr, name='SCIENCE'),
                             fits.ImageHDU(data=data, header=hdr, name='VARIANCE')] + bin_hdu + pixcal_hdu)

        getLogger(__name__).debug(f'FITS generated in {time.time() - tic:.0f} s')
        return hdul

    def query_header(self, name, last_if_series=False):
        """
        Returns a requested entry from the obs file header
        """
        if name not in self.file.root.photons.photontable.attrs:
            raise KeyError(name)
        # the implementation does not like missing get calls
        x = getattr(self.file.root.photons.photontable.attrs, name)
        if isinstance(x, mkidcore.metadata.MetadataSeries) and last_if_series:
            return x.values[-1]
        else:
            return x

    def update_header(self, key, value):
        """
        Modifies an entry in the header. Useful for indicating whether wavelength cals,
        flat cals, etc are applied

        Parameters
        ----------
        key: string
            Name of entry to be modified
        value: depends on title
            New value of entry
        """
        if self.mode != 'write':
            raise IOError("Must open file in write mode to do this!")

        if isinstance(value, CommentedSeq):
            # Always converted to a tuple because a commented sequence (list or tuple created by the YAML when reading
            # in a sequence of info) cannot be written to photontable header
            getLogger(__name__).debug(f"Converting CommentedSeq {value} to tuple so header can be updated.")
            value = tuple(value)

        if key in self.file.root.photons.photontable.attrs._f_list('sys'):
            raise KeyError(f'"{key}" is reserved for use by pytables')

        if key not in self.file.root.photons.photontable.attrs._f_list('user'):
            getLogger(__name__).info(f'Adding new header key: {key}')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)
            setattr(self.file.root.photons.photontable.attrs, key, value)

    def metadata(self, timestamp=None):
        """ Return an dict of key, value pairs associated with the dataset

        if no timestamp is specified the first record for a key (if a series) is returned.
        if a timestamp is specified then the most recent preceeding record is returned
        """

        records = {}

        for k in self.file.root.photons.photontable.attrs._f_list('user'):
            data = getattr(self.file.root.photons.photontable.attrs, k)
            try:
                data = data.get(timestamp, preceeding=True)
                records[k] = data
            except AttributeError:
                records[k] = data
            except ValueError as e:
                getLogger(__name__).warning(e)
                pass  # no data

        return records

    def attach_observing_metadata(self, metadata):
        if self.mode != 'write':
            raise IOError("Must open file in write mode to do this!")
        for k, v in metadata.items():
            self.update_header(k, v)

    @staticmethod
    def wavelength_bins(width=.1, start=700, stop=1500, energy=True):
        """
        returns an array of wavlength bin edges, with a fixed energy bin width
        withing the limits given in wave_start and wave_stop
        Args:
            width: bin width in eV or wavelength
            start: Lower wavelength edge in Angstrom
            stop: Upper wavelength edge in Angstrom
            energy: (True) set to false
        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """
        if not energy:
            return np.linspace(start, stop, int((stop - start) / width) + 1)

        h = astropy.constants.h.to('eV s').value
        c = astropy.constants.c.to('m/s').value
        const = h * c * 1e9
        # Calculate upper and lower energy limits from wavelengths, note that start and stop switch when going to energy
        e_stop = const / start
        e_start = const / stop
        n = int((e_stop - e_start) / width)
        # Construct energy bin edges (reversed) and convert back to wavelength
        return const / np.linspace(e_stop, e_start, n + 1)

    def resonators(self, exclude=None, select=None, pixel=False):
        """A resonator iterator excluding resonators flagged with any flags in exclude and selecting only pixels with
        all the flags in select (select=None|empty implies no restriction. set pixel=True to yield resID,(x,y) instead
        of resid"""
        excludemask = self.flagged(exclude, all_flags=False)
        selectmask = self.flagged(select, all_flags=False)
        for pix, resid in np.ndenumerate(self.beamImage):
            excl, sel = excludemask[pix], selectmask[pix]
            if excl or (select and not sel):
                continue
            yield (pix, resid) if pixel else resid

    def needed_ram(self):
        amount = len(self.photonTable) * self.photonTable.dtype.itemsize * 3  # 25
        return self.ram_manager(amount)
