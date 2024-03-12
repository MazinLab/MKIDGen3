import os
import multiprocessing

import h5py
import hdf5plugin

import numpy as np
import numpy.typing as nt

from typing import Optional

TABLE_TYPE = np.dtype(
    [("resID", "<u4"), ("time", "<u4"), ("wavelength", "<f4"), ("weight", "<f4")]
)

DEFAULT_FLAGS = (
    "beammap.noDacTone",
    "beammap.failed",
    "beammap.yFailed",
    "beammap.xFailed",
    "beammap.double",
    "beammap.wrongFeedline",
    "beammap.duplicatePixel",
    "flatcal.bad",
    "flatcal.not_all_weights_valid",
    "pixcal.hot",
    "pixcal.cold",
    "pixcal.dead",
    "wavecal.bad",
    "wavecal.failed_validation",
    "wavecal.failed_convergence",
    "wavecal.not_monotonic",
    "wavecal.not_enough_histogram_fits",
    "wavecal.not_attempted",
    "wavecal.no_histograms",
    "wavecal.histogram_fit_problems",
    "wavecal.linear",
    "wavecal.quadratic",
)


def _pyts(s: str | bytes | h5py.Empty):
    if not s:
        return h5py.Empty("S1")
    if type(s) is bytes:
        return np.array(s, dtype="S")
    if type(s) is h5py.Empty:
        return s
    return np.array(bytes(s, "utf-8"), dtype="S")


def _styp(s):
    if type(s) is h5py.Empty:
        return ""
    return str(bytes(s[::]), encoding="utf-8")


def _pytable_compat(node, name: bytes, c: bytes):
    pytc = _pyts(c)
    pytn = _pyts(name)
    if c == b"TABLE":
        pytv = _pyts(b"2.6")
    elif c == b"ARRAY":
        pytv = _pyts(b"2.4")
    elif c == b"GROUP":
        pytv = _pyts(b"1.0")
    elif c == b"FILE":
        pytc = _pyts(b"GROUP")
        pytv = _pyts(b"1.0")
    else:
        raise ValueError("Pytables class {:s} unsupported".format(c))

    node.attrs["VERSION"] = pytv
    node.attrs["CLASS"] = pytc
    node.attrs["TITLE"] = pytn
    if c == b"FILE":
        node.attrs["PYTABLES_FORMAT_VERSION"] = _pyts("2.0")
    if c == b"ARRAY":
        node.attrs["FLAVOR"] = _pyts("numpy")


class TableWriter:
    _opened = False

    def __init__(
        self,
        filename: str,
        wavelength_bounds: tuple[np.int64 | int, np.int64 | int],
        compress_kwargs=hdf5plugin.Blosc(),
    ):
        self._resize_lock = multiprocessing.Lock()
        self.f = f = h5py.File(filename, "w")
        self._opened = True
        _pytable_compat(f, b"MKID Photon File", b"FILE")
        self._photons = f.create_group("photons")
        _pytable_compat(self._photons, b"Photon Information", b"GROUP")
        self._pt = pt = self._photons.create_dataset(
            "photontable",
            shape=(0,),
            dtype=TABLE_TYPE,
            maxshape=(None,),
            chunks=True,
            **compress_kwargs
        )
        for i, n in enumerate(TABLE_TYPE.names):
            pt.attrs["FIELD_{:d}_FILL".format(i)] = TABLE_TYPE[i].type(0)
            pt.attrs["FIELD_{:d}_NAME".format(i)] = _pyts(n)
        _pytable_compat(pt, b"Photon Datatable", b"TABLE")

        pt.attrs["max_wavelength"] = max(wavelength_bounds)
        pt.attrs["min_wavelength"] = min(wavelength_bounds)

        self.with_start_time()
        self.with_flags()
        self.with_dead_time()
        self.with_energy_resolution()
        self.with_data_dir(str(os.path.dirname(os.path.abspath(filename))))

    def append_photons(self, photons: nt.NDArray[TABLE_TYPE]):
        if len(photons.shape) != 1:
            raise TypeError("Expected 1d array of photons")
        self._resize_lock.acquire()
        index = self._pt.size
        self._pt.resize((self._pt.size + photons.size,))
        self._pt.attrs["NROWS"] = np.int64(index + photons.size)
        self._resize_lock.release()

        self._pt[index : index + photons.size] = photons[::]
        return self

    def with_beammap(
        self, file: str, resids: nt.NDArray[np.int64], flags: nt.NDArray[np.int64]
    ):
        if resids.shape != flags.shape:
            raise TypeError("resids and flags must have the same shape")
        self._pt.attrs["E_BMAP"] = _pyts(str(file))
        b = self.f.create_group("beammap")
        _pytable_compat(b, b"Beammap Information", b"GROUP")
        resid = b.create_dataset("map", data=resids)
        flag = b.create_dataset("flag", data=flags)
        _pytable_compat(resid, b"resID map", b"ARRAY")
        _pytable_compat(flag, b"flag map", b"ARRAY")
        return self

    def with_start_time(self, t: Optional[np.int64 | int] = None):
        if t is None:
            import time

            t = time.time()
        t = np.int64(t)
        self._pt.attrs["UNIXSTR"] = t
        return self

    def with_end_time(self, t: Optional[np.int64 | int] = None):
        if t is None:
            import time

            t = time.time()
        t = np.int64(t)
        self._pt.attrs["UNIXEND"] = t
        if (
            "UNIXSTR" in self._pt.attrs.keys()
            and "EXPTIME" not in self._pt.attrs.keys()
        ):
            self._pt.attrs["EXPTIME"] = t - self._pt.attrs["UNIXSTR"]
        return self

    def with_exposure_time(self, t: np.int64 | int):
        self._pt.attrs["EXPTIME"] = np.int64(t)
        return self

    def with_flags(self, flags: tuple[str, ...] = DEFAULT_FLAGS):
        import pickle

        pick = np.array(pickle.dumps(flags, 0), dtype="S")
        self._pt.attrs["flags"] = pick
        return self

    def with_energy_resolution(self, resolution: float | np.float64 = 0.1):
        self._pt.attrs["energy_resolution"] = np.float64(resolution)
        return self

    def with_dead_time(self, dead_time: int | np.int64 = 0):
        self._pt.attrs["dead_time"] = np.int64(dead_time)
        return self

    def with_data_dir(self, dir: Optional[str]):
        self._pt.attrs["data_path"] = _pyts(dir)
        return self

    def with_wavecal(self, wavecal: Optional[str] = None):
        self._pt.attrs["wavecal"] = _pyts(wavecal)
        return self

    def with_speccal(self, speccal: Optional[str] = None):
        self._pt.attrs["speccal"] = _pyts(speccal)
        return self

    def with_flatcal(self, flatcal: Optional[str] = None):
        self._pt.attrs["flatcal"] = _pyts(flatcal)
        return self

    def with_cosmiccal(self, cosmiccal: bool = False):
        self._pt.attrs["cosmiccal"] = np.uint8(cosmiccal)
        return self

    def with_lincal(self, lincal: bool = False):
        self._pt.attrs["lincal"] = np.uint8(lincal)
        return self

    def with_pixcal(self, pixcal: bool = False):
        self._pt.attrs["pixcal"] = np.uint8(pixcal)
        return self

    def close(self):
        if self._opened:
            if "UNIXEND" not in self._pt.attrs.keys():
                self.with_end_time()
            self.f.close()
            self._opened = False

    def __del__(self):
        self.close()


def roundtrip(input_file: str, output_file: str):
    import pickle

    fi = h5py.File(input_file)
    pt = fi["photons"]["photontable"]
    pta = dict(pt.attrs)
    fo = (
        TableWriter(output_file, (pta["min_wavelength"], pta["max_wavelength"]))
        .with_data_dir(_styp(pta["data_path"]))
        .with_cosmiccal(bool(pta["cosmiccal"]))
        .with_lincal(bool(pta["lincal"]))
        .with_pixcal(bool(pta["pixcal"]))
        .with_wavecal(_styp(pta["wavecal"]))
        .with_speccal(_styp(pta["speccal"]))
        .with_flatcal(_styp(pta["flatcal"]))
        .with_start_time(pta["UNIXSTR"])
        .with_end_time(pta["UNIXEND"])
        .with_energy_resolution(pta["energy_resolution"])
        .with_dead_time(pta["dead_time"])
        .with_flags(pickle.loads(pta["flags"]))
        .append_photons(pt[::])
    )
    if "beammap" in fi.keys():
        fo.with_beammap(
            _styp(pta["E_BMAP"]), fi["beammap"]["map"], fi["beammap"]["flag"]
        )
    fo.close()
