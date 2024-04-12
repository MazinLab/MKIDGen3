import logging.config
import numpy as np
import os
import yaml
import importlib.resources
from logging import getLogger
import zmq
from mkidgen3.opfb import opfb_bin_number, opfb_bin_center
import numpy.typing as nt
from mkidgen3.system_parameters import (OPFB_CHANNEL_SAMPLE_RATE, ADC_SAMPLE_RATE, N_OPFB_CHANNELS, DAC_FREQ_MIN,
                                        DAC_FREQ_MAX, DAC_FREQ_RES)
from typing import Iterable
import subprocess


def pseudo_random_tones(n: int, buffer: float = 300e3, spread: bool = True, exclude: Iterable = None) -> (
        nt.NDArray)[float]:
    """
    Generate a 1D array of tones where each tone in randomly placed in an OPFB bin
    Args:
        n: number of tones
        buffer: bandwidth from the edges of the bin where tones will not be placed[Hz]
        spread: use every other opfb channels so tones are more spread out.
        exclude: list of tones whose bin will not have a tone placed anywhere in the same bin.
    Returns:
        A 1D array of tones where there's 1 tone per OPFB bin and buffer away from the upper and lower bin edges.
    Tones are placed symmetrically around DC, 1 per bin with the specified buffer

    """
    opfb_halfband = OPFB_CHANNEL_SAMPLE_RATE / 2
    exclude = np.asarray(exclude)
    assert n % 2 == 0, "Only even number of tones is supported."
    assert n < 4095, "Max number of tones is 4095 (one per bin excluding DC bin)."
    assert buffer < opfb_halfband, f"Buffer size is larger than channel width, max buffer allowed is {opfb_halfband}."
    rand_offsets = np.random.default_rng(seed=2).uniform(low=buffer-opfb_halfband, high=opfb_halfband-buffer, size=n)
    bc = (ADC_SAMPLE_RATE / N_OPFB_CHANNELS) * np.linspace(-N_OPFB_CHANNELS / 2, N_OPFB_CHANNELS / 2 - 1,
                                                                    N_OPFB_CHANNELS)
    if spread:
        bc = bc[::2]
    tone_bin_centers = np.concatenate((bc[bc.size//2-n//2:bc.size//2], bc[bc.size//2:+bc.size//2+n//2]))

    if exclude.any():
        exclude_bin_centers = opfb_bin_center(opfb_bin_number(exclude, ssr_raw_order=False), ssr_order=False)
        exclude_idx = []
        for i, tone_bin in enumerate(tone_bin_centers):
            if (tone_bin == exclude_bin_centers).any():
                exclude_idx.append(i)
        tone_bin_centers = np.delete(tone_bin_centers, exclude_idx)
        rand_offsets = np.delete(rand_offsets, exclude_idx)
    return np.clip(tone_bin_centers + rand_offsets, DAC_FREQ_MIN+DAC_FREQ_RES, DAC_FREQ_MAX-DAC_FREQ_RES)



def compute_max_val(x) -> float:
    return max(x.real.max(), x.imag.max(), np.abs(x.imag.min()), np.abs(x.imag.min()))


def rx_power(data: nt.NDArray[np.int32] | nt.NDArray[np.complex64]) -> tuple[float, float, float]:
    """
    Compute ADC average power received in milliwatts and dBm and maximum vppd.
    Args:
        data: ADC capture data: complex and integer data normalized to max val = 1.0 or raw ints are all supported
              normalized non-complex data will pass straight through to the power calculations.

    Returns: tuple: Average Power in mW, Average power in dBm, Maximum VPPD seen by ADC

    """
    if np.iscomplex(data).any():
        if np.any(abs(data.real) > 1.0) or np.any(abs(data.imag) > 1.0):  # normalize
            data = np.concatenate((np.int16(data.real[:, np.newaxis]), np.int16(data.imag[:, np.newaxis])), axis=1)/2**15

        else:  # already normalized
            data = np.concatenate((np.float64(data.real[:, np.newaxis]), np.float64(data.imag[:, np.newaxis])), axis=1)

    elif (abs(data) > 1.0).any():
        data = data / 2**15

    term = 100  # ohms
    vppd = data
    max_adc = np.max(np.abs(vppd), axis=0)
    current = vppd / term  # amps
    p_inst = np.square(current.real)*term # watts
    p_avg_mw = np.sum(p_inst, axis=0)*1000/vppd.size  # mean milliwatts
    dbm_avg = 10*np.log10(p_avg_mw)
    return p_avg_mw, dbm_avg, max_adc


def convert_freq_to_ddc_bins(freqs: Iterable[float | int]) -> np.ndarray:
    """
    Convert frequecnies to bins coming out of the OPFB--useful for programming bin2res
    Args:
        freqs: list or array of frequencies [Hz]

    Returns: np array of bins coming out of the opfb which will contain the frequencies

    """
    bins = np.zeros(2048, dtype=int)
    bins[:freqs.size] = opfb_bin_number(freqs, ssr_raw_order=True)
    return bins


def format_sample_duration(fs: float, n_samp: int) -> str:
    """
    Print the time of some number of samples with a given sample rate in nice units.
    Args:
        fs: sample rate in Hz
        n_samp: number of samples

    Returns:
    For Ex: '0.5 Seconds'

    """
    seconds = n_samp * (1 / fs)

    if seconds > 0.1:
        return f'{seconds:.2f} s'
    elif seconds > 1e-3:
        return f'{seconds / 1e-3:.2f} ms'
    elif seconds > 1e-6:
        return f'{seconds / 1e-6:.2f} us'
    elif seconds > 1e-9:
        return f'{seconds / 1e-9:.2f} ns'


def format_time(t: float) -> str:
    """
    Print the time in a convenient order of magnitude.
    Args:
        t: time in seconds

    Returns:
    For Ex: '0.5 Seconds'

    """

    if t > 0.1:
        return f'{t:.2f} s'
    elif t > 1e-3:
        return f'{t / 1e-3:.2f} ms'
    elif t > 1e-6:
        return f'{t / 1e-6:.2f} us'
    elif t > 1e-9:
        return f'{t / 1e-9:.2f} ns'


def format_bytes(n_bytes: int) -> str:
    """
    Print the number of bytes with a convenient order of magnitude.
    Args:
        n_bytes: number of bytes

    Returns:
    For Ex: '129.2 MiB'
    """
    if n_bytes > 2 ** 30:
        return f'{n_bytes / 2 ** 30:.1f} GiB'
    elif n_bytes > 2 ** 20:
        return f'{n_bytes / 2 ** 20:.1f} MiB'
    elif n_bytes > 2 ** 10:
        return f'{n_bytes / 2 ** 10:.1f} KiB'
    else:
        return f'{n_bytes} bytes'


def delete_pynq_cache_file():
    try:
        os.remove('/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq/pl_server/_current_metadata.pkl')
    except Exception as e:
        return str(e)


def buf2complex(b, free=True, unsigned=False, floating=True):
    """ Convert a pynq buffer to normal numpy array, copying it out of PL DDR4

    Treat input as uint16 if unsigned is set.

    Divide by 2**15 (or 2**16 if unsigned) if floating is set.

    Frees the pynq buffer if free is set
    """
    x = np.array(b)
    if unsigned:
        x = x.astype(np.uint16, copy=False)
    if floating:
        x /= 2 ** (16 if unsigned else 15)
    x = x[..., 0] + 1j * x[..., 1]
    try:
        if free:
            b.freebuffer()
    except AttributeError:
        pass  # not a pynq buffer
    return x


def setup_logging(name):
    ref = importlib.resources.files('mkidgen3.config').joinpath('logging.yaml')
    with importlib.resources.as_file(ref) as path:
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())

    # postprocess loggers dict
    # keys are program names values are either
    #   1) a level for the log of the program
    #   2) a dict of log names and levels
    #   3) a dict of log names and dicts describing how to configure the corresponding Logger instance.
    #  See https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
    # The configuring dict is searched for the following keys:
    #     level (optional). The level of the logger.
    #     propagate (optional). The propagation setting of the logger.
    #     filters (optional). A list of ids of the filters for this logger.
    #     handlers (optional). A list of ids of the handlers for this logger.
    cfg = config['loggers'][name]  # extract one we care about
    if isinstance(cfg, str):
        config['loggers'] = {name: {'level': cfg.upper()}}
    else:
        loggers = {}
        for k, v in cfg.items():
            loggers[k] = {'level': v.upper()} if isinstance(v, str) else v
        config['loggers'] = loggers

    logging.config.dictConfig(config)
    return logging.getLogger(name)


def _which_one_bit_set(x, nbits):
    """
    Given the number x that only has a single bit set return the index of that bit.
    Return None if no bits < nbits bit is set (e.g. nbits=16 will check bits 0-15)
    """
    for i in range(nbits):
        if x & (1 << i):
            return i
    return None


def pack16_to_32(data):
    it = iter(data)
    vals = [x | (y << 16) for x, y in zip(it, it)]
    if data.size % 2:
        vals.append(data[-1])
    return np.array(vals, dtype=np.uint32)


def ensure_array_or_scalar(x):
    if x is None:
        return x
    return x if np.isscalar(x) else np.asarray(x)


def clear_interrupt(core, clear_axi_interrupt_controller=False):
    """
    Attempt to clear an interrupt on both an HLS core and at the Axi interrupt controller. For testing, not production.

    Args:
        core: a pynq.Interrupt or object with a pynq.DefaultIP with both an interrupt and
            register_map.IP_ISR.CHAN0_INT_ST
        clear_axi_interrupt_controller: set to true to reach in and futz with the interrupt controller as well

    Returns: None

    """
    import pynq
    given_int = isinstance(core, pynq.Interrupt)
    interrupt = core if given_int else core.interrupt

    int_number = interrupt.number

    if not given_int and core.register_map.IP_ISR.CHAN0_INT_ST:
        getLogger(__name__).info('Interrupt cleared at controller')

    val = 1 << int_number
    if clear_axi_interrupt_controller:
        parent_mmio = interrupt.parent().mmio
        if not (parent_mmio.read(0x8) & val):
            getLogger(__name__).info('Interrupt is not enabled at controller')
        if parent_mmio.read(0x4) & val:
            parent_mmio.write(0x0C, val)
            getLogger(__name__).info('Interrupt state was cleared at controller')


def ps_ram_sane(nbytes, pynq_region=False):
    """
    Return True if the bytes might be available in PS ram
    Args:
        nbytes: number of bytes
        pynq_region: select if the bytes should be in the region available to pynq.allocate

    Returns: True if the ram might be available

    """
    if pynq_region:
        return nbytes < 2 * 1024 ** 3
    else:
        return nbytes < 2 * 1024 ** 3


def do_asyncio_thing(thing, use_new_thread=False):
    """

    Args:
        thing: an asyncio coroutine. not really clear on the scope that will work safely here yet
           written with do_asyncio_thing(interrupt.wait()) in mind
        use_new_thread: Do the waiting in a new daemon thread

    Returns: None or a future if a new thread is in use

    """
    import asyncio
    from threading import Thread

    def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    newloop = False
    try:
        if use_new_thread:
            raise Exception
        loop = asyncio.get_running_loop()
        getLogger(__name__).debug(f"Using existing loop")
    except:
        loop = asyncio.new_event_loop()
        newloop = True

    if use_new_thread:
        getLogger(__name__).debug(f"Using new loop in new thread")
        thread = Thread(target=start_background_loop, args=(loop,), daemon=True)
        thread.start()
        return asyncio.run_coroutine_threadsafe(thing, loop)

    if newloop:
        getLogger(__name__).debug(f"Using new loop")

    task = loop.create_task(thing)
    getLogger(__name__).debug(f"Will wait in {loop} for {thing} (use_new_thread={use_new_thread})")
    loop.run_until_complete(task)
    if newloop:
        loop.close()


class AbortedException(Exception):
    pass


def check_zmq_abort_pipe(pipe):
    try:
        abort = pipe.recv(zmq.NOBLOCK)
        raise AbortedException(abort)
    except zmq.ZMQError as e:
        if e.errno != zmq.EAGAIN:
            raise


def check_active_jupyter_notebook():
    """Get a list of jupyter notebooks that are running and return true if any have 'http' in the listing """
    x = subprocess.run(['jupyter', 'notebook', 'list'], capture_output=True)
    return False  # 'http' in x.stdout.decode()
