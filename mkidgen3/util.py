import logging.config
import numpy as np
import os
import yaml
import importlib.resources
from logging import getLogger
import zmq


def print_bytes(n_bytes: int) -> str:
    """
    Print the number of bytes with a convenient order of magnitude.
    Args:
        n_bytes:

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
