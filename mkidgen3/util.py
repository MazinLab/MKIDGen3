import logging.config
import numpy as np
import os
import yaml
import importlib.resources


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
