import pkg_resources
import os
import yaml
import logging.config
import requests
import numpy as np


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
    if free:
        b.freebuffer()
    return x



def set_anritsu(f):
    """NB make sure Anritsu server is running"""
    r = requests.get(f'http://skynet.physics.ucsb.edu:51111/loset/{f}')
    return r.json()


def setup_logging(name):
    path = pkg_resources.resource_filename('mkidgen3', 'config/logging.yaml')
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
    else:
        return
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
