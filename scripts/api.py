import time
import numpy as np
import requests
from logging import getLogger
from mkidgen3.util import setup_logging

setup_logging('gen3-flask')

TIMEOUT = 2.0  # timeout for request post
ADDRESS = 'mkidzcu111b.physics.ucsb.edu:8080'


def _timeout(t):
    return t if t is not None else TIMEOUT


def set_frequencies(frequencies, timeout=None):
    r = requests.post(ADDRESS + '/set_freq', json={'frequencies': frequencies}, timeout=_timeout(timeout))
    return r.json()


def capture(n, timeout=None):
    r = requests.get(ADDRESS + '/iqcap/'+str(n), timeout=_timeout(timeout))
    return r.json()
