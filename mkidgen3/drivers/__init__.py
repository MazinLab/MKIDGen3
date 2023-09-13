try:
    from . import axiswitch, bintores, capture, ddc, dactable, axififo, phasematch, iqgen, rfdc, trigger
except OSError:
    pass
__all__ = ['axiswitch', 'bintores', 'capture', 'ddc', 'dactable', 'axififo', 'rfdc', 'iqgen',
           'phasematch', 'trigger']
