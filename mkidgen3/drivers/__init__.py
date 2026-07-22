try:
    from . import axiswitch, bintores, capture, ddc, dactable, axififo, phasematch, iqgen, rfdc, trigger, triggerv2, ppssync
except (OSError, ImportError):
    pass
__all__ = ['axiswitch', 'bintores', 'capture', 'ddc', 'dactable', 'axififo', 'rfdc', 'iqgen',
           'phasematch', 'trigger', 'triggerv2', 'ppssync']
