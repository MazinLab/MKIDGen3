import zmq
import uuid
from logging import getLogger

def zpipe(ctx):
    """
    build an inproc pipe for talking to threads
    mimic pipe used in czmq zthread_fork.
    Returns a pair of PAIRs connected via inproc
    """
    getLogger('mkidgen3.zmq').debug(f'Creating inproc pipe with context {ctx}')
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % uuid.uuid4().hex
    a.bind(iface)
    b.connect(iface)
    return a, b
