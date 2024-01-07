from pynq.interrupt import _InterruptController, get_uio_irq, Interrupt
from pynq.uio import UioController
from pynq import PL
import threading
import queue
import asyncio
from logging import getLogger
import time
from collections import defaultdict


log = getLogger(__name__)


def _background_uio_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    _InterruptController.get_controller('axi_intc_0')
    loop.run_forever()


def create_uio_loop_thread() -> (asyncio.AbstractEventLoop, threading.Thread):
    eventloop = asyncio.new_event_loop()

    _InterruptController._controllers.clear()
    _InterruptController._uio_devices.clear()
    thread = threading.Thread(target=_background_uio_loop, args=(eventloop,), daemon=True,
                              name='UIO_Asyncio loop thread')
    thread.start()
    return eventloop, thread


class ThreadedPLInterruptManager:
    """
    A thread-safe way of getting interrupt events with pynq. pynq's asyncio interrupt approach requires that all
    await interrupt.wait() and the UIO asyncio reader exist in the same thread (and more, the asyncio.Event objects
    need to have been created in that thread as well.

    This manager provides a static method that is safe to use from any thread that returns interrupt event queue
    (of limited depth!) and a threading.Event that may be waitied on, the queue is unique the interrupt and the event
    to each call of get_monitor(interrupt_name).

    When get_monitor is first called a manager instance will be created that deletes all pynq UioControllers
    and recreates them in a new internal thread that will be using to await interrupt events.
    """
    _loop = None
    _thread = None
    _futures = []

    _queues = {}
    _events = defaultdict(list)
    _interrupts = {}

    @staticmethod
    def get_manager():
        if ThreadedPLInterruptManager._thread is None:
            loop, thread = create_uio_loop_thread()
            ThreadedPLInterruptManager._loop = loop
            ThreadedPLInterruptManager._thread = thread
        return ThreadedPLInterruptManager

    @staticmethod
    def get_monitor(name, maxq=1):
        m = ThreadedPLInterruptManager.get_manager()  # starts up UIO monitoring loop thread first time

        e = threading.Event()
        m._events[name].append(e)
        try:
            q = m._queues[name]
            log.debug(f"Ignoring maxq as have existing q")
            return q, e
        except KeyError:
            pass
        m._queues[name] = q = queue.Queue(maxsize=maxq)

        coro = m.interrupt_monitor(name)
        fut = asyncio.run_coroutine_threadsafe(coro, m._loop)
        m._futures.append(fut)

        return q, e

    @staticmethod
    async def interrupt_monitor(name):
        i = ThreadedPLInterruptManager._interrupts[name] = Interrupt(name)
        q = ThreadedPLInterruptManager._queues[name]

        while True:
            t = time.perf_counter()
            log.info(f"Waiting on interrupt {name}:{i.event} @ {t}. \n"
                     f"  Present watchers: {ThreadedPLInterruptManager._events[name]}")
            await i.wait()
            t = time.perf_counter()
            log.info(f"Interrupt {name} @ {t}")
            for e in ThreadedPLInterruptManager._events[name]:
                e.set()
            try:
                q.put_nowait(t)  # will raise full and kill the loop at max queue size
            except queue.Full:
                pass
