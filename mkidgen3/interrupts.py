from pynq.interrupt import _InterruptController, Interrupt
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
    _event_by_id = defaultdict(threading.Event)
    _interrupts = {}

    @staticmethod
    def get_manager():
        if ThreadedPLInterruptManager._thread is None:
            loop, thread = create_uio_loop_thread()
            ThreadedPLInterruptManager._loop = loop
            ThreadedPLInterruptManager._thread = thread
        return ThreadedPLInterruptManager

    @staticmethod
    def get_monitor(name, maxq=1, id=None):
        m = ThreadedPLInterruptManager.get_manager()  # starts up UIO monitoring loop thread first time
        try:
            name = name._interrupts['interrupt']['fullpath']
        except (AttributeError, KeyError):
            if not isinstance(name, str) and name in PL.interrupt_pins:
                raise ValueError("name must either be an interrupt pin name "
                                 "or have _interrupts['interrupt']['fullpath']")

        e = threading.Event() if id is None else m._event_by_id[f"{name}{id}"]
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
    def remove_monitor(event):
        """
        Stop monitoring interrupt for event, removing the last monitor will also stop the queue and disable
        the interrupt
        """
        m = ThreadedPLInterruptManager.get_manager()  # starts up UIO monitoring loop thread first time
        name = None
        for k,v in m._events.items():
            try:
                v.remove(event)
                name = k
            except ValueError:
                pass
        if not name:
            return
        for k, _ in filter(lambda kv: kv[1] == event, m._event_by_id.items()):
            m._event_by_id.pop(k)
        if not m._events[name]:
            ThreadedPLInterruptManager._queues.pop(name)  # no more events, kill the queue

    @staticmethod
    async def interrupt_monitor(name):
        i = ThreadedPLInterruptManager._interrupts[name] = Interrupt(name)

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
                q = ThreadedPLInterruptManager._queues[name]
            except KeyError:
                break
            try:
                q.put_nowait(t)  # will raise full and kill the loop at max queue size
            except queue.Full:
                pass
