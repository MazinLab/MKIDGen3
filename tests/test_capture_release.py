"""Off-board checks of capture.py's buffer release path.

capture.py imports pynq at module scope and cannot be imported off-board, so
the executable settle/release logic lives in a PYNQ-free mixin. Scripted
registers verify runtime ordering and failure behavior; AST checks remain a
cheap net against bypassing that tested path in capture.py.
"""
import ast
from pathlib import Path

import pytest

from mkidgen3.mkidpynq import CaptureBufferRelease

CAPTURE = Path(__file__).resolve().parents[1] / 'mkidgen3' / 'drivers' / 'capture.py'
TREE = ast.parse(CAPTURE.read_text())


def _enclosing_function(node, tree=TREE):
    """Name of the innermost function containing node, or None."""
    best = None
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.lineno <= node.lineno and node.lineno <= (fn.end_lineno or fn.lineno):
            if best is None or fn.lineno > best.lineno:
                best = fn
    return None if best is None else best.name


def _method_references(name):
    """Attribute/getattr references, including aliases later called by name."""
    references = [n for n in ast.walk(TREE)
                  if isinstance(n, ast.Attribute) and n.attr == name]
    references.extend(
        n for n in ast.walk(TREE)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == 'getattr' and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant) and n.args[1].value == name))
    return references


def test_capture_hierarchy_delegates_to_the_tested_release_logic():
    hierarchy = next(n for n in TREE.body
                     if isinstance(n, ast.ClassDef)
                     and n.name == 'CaptureHierarchy')
    methods = {n.name: n for n in hierarchy.body
               if isinstance(n, ast.FunctionDef)}
    for method_name in ('_settle_axis2mm', '_retain_stuck_buffer', '_release'):
        references = [n for n in ast.walk(methods[method_name])
                      if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name)
                      and n.value.id == 'CaptureBufferRelease'
                      and n.attr == method_name]
        assert references, f'{method_name} does not delegate to tested logic'


def test_sweep_arm_state_is_marked_at_the_start_write_boundary():
    functions = {n.name: n for n in ast.walk(TREE)
                 if isinstance(n, ast.FunctionDef)}
    capture = functions['_capture']
    assert any(arg.arg == 'armed' for arg in capture.args.args)
    start_calls = [n for n in ast.walk(capture)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'start']
    assert len(start_calls) == 1
    assert any(k.arg == 'armed' and isinstance(k.value, ast.Name)
               and k.value.id == 'armed' for k in start_calls[0].keywords)

    start = functions['start']
    start_write = next(n for n in ast.walk(start)
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == 'write')
    armed_assign = next(n for n in ast.walk(start)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Subscript)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == 'armed' for t in n.targets))
    assert start_write.lineno < armed_assign.lineno
    assert isinstance(armed_assign.value, ast.Constant)
    assert armed_assign.value.value is True

    for caller_name in ('capture_sweep_sums', 'probe_sweep_burst'):
        calls = [n for n in ast.walk(functions[caller_name])
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == '_capture']
        assert len(calls) == 1
        assert any(k.arg == 'armed' for k in calls[0].keywords)

    flush_starts = [n for n in ast.walk(functions['flush'])
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'start']
    assert len(flush_starts) == 1
    assert any(k.arg == 'armed' and isinstance(k.value, ast.Name)
               and k.value.id == 'writing'
               for k in flush_starts[0].keywords)


def test_capture_does_not_bypass_the_settle_abort():
    # looping_capture's pre-existing self.abort() is unrelated to managed
    # capture buffers. Every other spelling, including aliases/getattr, is a
    # bypass around the tested release path.
    offenders = [(_enclosing_function(n), n.lineno)
                 for n in _method_references('abort')
                 if _enclosing_function(n) != 'looping_capture']
    assert not offenders, f'abort referenced directly in capture.py: {offenders}'


def test_clear_error_never_precedes_quiescence():
    """clear_error is what happens once the core is quiet, never instead."""
    callers = {_enclosing_function(c) for c in _method_references('clear_error')}
    # `capture` clears a reported error after the fact; it arms nothing.
    assert callers <= {'_settle_axis2mm', 'capture'}, (
        f'clear_error() called outside the settle path: {sorted(callers)}')


def test_no_finally_block_frees_a_buffer_directly():
    """Releases go through _release, which checks the DMA first.

    A bare `finally: buffer.freebuffer()` is the exact shape of the bug this
    module guards: it runs on the timeout path, which is precisely when the
    transfer is still in flight.
    """
    offenders = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.finalbody:
            for call in ast.walk(stmt):
                if ((isinstance(call, ast.Attribute)
                     and call.attr == 'freebuffer')
                        or (isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Name)
                            and call.func.id == 'getattr'
                            and len(call.args) >= 2
                            and isinstance(call.args[1], ast.Constant)
                            and call.args[1].value == 'freebuffer')
                        or isinstance(call, ast.Delete)):
                    offenders.append((_enclosing_function(node), call.lineno))
    assert not offenders, (
        f'freebuffer() called directly in a finally block at {offenders}; '
        'release through self._release so the DMA is settled first')


def test_managed_capture_methods_do_not_delete_local_buffers():
    managed = {'flush', 'capture_sweep_sums', 'probe_sweep_burst'}
    offenders = [(_enclosing_function(n), n.lineno)
                 for n in ast.walk(TREE) if isinstance(n, ast.Delete)
                 and _enclosing_function(n) in managed]
    assert not offenders, f'buffer lifetime bypassed with del at {offenders}'


def test_release_is_the_only_thing_that_frees_a_capture_buffer():
    freers = {_enclosing_function(c) for c in _method_references('freebuffer')}
    # _allocate frees a buffer it just made and never armed.
    assert freers <= {'_allocate'}, (
        f'freebuffer referenced outside _allocate: {sorted(freers)}')


def _status(**updates):
    status = {'r_busy': False, 'aborting': False}
    status.update(updates)
    return status


class FakeAxis2MM:
    """Scripted register dict with an observable MMIO operation trace."""

    def __init__(self, statuses=(), failure_site=None):
        self.registers = {'cmd_ctrl': list(statuses)}
        self.failure_site = failure_site
        self.operations = []

    def abort(self):
        self.operations.append(('write', 'abort'))
        if self.failure_site == 'abort':
            raise OSError('scripted MMIO abort write failed')

    @property
    def cmd_ctrl_reg(self):
        self.operations.append(('read', 'cmd_ctrl'))
        if self.failure_site == 'read':
            raise OSError('scripted MMIO status read failed')
        return self.registers['cmd_ctrl'].pop(0)

    def clear_error(self):
        self.operations.append(('write', 'clear_error'))
        if self.failure_site == 'clear':
            raise OSError('scripted MMIO clear write failed')


class FakeBuffer:
    nbytes = 4096
    device_address = 0x1234_0000

    def __init__(self, free_failure=False):
        self.freed = False
        self.free_failure = free_failure

    def freebuffer(self):
        if self.free_failure:
            raise RuntimeError('scripted freebuffer failed')
        self.freed = True


class FakeCapture:
    """Production-shaped proxy: delegates without inheriting the helper."""

    def __init__(self, axis2mm):
        self.axis2mm = axis2mm
        self._stuck_buffers = []

    def _settle_axis2mm(self, timeout=0.5):
        return CaptureBufferRelease._settle_axis2mm(self, timeout)

    def _retain_stuck_buffer(self, capture_buffer):
        return CaptureBufferRelease._retain_stuck_buffer(self, capture_buffer)

    def _release(self, capture_buffer, writing, what, abort_timeout=0.5):
        return CaptureBufferRelease._release(
            self, capture_buffer, writing, what, abort_timeout)


def test_settle_orders_abort_poll_then_clear_error():
    port = FakeAxis2MM([_status(r_busy=True), _status()])
    capture = FakeCapture(port)

    assert capture._settle_axis2mm(timeout=1.0)
    assert port.operations == [
        ('write', 'abort'),
        ('read', 'cmd_ctrl'),
        ('read', 'cmd_ctrl'),
        ('write', 'clear_error'),
    ]


@pytest.mark.parametrize('failure_site', ['abort', 'read', 'clear', 'free'])
def test_release_never_throws_on_any_cleanup_failure(failure_site, caplog):
    port = FakeAxis2MM([_status()], failure_site=failure_site)
    capture = FakeCapture(port)
    capture_buffer = FakeBuffer(free_failure=failure_site == 'free')
    writing = failure_site != 'free'

    assert capture._release(capture_buffer, writing, 'test capture') is False
    assert capture_buffer.freed is False
    assert capture._stuck_buffers == [capture_buffer]
    assert 'scripted' in caplog.text


def test_release_failure_does_not_replace_the_inflight_capture_error():
    port = FakeAxis2MM(failure_site='read')
    capture = FakeCapture(port)
    capture_buffer = FakeBuffer()
    expected = 'the original sweep IOError must survive byte-for-byte'

    with pytest.raises(IOError, match=expected) as caught:
        try:
            raise IOError(expected)
        finally:
            capture._release(capture_buffer, True, 'test capture')
    assert str(caught.value) == expected
    assert capture._stuck_buffers == [capture_buffer]


def test_stuck_dma_retains_the_buffer_instead_of_hollow_leaking(caplog):
    port = FakeAxis2MM([_status(r_busy=True), _status(r_busy=True)])
    capture = FakeCapture(port)
    capture_buffer = FakeBuffer()

    assert capture._release(capture_buffer, True, 'test capture',
                            abort_timeout=-1) is False
    assert capture_buffer.freed is False
    assert capture._stuck_buffers == [capture_buffer]
    assert 'Restart to reclaim it' in caplog.text
