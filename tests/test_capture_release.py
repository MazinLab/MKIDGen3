"""Structural invariants of capture.py's buffer release path.

capture.py imports pynq at module scope and cannot be imported off-board, so
these read it as source. They are not style checks: both invariants exist
because breaking either one frees CMA that axis2mm still holds the physical
address of, which corrupts whatever is allocated next, at an arbitrary later
time, with nothing in the traceback pointing back here.
"""
import ast
from pathlib import Path

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


def _calls(name):
    """Every Call node whose callee attribute is `name`."""
    return [n for n in ast.walk(TREE)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == name]


def test_only_settle_aborts_the_dma():
    """abort() returns before the core stops; only the waiter may call it.

    _settle_axis2mm aborts and then polls until r_busy and aborting clear. An
    abort anywhere else is an abort nobody waits out.
    """
    callers = {_enclosing_function(c) for c in _calls('abort')
               if isinstance(c.func.value, ast.Attribute)
               and c.func.value.attr == 'axis2mm'}
    assert callers <= {'_settle_axis2mm'}, (
        f'self.axis2mm.abort() called outside _settle_axis2mm: {sorted(callers)}')


def test_clear_error_never_precedes_quiescence():
    """clear_error is what happens once the core is quiet, never instead."""
    callers = {_enclosing_function(c) for c in _calls('clear_error')}
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
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'freebuffer'):
                    offenders.append((_enclosing_function(node), call.lineno))
    assert not offenders, (
        f'freebuffer() called directly in a finally block at {offenders}; '
        'release through self._release so the DMA is settled first')


def test_release_is_the_only_thing_that_frees_a_capture_buffer():
    freers = {_enclosing_function(c) for c in _calls('freebuffer')}
    # _allocate frees a buffer it just made and never armed.
    assert freers <= {'_release', '_allocate'}, (
        f'freebuffer() called outside _release: {sorted(freers)}')
