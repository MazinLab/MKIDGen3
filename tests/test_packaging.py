"""The wheel is the only supported way this package reaches the board.

The board's pynq venv carries setuptools 59.6.0, which predates PEP 621: a
board-side build of this source tree produces an empty ``UNKNOWN-0.0.0``
wheel while pip reports success, and the old mkidgen3 stays installed. So the
wheel is always built here and installed there -- and this test is what says
the wheel built here is a real one.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Modules that must be inside the wheel. triggerv2 hard-imports recordfmt, so
# a wheel missing it breaks the board at import, not at first use.
REQUIRED_MEMBERS = (
    'mkidgen3/recordfmt.py',
    'mkidgen3/drivers/iqtransform.py',
    'mkidgen3/drivers/triggerv2.py',
    'mkidgen3/drivers/phasematch.py',
    'mkidgen3/drivers/capture.py',
)


@pytest.mark.slow
def test_wheel_builds_and_carries_the_drivers(tmp_path):
    pip = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                         capture_output=True, text=True)
    if pip.returncode:
        pytest.skip(f'pip is unavailable: {pip.stdout}{pip.stderr}')

    wheel_dir = tmp_path / 'wheel'
    wheel_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '--no-deps',
         '--no-build-isolation', '-w', str(wheel_dir), str(REPO)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = sorted(wheel_dir.glob('*.whl'))
    assert len(wheels) == 1, f'expected exactly one wheel, got {wheels}'
    name = wheels[0].name
    assert name.startswith('mkidgen3-'), (
        f'{name}: not a mkidgen3 wheel -- a name like UNKNOWN-0.0.0 means the '
        'build backend fell back to legacy setuptools and the wheel is empty')
    with zipfile.ZipFile(wheels[0]) as z:
        members = set(z.namelist())
    for m in REQUIRED_MEMBERS:
        assert m in members, f'{m} missing from {name}'

    # Inspecting zip members is weaker than exercising Python's real import
    # machinery. Install without a venv or dependencies, then run isolated
    # from the checkout so an editable/source-tree import cannot mask a
    # missing module in the wheel.
    target = tmp_path / 'target'
    install = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
         str(target), str(wheels[0])], capture_output=True, text=True)
    assert install.returncode == 0, install.stdout + install.stderr
    isolated_cwd = tmp_path / 'outside-checkout'
    isolated_cwd.mkdir()
    modules = ('mkidgen3.recordfmt', 'mkidgen3.mkidpynq',
               'mkidgen3.drivers.iqtransform', 'mkidgen3.drivers.triggerv2')
    script = (
        'import importlib, pathlib, sys\n'
        f'target = pathlib.Path({str(target)!r}).resolve()\n'
        'sys.path.insert(0, str(target))\n'
        f'modules = {modules!r}\n'
        'for name in modules:\n'
        '    module = importlib.import_module(name)\n'
        '    assert pathlib.Path(module.__file__).resolve().is_relative_to(target), '
        '(name, module.__file__)\n')
    imported = subprocess.run(
        [sys.executable, '-I', '-c', script], cwd=isolated_cwd,
        capture_output=True, text=True)
    assert imported.returncode == 0, imported.stdout + imported.stderr
