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
    proc = subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '--no-deps',
         '--no-build-isolation', '-w', str(tmp_path), str(REPO)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = sorted(tmp_path.glob('*.whl'))
    assert len(wheels) == 1, f'expected exactly one wheel, got {wheels}'
    name = wheels[0].name
    assert name.startswith('mkidgen3-'), (
        f'{name}: not a mkidgen3 wheel -- a name like UNKNOWN-0.0.0 means the '
        'build backend fell back to legacy setuptools and the wheel is empty')
    with zipfile.ZipFile(wheels[0]) as z:
        members = set(z.namelist())
    for m in REQUIRED_MEMBERS:
        assert m in members, f'{m} missing from {name}'
