"""Ensure editable PyNAMod imports resolve in a dev checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_pynamod_importable() -> None:
    pkg_init = Path(__file__).resolve()
    if pkg_init.parent.name != 'base2backbone' or pkg_init.parent.parent.name != 'src':
        return
    repo_root = pkg_init.parent.parent.parent
    pynamod_root = repo_root / 'pynamod'
    if not (pynamod_root / 'pynamod' / '__init__.py').is_file():
        return
    pynamod_path = str(pynamod_root)
    if pynamod_path not in sys.path:
        sys.path.insert(0, pynamod_path)
    cached = sys.modules.get('pynamod')
    if cached is not None and getattr(cached, '__file__', None) is None:
        del sys.modules['pynamod']
