"""qpce.progress -- tqdm shim, so the package has no hard dependency."""
from __future__ import annotations


def progress_iter(iterable, total=None, desc="", enabled=True):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False)
    except ImportError:
        return iterable
