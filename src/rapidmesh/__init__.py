"""
RapidMesh — survey-grade per-station mesh engine for terrestrial laser scans.

Read 00-PRODUCT-DEFINITION.md before this package. The one decision everything
here follows from: mesh at the scanner's native lattice, then decimate to a
measured tolerance. Never decimate first.

Nothing at module scope imports numpy, scipy or pye57, so `import rapidmesh`
works on a bare interpreter and `deps` can report what is actually installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "deps",
]


def deps() -> dict[str, bool]:
    """Which optional dependencies are importable right now.

    Returned rather than raised so a caller (Cairn, eventually) can degrade
    gracefully and tell an operator exactly what to install, instead of
    dying on an ImportError at module load.
    """
    found: dict[str, bool] = {}
    for name in ("numpy", "scipy", "pye57"):
        try:
            __import__(name)
            found[name] = True
        except ImportError:
            found[name] = False
    return found
