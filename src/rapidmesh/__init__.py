"""
RapidMesh — spatially aligned multiresolution point-cloud mesh engine.

Read 00-PRODUCT-DEFINITION.md and SPATIAL-CONTRACT.md before this package.
Spatial truth comes first; preserve the strongest source structure, then
decimate to a measured tolerance. The current implemented front half is
structured E57; required LAS/LAZ support is not yet built.

Nothing at module scope imports numpy, scipy or pye57, so `import rapidmesh`
works on a bare interpreter and `deps` can report what is actually installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "deps",
    "mesh_station_streamed",
    "StreamedMeshingNotImplemented",
]


def __getattr__(name: str) -> object:
    """Lazy export of the streamed entry stub (avoids importing numpy at load)."""
    if name in ("mesh_station_streamed", "StreamedMeshingNotImplemented"):
        from .pipeline import StreamedMeshingNotImplemented, mesh_station_streamed

        return {
            "mesh_station_streamed": mesh_station_streamed,
            "StreamedMeshingNotImplemented": StreamedMeshingNotImplemented,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
