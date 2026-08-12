"""OVRO-LWA catalog creation and analysis."""

from __future__ import annotations

try:
    from lwa_catalog._version import __version__
except ImportError:  # pragma: no cover - editable / missing hatch-vcs build
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
