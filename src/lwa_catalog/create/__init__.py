"""Build per-image catalogs and fuse them into a metacatalog.

Logic to extract from ``notebooks/ovro_lwa_metacatalog.ipynb``:

* FITS discovery (LST hour + band from paths/filenames)
* PyBDSF detection → Gaussian (``gaul``) tables
* LST merge within each band
* Sequential band merge → global metacatalog
"""

from __future__ import annotations

from lwa_catalog.create.detect import detect_sources
from lwa_catalog.create.merge import (
    associate_band_into_metacatalog,
    build_global_metacatalog,
    merge_lst_metacatalog,
)

__all__ = [
    "associate_band_into_metacatalog",
    "build_global_metacatalog",
    "detect_sources",
    "merge_lst_metacatalog",
]
