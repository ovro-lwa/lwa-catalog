"""Smoke tests for package import and public exports."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq


def test_package_version_and_exports() -> None:
    import lwa_catalog

    assert isinstance(lwa_catalog.__version__, str)
    assert lwa_catalog.__version__
    assert hasattr(lwa_catalog, "CatalogLayout")
    assert hasattr(lwa_catalog, "read_metacatalog")
    assert hasattr(lwa_catalog, "write_metacatalog")


def test_pyarrow_parquet_importable() -> None:
    assert hasattr(pq, "read_table")
    assert hasattr(pq, "write_table")


def test_analyze_summary() -> None:
    from lwa_catalog.analyze import bands_present_counts, summarize_metacatalog

    catalog = pd.DataFrame(
        {
            "RA": [1.0, 2.0],
            "DEC": [0.0, 1.0],
            "Peak_flux": [10.0, 5.0],
            "origin_band": ["Full", "Blue"],
            "bands_present": ["Full,Blue", "Blue"],
        }
    )
    summary = summarize_metacatalog(catalog)
    assert summary["n_sources"] == 2
    counts = bands_present_counts(catalog)
    assert int(counts["Blue"]) == 2
    assert int(counts["Full"]) == 1


def test_create_apis_importable() -> None:
    from lwa_catalog.create import (
        detect_sources,
        discover_fits_files,
        merge_lst_metacatalog,
    )

    assert callable(detect_sources)
    assert callable(discover_fits_files)
    assert merge_lst_metacatalog([], band="Full").empty
