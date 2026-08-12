"""Smoke test mirroring the README CatalogLayout + write_metacatalog example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lwa_catalog import CatalogLayout, read_metacatalog, write_metacatalog


def test_readme_cataloglayout_write_metacatalog_example(tmp_path: Path) -> None:
    """README quick example must run against a temporary catalog root."""
    layout = CatalogLayout(tmp_path)
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [1.0],
            "origin_band": ["Full"],
            "bands_present": ["Full"],
        }
    )
    path = write_metacatalog(df, layout)
    loaded = read_metacatalog(layout)
    assert path.name == "metacatalog.parquet"
    assert path == layout.metacatalog()
    assert len(loaded) == 1
    assert float(loaded["RA"].iloc[0]) == 10.0
