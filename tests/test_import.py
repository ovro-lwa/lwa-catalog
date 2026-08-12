"""Smoke tests for package import and stub API surface."""

from __future__ import annotations

import pandas as pd
import pytest


def test_package_version() -> None:
    import lwa_catalog

    assert isinstance(lwa_catalog.__version__, str)
    assert lwa_catalog.__version__


def test_io_roundtrip_csv(tmp_path) -> None:
    from lwa_catalog.io import read_catalog, validate_metacatalog, write_catalog

    catalog = pd.DataFrame(
        {
            "RA": [123.4],
            "DEC": [45.6],
            "Peak_flux": [1.0],
            "origin_band": ["Full"],
            "bands_present": ["Full,Blue"],
        }
    )
    path = tmp_path / "meta.csv"
    write_catalog(catalog, path)
    loaded = read_catalog(path)
    validate_metacatalog(loaded)
    assert len(loaded) == 1


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


def test_create_stubs_raise() -> None:
    from lwa_catalog.create import detect_sources, merge_lst_metacatalog

    with pytest.raises(NotImplementedError):
        detect_sources("dummy.fits")
    with pytest.raises(NotImplementedError):
        merge_lst_metacatalog([], band="Full")
