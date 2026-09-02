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


def test_analyze_vlssr_exports() -> None:
    from lwa_catalog.analyze import (
        VlssrMatchConfig,
        VlssrMatchResult,
        load_vlssr_catalog,
        match_catalog_to_vlssr,
        select_blue_associated_rows,
        summarize_vlssr_match,
    )

    assert callable(load_vlssr_catalog)
    assert callable(match_catalog_to_vlssr)
    assert callable(select_blue_associated_rows)
    assert callable(summarize_vlssr_match)
    assert VlssrMatchConfig().target == "metacatalog_blue"
    assert hasattr(VlssrMatchResult, "__dataclass_fields__")


def test_analyze_nedlvs_exports() -> None:
    from lwa_catalog.analyze import (
        NedlvsMatchConfig,
        NedlvsMatchResult,
        load_nedlvs_catalog,
        match_catalog_to_nedlvs,
        summarize_nedlvs_match,
    )

    assert callable(load_nedlvs_catalog)
    assert callable(match_catalog_to_nedlvs)
    assert callable(summarize_nedlvs_match)
    assert NedlvsMatchConfig().target == "metacatalog"
    assert hasattr(NedlvsMatchResult, "__dataclass_fields__")


def test_analyze_spectral_exports() -> None:
    from lwa_catalog.analyze import (
        SingleSpectrumFit,
        SpectralFitConfig,
        SpectralFitResult,
        fit_metacatalog_spectra,
        fit_single_spectrum,
        gather_band_flux_measurements,
        summarize_spectral_fit,
    )
    from lwa_catalog.constants import SUBBAND_REF_FREQ_MHZ

    assert callable(fit_metacatalog_spectra)
    assert callable(fit_single_spectrum)
    assert callable(gather_band_flux_measurements)
    assert callable(summarize_spectral_fit)
    assert SpectralFitConfig().flux_kind == "total"
    assert SpectralFitConfig().ref_freq_mhz == SUBBAND_REF_FREQ_MHZ
    assert SpectralFitConfig().column_prefix == "spec_"
    assert hasattr(SpectralFitResult, "__dataclass_fields__")
    assert hasattr(SingleSpectrumFit, "__dataclass_fields__")


def test_create_apis_importable() -> None:
    from lwa_catalog.create import (
        detect_sources,
        detect_sources_many,
        discover_fits_files,
        iter_detect_sources,
        merge_lst_metacatalog,
    )

    assert callable(detect_sources)
    assert callable(detect_sources_many)
    assert callable(iter_detect_sources)
    assert callable(discover_fits_files)
    assert merge_lst_metacatalog([], band="Full").empty
