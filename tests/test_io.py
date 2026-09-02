"""Tests for Parquet catalog I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from astropy.io import fits

from lwa_catalog.constants import COLOR_BANDS, DEFAULT_QUALITY_FLAG_MASK
from lwa_catalog.io import (
    discover_lst_merged_bands,
    ensure_beam_columns,
    lst_merged_cache_complete,
    read_all_lst_merged,
    read_beam_from_fits,
    read_lst_merged,
    read_metacatalog,
    read_sources_catalog,
    read_table,
    resolve_metacatalog_path,
    rewrite_output_dir_gaul_columns,
    sources_cache_complete,
    validate_metacatalog,
    write_lst_merged,
    write_metacatalog,
    write_sources_catalog,
    write_table,
)
from lwa_catalog.paths import CatalogLayout
from lwa_catalog.schemas import sources_schema


def _minimal_metacatalog_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RA": [123.4],
            "DEC": [45.6],
            "Peak_flux": [1.0],
            "origin_band": ["Full"],
            "bands_present": ["Full,Blue"],
        }
    )


def _write_beam_fits(
    path: Path, *, bmaj: float = 0.1, bmin: float = 0.05, bpa: float = 12.0
) -> Path:
    header = fits.Header()
    header["BMAJ"] = bmaj
    header["BMIN"] = bmin
    header["BPA"] = bpa
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=header).writeto(path, overwrite=True)
    return path


def test_write_read_sources_roundtrip_and_empty_schema(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    empty = pd.DataFrame(columns=["RA", "DEC", "Peak_flux", "lst_hour", "band"])
    write_sources_catalog(empty, layout, "01h", "Full")
    loaded = read_sources_catalog(layout, "01h", "Full", as_pandas=False)
    assert isinstance(loaded, pa.Table)
    assert loaded.num_rows == 0
    assert set(sources_schema().names) <= set(loaded.column_names)

    rows = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [3.5],
            "lst_hour": ["01h"],
            "band": ["Full"],
            "source_file": ["img.fits"],
            "BMAJ": [0.2],
            "BMIN": [0.1],
            "BPA": [0.0],
        }
    )
    write_sources_catalog(rows, layout, "01h", "Full")
    out = read_sources_catalog(layout, "01h", "Full")
    assert isinstance(out, pd.DataFrame)
    assert float(out.iloc[0]["RA"]) == pytest.approx(10.0)
    assert out.iloc[0]["band"] == "Full"


def test_beam_backfill_from_fits(tmp_path: Path) -> None:
    fits_path = _write_beam_fits(tmp_path / "img.fits", bmaj=0.3, bmin=0.2, bpa=45.0)
    layout = CatalogLayout(tmp_path)
    df = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "Peak_flux": [1.0]})
    write_sources_catalog(df, layout, "02h", "Blue")
    loaded = read_sources_catalog(layout, "02h", "Blue", fits_path=fits_path)
    assert isinstance(loaded, pd.DataFrame)
    assert float(loaded.iloc[0]["BMAJ"]) == pytest.approx(0.3)
    assert float(loaded.iloc[0]["BMIN"]) == pytest.approx(0.2)
    assert float(loaded.iloc[0]["BPA"]) == pytest.approx(45.0)

    bmaj, bmin, bpa = read_beam_from_fits(fits_path)
    assert bmaj == pytest.approx(0.3)
    filled = ensure_beam_columns(pd.DataFrame({"RA": [0.0]}), fits_path=fits_path)
    assert isinstance(filled, pd.DataFrame)
    assert float(filled.iloc[0]["BMAJ"]) == pytest.approx(bmaj)


def test_lst_merged_and_cache_gates(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    slots = [("01h", "Full"), ("01h", "Blue")]
    assert sources_cache_complete(layout, slots) is False
    write_sources_catalog(pd.DataFrame({"RA": [1.0], "DEC": [2.0]}), layout, "01h", "Full")
    write_sources_catalog(pd.DataFrame({"RA": [1.1], "DEC": [2.1]}), layout, "01h", "Blue")
    assert sources_cache_complete(layout, slots) is True

    assert lst_merged_cache_complete(layout) is False
    for band in COLOR_BANDS:
        write_lst_merged(
            pd.DataFrame(
                {
                    "RA": [1.0],
                    "DEC": [2.0],
                    "Peak_flux": [1.0],
                    "band": [band],
                    "n_lst_contributions": [1],
                    "lst_hours": ["01h"],
                    "representative_lst": ["01h"],
                }
            ),
            layout,
            band,
        )
    assert lst_merged_cache_complete(layout) is True
    all_bands = read_all_lst_merged(layout)
    assert set(all_bands) == set(COLOR_BANDS)
    one = read_lst_merged(layout, "Full")
    assert isinstance(one, pd.DataFrame)
    assert len(one) == 1


def test_discover_lst_merged_bands_rgb_and_frequency(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    assert discover_lst_merged_bands(layout) == COLOR_BANDS

    for band in ("Full", "Blue"):
        write_lst_merged(
            pd.DataFrame(
                {
                    "RA": [1.0],
                    "DEC": [2.0],
                    "Peak_flux": [1.0],
                    "band": [band],
                    "n_lst_contributions": [1],
                    "lst_hours": ["01h"],
                    "representative_lst": ["01h"],
                }
            ),
            layout,
            band,
        )
    assert discover_lst_merged_bands(layout) == ("Full", "Blue")

    sub = tmp_path / "subband"
    sub.mkdir()
    sub_layout = CatalogLayout(sub)
    for band in ("82MHz", "18MHz", "55MHz"):
        write_lst_merged(
            pd.DataFrame(
                {
                    "RA": [1.0],
                    "DEC": [2.0],
                    "Peak_flux": [1.0],
                    "band": [band],
                    "n_lst_contributions": [1],
                    "lst_hours": ["01h"],
                    "representative_lst": ["01h"],
                }
            ),
            sub_layout,
            band,
        )
    assert discover_lst_merged_bands(sub_layout) == ("18MHz", "55MHz", "82MHz")


def test_write_subband_metacatalog_does_not_inject_primary_flux(tmp_path: Path) -> None:
    from lwa_catalog.constants import SUBBAND_METACATALOG_REQUIRED_COLUMNS
    from lwa_catalog.create.merge import build_subband_metacatalog

    layout = CatalogLayout(tmp_path)
    catalogs = {
        "82MHz": pd.DataFrame(
            [
                {
                    "RA": 10.0,
                    "DEC": 20.0,
                    "Peak_flux": 2.0,
                    "Total_flux": 2.0,
                    "E_Peak_flux": 0.2,
                    "E_Total_flux": 0.2,
                    "Maj": 0.1,
                    "Min": 0.05,
                    "PA": 0.0,
                    "DC_Maj": 0.1,
                    "DC_Min": 0.05,
                    "DC_PA": 0.0,
                    "BMAJ": 0.5,
                    "lst_hour": "01h",
                    "band": "82MHz",
                    "source_file": "82MHz_01h.fits",
                    "n_lst_contributions": 1,
                    "lst_hours": "01h",
                    "representative_lst": "01h",
                }
            ]
        ),
        "18MHz": pd.DataFrame([]),
    }
    meta = build_subband_metacatalog(
        catalogs,
        seed_band="82MHz",
        assoc_bands=("18MHz",),
        color_bands=("82MHz", "18MHz"),
        band_freq_hz={"82MHz": 82e6, "18MHz": 18e6},
    )
    write_metacatalog(
        meta,
        layout,
        required=SUBBAND_METACATALOG_REQUIRED_COLUMNS,
        schema=None,
    )
    loaded = read_metacatalog(layout, validate=False)
    assert "Peak_flux" not in loaded.columns
    assert "Total_flux" not in loaded.columns
    assert "Peak_flux_82MHz" in loaded.columns


def test_metacatalog_write_read_preserves_string_and_float(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    catalog = _minimal_metacatalog_df()
    path = write_metacatalog(catalog, layout)
    assert path.name == "metacatalog.parquet"
    loaded = read_metacatalog(layout)
    assert isinstance(loaded, pd.DataFrame)
    assert float(loaded.iloc[0]["RA"]) == pytest.approx(123.4)
    assert loaded.iloc[0]["bands_present"] == "Full,Blue"
    as_table = read_metacatalog(layout.metacatalog(), as_pandas=False)
    assert isinstance(as_table, pa.Table)
    validate_metacatalog(as_table)


def test_validate_metacatalog_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_metacatalog(pd.DataFrame({"RA": [1.0]}))


def test_read_missing_raises(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_sources_catalog(layout, "00h", "Full")
    with pytest.raises(FileNotFoundError):
        read_metacatalog(layout)


def test_read_metacatalog_prefers_quality_and_default_mask(tmp_path: Path) -> None:
    from lwa_catalog.analyze.reliability import SourceQualityFlag

    layout = CatalogLayout(tmp_path)
    fusion = pd.DataFrame(
        {
            "meta_id": [0, 1],
            "RA": [123.4, 124.0],
            "DEC": [45.6, 46.0],
            "Peak_flux": [1.0, 2.0],
            "origin_band": ["Full", "Blue"],
            "bands_present": ["Full,Blue", "Blue"],
        }
    )
    write_metacatalog(fusion, layout)

    quality = fusion.copy()
    quality["quality_flag"] = np.uint32(
        [
            0,
            int(SourceQualityFlag.SINGLE_LST),
        ]
    )
    write_table(quality, layout.metacatalog_quality())

    assert resolve_metacatalog_path(layout) == layout.metacatalog_quality()
    loaded = read_metacatalog(layout)
    assert len(loaded) == 1
    assert int(loaded.iloc[0]["meta_id"]) == 0

    unfiltered = read_metacatalog(layout, quality_mask=None)
    assert len(unfiltered) == 2

    fusion_only = read_metacatalog(
        layout,
        prefer_quality=False,
        quality_mask=None,
    )
    assert len(fusion_only) == 2
    assert "quality_flag" not in fusion_only.columns

    assert DEFAULT_QUALITY_FLAG_MASK == 247


def test_read_metacatalog_prefers_spectral(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    fusion = pd.DataFrame(
        {
            "meta_id": [0],
            "RA": [123.4],
            "DEC": [45.6],
            "Peak_flux": [1.0],
            "origin_band": ["Full"],
            "bands_present": ["Full"],
        }
    )
    write_metacatalog(fusion, layout)

    spectral = fusion.copy()
    spectral["spec_model_n_terms"] = np.int32(2)
    spectral["spec_model_a0"] = np.float64(0.0)
    write_table(spectral, layout.metacatalog_spectral())

    assert resolve_metacatalog_path(layout) == layout.metacatalog()
    assert resolve_metacatalog_path(layout, prefer_spectral=True) == layout.metacatalog_spectral()

    loaded = read_metacatalog(layout, prefer_spectral=True, quality_mask=None)
    assert "spec_model_n_terms" in loaded.columns
    assert int(loaded.iloc[0]["spec_model_n_terms"]) == 2

    missing = read_metacatalog(
        layout,
        prefer_spectral=True,
        prefer_quality=False,
        quality_mask=None,
    )
    assert "spec_model_n_terms" in missing.columns

    layout.metacatalog_spectral().unlink()
    fallback = read_metacatalog(
        layout,
        prefer_spectral=True,
        prefer_quality=False,
        quality_mask=None,
    )
    assert "spec_model_n_terms" not in fallback.columns


def test_write_table_rejects_csv_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Parquet"):
        write_table(pd.DataFrame({"RA": [1.0]}), tmp_path / "x.csv")
    with pytest.raises(ValueError, match="Parquet"):
        read_table(tmp_path / "x.fits")


def test_rewrite_output_dir_gaul_columns_drops_retired_fields(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    rows = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [1.0],
            "E_RA": [0.01],
            "S_Code": ["S"],
            "Source_id": [7],
            "keep_me": ["x"],
            "n_lst_contributions": [1],
            "lst_hours": ["01h"],
            "representative_lst": ["01h"],
        }
    )
    write_sources_catalog(rows, layout, "01h", "Full")
    write_lst_merged(rows, layout, "Full")
    planned = rewrite_output_dir_gaul_columns(layout, dry_run=True)
    assert layout.sources("01h", "Full") in planned
    assert layout.lst_merged("Full") in planned

    done = rewrite_output_dir_gaul_columns(layout)
    assert set(done) == set(planned)

    sources = read_sources_catalog(layout, "01h", "Full")
    assert isinstance(sources, pd.DataFrame)
    assert "E_RA" not in sources.columns
    assert "Source_id" not in sources.columns
    assert sources.iloc[0]["S_Code"] == "S"
    assert sources.iloc[0]["keep_me"] == "x"

    lst = read_lst_merged(layout, "Full")
    assert isinstance(lst, pd.DataFrame)
    assert "E_RA" not in lst.columns
    assert lst.iloc[0]["S_Code"] == "S"
    assert rewrite_output_dir_gaul_columns(layout) == []
