"""Tests for empty sources tables and legacy CSV/FITS migration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.io import (
    empty_sources_table,
    import_legacy_metacatalog,
    import_legacy_sources_csv,
    migrate_output_dir,
    read_metacatalog,
    read_sources_catalog,
    validate_sources_catalog,
)
from lwa_catalog.paths import CatalogLayout
from lwa_catalog.schemas import sources_schema


def test_empty_sources_table_schema_and_metadata() -> None:
    table = empty_sources_table(
        "01h",
        "Full",
        source_file="img.fits",
        bmaj=0.1,
        bmin=0.05,
        bpa=0.0,
    )
    assert table.num_rows == 0
    assert set(sources_schema().names) <= set(table.column_names)
    validate_sources_catalog(table)
    meta = table.schema.metadata or {}
    assert meta[b"lst_hour"] == b"01h"
    assert meta[b"band"] == b"Full"
    assert meta[b"BMAJ"] == b"0.1"


def test_validate_sources_catalog_missing_ra() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_sources_catalog(pd.DataFrame({"DEC": [1.0]}))


def test_import_legacy_sources_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "sources_01h_Full.csv"
    pd.DataFrame(
        {
            "RA": [10.0, 11.0],
            "DEC": [20.0, 21.0],
            "Peak_flux": [1.0, 2.0],
            "lst_hour": ["01h", "01h"],
            "band": ["Full", "Full"],
        }
    ).to_csv(csv_path, index=False)
    table = import_legacy_sources_csv(csv_path)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    validate_sources_catalog(table)
    assert "BMAJ" in table.column_names


def test_import_legacy_metacatalog_prefers_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "metacatalog.csv"
    fits_path = tmp_path / "metacatalog.fits"
    csv_df = pd.DataFrame(
        {
            "RA": [1.0],
            "DEC": [2.0],
            "Peak_flux": [3.0],
            "origin_band": ["Full"],
            "bands_present": ["Full"],
        }
    )
    csv_df.to_csv(csv_path, index=False)
    fits_df = csv_df.copy()
    fits_df["RA"] = [99.0]
    Table.from_pandas(fits_df).write(fits_path, overwrite=True)

    table = import_legacy_metacatalog(csv_path=csv_path, fits_path=fits_path)
    assert table.num_rows == 1
    assert float(table.column("RA")[0].as_py()) == pytest.approx(1.0)


def test_import_legacy_metacatalog_fits_only(tmp_path: Path) -> None:
    fits_path = tmp_path / "metacatalog.fits"
    Table.from_pandas(
        pd.DataFrame(
            {
                "RA": [5.0],
                "DEC": [6.0],
                "Peak_flux": [7.0],
                "origin_band": ["Blue"],
                "bands_present": ["Blue"],
            }
        )
    ).write(fits_path, overwrite=True)
    table = import_legacy_metacatalog(fits_path=fits_path)
    assert float(table.column("RA")[0].as_py()) == pytest.approx(5.0)


def test_migrate_output_dir_dry_run_and_write(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    (tmp_path / "sources_01h_Full.csv").write_text(
        "RA,DEC,Peak_flux,lst_hour,band\n1.0,2.0,3.0,01h,Full\n"
    )
    (tmp_path / "sources_01h_Blue.csv").write_text(
        "RA,DEC,Peak_flux,lst_hour,band\n1.1,2.1,3.1,01h,Blue\n"
    )
    (tmp_path / "metacatalog_lst_Full.csv").write_text(
        "RA,DEC,Peak_flux,band,n_lst_contributions,lst_hours,representative_lst\n"
        "1.0,2.0,3.0,Full,1,01h,01h\n"
    )
    pd.DataFrame(
        {
            "RA": [1.0],
            "DEC": [2.0],
            "Peak_flux": [3.0],
            "origin_band": ["Full"],
            "bands_present": ["Full"],
        }
    ).to_csv(tmp_path / "metacatalog.csv", index=False)

    planned = migrate_output_dir(layout, dry_run=True)
    kinds = {item.kind for item in planned}
    assert kinds == {"sources", "lst_merged", "metacatalog"}
    assert all(not item.destination.is_file() for item in planned)

    # Optional beam FITS for sources backfill
    header = fits.Header({"BMAJ": 0.2, "BMIN": 0.1, "BPA": 0.0})
    fits_path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32), header=header).writeto(fits_path)

    done = migrate_output_dir(
        layout,
        fits_paths={("01h", "Full"): fits_path},
        dry_run=False,
    )
    assert len(done) == len(planned)
    assert layout.sources("01h", "Full").is_file()
    assert layout.lst_merged("Full").is_file()
    assert layout.metacatalog().is_file()

    sources = read_sources_catalog(layout, "01h", "Full")
    assert isinstance(sources, pd.DataFrame)
    assert float(sources.iloc[0]["BMAJ"]) == pytest.approx(0.2)
    meta = read_metacatalog(layout)
    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == 1
    # Legacy CSV kept by default
    assert (tmp_path / "metacatalog.csv").is_file()


def test_migrate_delete_legacy(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    csv_path = tmp_path / "sources_02h_Green.csv"
    csv_path.write_text("RA,DEC\n1.0,2.0\n")
    migrate_output_dir(layout, delete_legacy=True)
    assert layout.sources("02h", "Green").is_file()
    assert not csv_path.is_file()
