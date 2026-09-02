"""Tests for CatalogLayout path naming."""

from __future__ import annotations

from pathlib import Path

from lwa_catalog.paths import CatalogLayout


def test_catalog_layout_parquet_names(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    assert layout.sources("01h", "Full") == tmp_path / "sources_01h_Full.parquet"
    assert layout.lst_merged("Blue") == tmp_path / "metacatalog_lst_Blue.parquet"
    assert layout.metacatalog() == tmp_path / "metacatalog.parquet"
    assert layout.metacatalog_parquet() == layout.metacatalog()


def test_catalog_layout_root_coerced_to_path() -> None:
    layout = CatalogLayout("/tmp/catalogs")
    assert isinstance(layout.root, Path)


def test_catalog_layout_custom_metacatalog_file(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path, metacatalog_file="metacatalog_spectral.parquet")
    assert layout.metacatalog() == tmp_path / "metacatalog_spectral.parquet"


def test_catalog_layout_with_metacatalog(tmp_path: Path) -> None:
    base = CatalogLayout(tmp_path)
    spectral = base.with_metacatalog("metacatalog_spectral.parquet")
    assert base.metacatalog() == tmp_path / "metacatalog.parquet"
    assert spectral.metacatalog() == tmp_path / "metacatalog_spectral.parquet"
    assert spectral.root == base.root


def test_catalog_layout_absolute_metacatalog_path(tmp_path: Path) -> None:
    alt = tmp_path / "derived" / "custom_meta.parquet"
    layout = CatalogLayout(tmp_path, metacatalog_file=alt)
    assert layout.metacatalog() == alt


def test_catalog_layout_metacatalog_quality_path(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    assert layout.metacatalog_quality() == tmp_path / "metacatalog_quality.parquet"


def test_catalog_layout_metacatalog_spectral_path(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    assert layout.metacatalog_spectral() == tmp_path / "metacatalog_spectral.parquet"
