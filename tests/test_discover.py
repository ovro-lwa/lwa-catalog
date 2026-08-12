"""Tests for FITS discovery metadata parsing."""

from __future__ import annotations

from pathlib import Path

from lwa_catalog.create.discover import (
    discover_fits_files,
    format_lst_hour,
    lst_hours_from_discovery,
    parse_fits_metadata,
)


def test_parse_deep_color_filename() -> None:
    path = Path("/data/01h/I_01h_deep_Taper_R0_Full.fits")
    meta = parse_fits_metadata(path)
    assert meta is not None
    assert meta.lst_hour == "01h"
    assert meta.band == "Full"
    assert meta.time_key is None


def test_parse_lst_color_filename() -> None:
    path = Path(
        "/data/Blue_I_something_20250508_LST22h_t0001.fits"
    )
    meta = parse_fits_metadata(path)
    assert meta is not None
    assert meta.lst_hour == "22h"
    assert meta.band == "Blue"
    assert meta.time_key == "20250508_LST22h_t0001"


def test_parse_band_prefix_with_parent_hour(tmp_path: Path) -> None:
    hour_dir = tmp_path / "03h"
    hour_dir.mkdir()
    path = hour_dir / "Green_I_deep.fits"
    path.write_bytes(b"")
    meta = parse_fits_metadata(path)
    assert meta is not None
    assert meta.lst_hour == "03h"
    assert meta.band == "Green"


def test_parse_unrecognized_returns_none() -> None:
    assert parse_fits_metadata(Path("/tmp/random.fits")) is None


def test_discover_and_lst_hours(tmp_path: Path) -> None:
    (tmp_path / "I_05h_deep_Taper_R0_Blue.fits").write_bytes(b"")
    (tmp_path / "I_05h_deep_Taper_R0_Full.fits").write_bytes(b"")
    (tmp_path / "noise.txt").write_text("x")
    found = discover_fits_files(tmp_path)
    assert len(found) == 2
    assert lst_hours_from_discovery(found) == ["05h"]
    assert format_lst_hour(5) == "05h"
