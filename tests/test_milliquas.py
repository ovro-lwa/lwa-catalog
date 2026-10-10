"""Tests for MilliQUAS optical AGN/quasar cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    MILLIQUAS_REFERENCE_RADIUS_FIXED,
)
from lwa_catalog.analyze.milliquas import (
    MilliquasMatchConfig,
    attach_milliquas_to_metacatalog,
    filter_milliquas_by_class,
    load_milliquas_catalog,
    match_catalog_to_milliquas,
    milliquas_class_counts,
    parse_milliquas_type,
    select_unique_milliquas_matches,
    summarize_milliquas_match,
)


def _write_mini_milliquas_fits(path) -> None:
    table = Table(
        {
            "RAdeg": [180.0, 180.05, 10.0],
            "DEdeg": [40.0, 40.0, 0.0],
            "Name": [b"QSO J120000+400000", b"AGN J120012+400000", b"BL J004000+000000"],
            "Type": [b"QR2X", b"A", b"BX"],
            "Rmag": [18.0, 17.5, 19.0],
            "Bmag": [18.5, 18.0, 19.5],
            "Comment": [b"G  ", b"g  ", b"N  "],
            "R": [b"-", b"1", b"n"],
            "B": [b"-", b"1", b"n"],
            "z": [1.2, 0.05, 0.3],
            "rName": [b"ref1", b"ref2", b"ref3"],
            "rz": [b"z1", b"z2", b"z3"],
            "XName": [b"2CXO J1", b"", b""],
            "RName": [b"NVSS J1", b"", b"NVSS J2"],
            "Lobe1": [b"lobeA", b"", b""],
            "Lobe2": [b"lobeB", b"", b""],
        }
    )
    table.write(path, format="fits", overwrite=True)


def _meta_row(*, meta_id: int, ra: float, dec: float, bmaj: float = 0.2) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "73MHz",
        "Peak_flux": 1.0,
        "origin_band": "73MHz",
    }


def test_parse_milliquas_type() -> None:
    parsed = parse_milliquas_type("QR2X")
    assert parsed["class_primary"] == "Q"
    assert "QSO" in str(parsed["class_label"])
    assert parsed["has_radio"] is True
    assert parsed["has_xray"] is True
    assert parsed["has_double_lobes"] is True

    bl = parse_milliquas_type("B")
    assert bl["class_primary"] == "B"
    assert bl["has_radio"] is False


def test_load_milliquas_catalog_fits(tmp_path) -> None:
    path = tmp_path / "milliquas.dat"
    _write_mini_milliquas_fits(path)
    df = load_milliquas_catalog(path)
    assert len(df) == 3
    assert "RA" in df.columns and "DEC" in df.columns
    assert df["class_primary"].tolist() == ["Q", "A", "B"]
    assert bool(df.loc[0, "has_radio"]) is True
    assert bool(df.loc[0, "has_double_lobes"]) is True
    assert bool(df.loc[1, "has_radio"]) is False
    assert "QSO" in df.loc[0, "class_label"]


def test_load_milliquas_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="MilliQUAS catalog not found"):
        load_milliquas_catalog(tmp_path / "missing.dat")


def test_filter_milliquas_by_class(tmp_path) -> None:
    path = tmp_path / "milliquas.dat"
    _write_mini_milliquas_fits(path)
    df = load_milliquas_catalog(path)
    qs = filter_milliquas_by_class(df, ("Q", "B"))
    assert len(qs) == 2
    assert set(qs["class_primary"]) == {"Q", "B"}


def test_match_and_attach_milliquas(tmp_path) -> None:
    path = tmp_path / "milliquas.dat"
    _write_mini_milliquas_fits(path)
    ref = load_milliquas_catalog(path)
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=180.0, dec=40.0),
            _meta_row(meta_id=2, ra=10.0, dec=0.0),
            _meta_row(meta_id=3, ra=0.0, dec=-80.0),
        ]
    )
    result = match_catalog_to_milliquas(
        meta,
        milliquas=ref,
        config=MilliquasMatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=MILLIQUAS_REFERENCE_RADIUS_FIXED,
        ),
    )
    assert result.summary["n_milliquas_footprint"] == 3
    assert int(result.meta_flags.loc[0, "n_milliquas"]) >= 1
    assert int(result.meta_flags.loc[2, "n_milliquas"]) == 0
    unique = select_unique_milliquas_matches(result.meta_flags)
    assert len(unique) >= 1
    text = summarize_milliquas_match(result)
    assert "Meta matched" in text

    attached = attach_milliquas_to_metacatalog(meta, result.milliquas_footprint, result.meta_flags)
    assert len(attached) == len(meta)
    assert "MilliQUAS_Type" in attached.columns
    assert "MilliQUAS_class_primary" in attached.columns
    assert "MilliQUAS_class_label" in attached.columns
    assert attached.loc[0, "MilliQUAS_class_primary"] in {"Q", "A"}
    counts = milliquas_class_counts(attached.loc[attached["n_milliquas"] >= 1])
    assert counts.sum() >= 1
