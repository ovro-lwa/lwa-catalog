"""Tests for Radio Fundamental Catalogue (RFC) cross-match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.associations import (
    ASSOCIATION_SPEC_RFC,
    association_rows_from_match,
    harmonize_class_native,
)
from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.analyze.rfc import (
    RfcMatchConfig,
    load_rfc_catalog,
    load_rfc_cross_match,
    match_catalog_to_rfc,
    rfc_type_counts,
    select_unique_rfc_matches,
    summarize_rfc_match,
)


def _write_mini_rfc_cat(path: Path) -> None:
    """Two fixed-width RFC rows near (10°, +25°) and one far south."""
    # Use decimal RA/DEC via the parquet path for unit tests; write CSV-like parquet.
    df = pd.DataFrame(
        {
            "Name": ["RFC J0040+2500", "RFC J0040+2501", "RFC J0040-4000"],
            "Comnam": ["0040+250", "0040+251", "0040-400"],
            "RA": [10.0, 10.01, 10.0],
            "DEC": [25.0, 25.0, -40.0],
            "E_RA": [1e-6, 1e-6, 1e-6],
            "E_DEC": [1e-6, 1e-6, 1e-6],
            "FsS": [0.5, 0.2, 1.0],
            "FsX": [0.3, 0.1, 0.8],
        }
    )
    df.to_parquet(path, index=False)


def _write_mini_rfc_xm(path: Path) -> None:
    pd.DataFrame(
        {
            "Name": ["RFC J0040+2500", "RFC J0040+2501", "RFC J0040-4000"],
            "type": ["QSO", "G", "BLLac"],
            "z": [1.2, np.nan, 0.3],
            "PA_jet": [-40.0, np.nan, 12.0],
        }
    ).to_csv(path, index=False)


def _meta_row(
    *,
    meta_id: int = 0,
    ra: float = 10.0,
    dec: float = 25.0,
    bmaj: float = 0.5,
) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "73MHz",
        "Peak_flux": 1.0,
        "origin_band": "73MHz",
    }


def test_load_rfc_catalog_parquet_with_cross_match(tmp_path) -> None:
    cat = tmp_path / "rfc.parquet"
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_cat(cat)
    _write_mini_rfc_xm(xm)
    df = load_rfc_catalog(cat, cross_match_path=xm)
    assert len(df) == 3
    assert set(["RA", "DEC", "Name", "E_RA", "E_DEC", "type"]).issubset(df.columns)
    assert df.loc[df["Name"] == "RFC J0040+2500", "type"].iloc[0] == "QSO"
    assert df.loc[df["Name"] == "RFC J0040+2501", "type"].iloc[0] == "G"


def test_load_rfc_cross_match(tmp_path) -> None:
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_xm(xm)
    df = load_rfc_cross_match(xm)
    assert len(df) == 3
    assert "QSO" in set(df["type"])


def test_load_rfc_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="RFC catalog not found"):
        load_rfc_catalog(tmp_path / "missing.txt")


def test_match_catalog_to_rfc_unique_hit(tmp_path) -> None:
    cat = tmp_path / "rfc.parquet"
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_cat(cat)
    _write_mini_rfc_xm(xm)
    rfc = load_rfc_catalog(cat, cross_match_path=xm)
    # Keep only the exact (10, 25) source.
    rfc = rfc.loc[rfc["Name"] == "RFC J0040+2500"].reset_index(drop=True)
    meta = pd.DataFrame([_meta_row(meta_id=7, ra=10.0, dec=25.0, bmaj=0.5)])
    result = match_catalog_to_rfc(
        meta,
        rfc=rfc,
        config=RfcMatchConfig(
            catalog_path=cat,
            cross_match_path=xm,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    assert int(result.summary["n_meta_matched"]) == 1
    assert int(result.meta_flags.iloc[0]["n_rfc"]) == 1
    assert int(result.summary["n_unique_with_type"]) == 1
    unique = select_unique_rfc_matches(result.meta_flags)
    assert len(unique) == 1
    text = summarize_rfc_match(result)
    assert "RFC" in text
    counts = rfc_type_counts(result.rfc_footprint)
    assert int(counts.get("QSO", 0)) == 1


def test_match_respects_lwa_match_frame(tmp_path) -> None:
    cat = tmp_path / "rfc.parquet"
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_cat(cat)
    _write_mini_rfc_xm(xm)
    rfc = load_rfc_catalog(cat, cross_match_path=xm)
    meta = pd.DataFrame([_meta_row(meta_id=1, ra=10.0, dec=25.0, bmaj=0.5)])
    lwa_match = pd.DataFrame(
        {"RA": [180.0], "DEC": [25.0], "BMAJ": [0.01]},
        index=meta.index,
    )
    result = match_catalog_to_rfc(
        meta,
        rfc=rfc,
        config=RfcMatchConfig(catalog_path=cat, cross_match_path=xm),
        lwa_match=lwa_match,
    )
    assert int(result.summary["n_meta_matched"]) == 0


def test_association_rows_and_harmonize(tmp_path) -> None:
    cat = tmp_path / "rfc.parquet"
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_cat(cat)
    _write_mini_rfc_xm(xm)
    rfc = load_rfc_catalog(cat, cross_match_path=xm)
    rfc = rfc.loc[rfc["Name"] == "RFC J0040+2500"].reset_index(drop=True)
    meta = pd.DataFrame([_meta_row(meta_id=42, ra=10.0, dec=25.0, bmaj=0.5)])
    result = match_catalog_to_rfc(meta, rfc=rfc)
    rows = association_rows_from_match(
        result.meta_flags, result.rfc_footprint, ASSOCIATION_SPEC_RFC
    )
    assert len(rows) == 1
    assert rows.iloc[0]["catalog"] == "RFC"
    assert rows.iloc[0]["ref_id"] == "RFC J0040+2500"
    assert rows.iloc[0]["class_native"] == "QSO"
    assert harmonize_class_native("RFC", "QSO") == "agn"
    assert harmonize_class_native("RFC", "G") == "galaxy"
    assert harmonize_class_native("RFC", "BLLac") == "agn"
    assert harmonize_class_native("RFC", "GClstr") == "cluster"
    assert harmonize_class_native("RFC", "") == "radio"
    assert harmonize_class_native("RFC", "*") == "star"


def test_footprint_excludes_far_south(tmp_path) -> None:
    cat = tmp_path / "rfc.parquet"
    xm = tmp_path / "rfc_xm.csv"
    _write_mini_rfc_cat(cat)
    _write_mini_rfc_xm(xm)
    rfc = load_rfc_catalog(cat, cross_match_path=xm)
    meta = pd.DataFrame([_meta_row(meta_id=1, ra=10.0, dec=25.0, bmaj=0.01)])
    result = match_catalog_to_rfc(meta, rfc=rfc)
    # Southern Dec=-40 source should be outside LWA Dec footprint pad.
    assert all(result.rfc_footprint["DEC"] > -10.0)
    assert not np.any(result.rfc_footprint["Name"] == "RFC J0040-4000")
