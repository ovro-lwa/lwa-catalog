"""Tests for Xu et al. 2022 RXGCC galaxy-cluster cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    RXGCC_REFERENCE_RADIUS_BEAM,
)
from lwa_catalog.analyze.xu2022 import (
    Xu2022MatchConfig,
    filter_xu2022_by_class,
    load_xu2022_catalog,
    match_catalog_to_xu2022,
    select_unique_xu2022_matches,
    summarize_xu2022_match,
)


def _write_mini_rxgcc_fits(path) -> None:
    table = Table(
        {
            "RXGCC": [1, 2, 3],
            "RAdeg": [180.0, 180.5, 10.0],
            "DEdeg": [40.0, 40.0, 0.0],
            "Ext": [3.0, 4.0, 2.0],
            "Extml": [50.0, 40.0, 30.0],
            "z": [0.05, 0.1, 0.2],
            "e_z": [0.005, 0.005, 0.005],
            "r_z": [b"z1", b"z1", b"z_opt"],
            "Class": [b"G", b"B", b"S"],
            "GCXSZ": [b"", b"MCXC", b""],
            "GCOPT": [b"", b"", b"RM"],
            "GC*": [b"", b"A", b""],
            "Rsig": [12.0, 15.0, 10.0],
            "R500*": [9.0, 10.0, 8.0],
            "R500": [0.8, 1.0, 0.7],
            "CRsig": [0.3, 0.4, 0.2],
            "e_CRsig": [0.03, 0.04, 0.02],
            "CR500": [0.3, 0.4, 0.2],
            "e_CR500": [0.03, 0.04, 0.02],
            "L500": [0.5, 1.0, 0.2],
            "e_L500": [0.1, 0.2, 0.05],
            "F500": [3.0, 5.0, 1.0],
            "e_F500": [0.3, 0.5, 0.1],
            "M500": [1.0, 2.0, 0.5],
            "e_M500": [0.2, 0.3, 0.1],
            "TX": [3.0, 4.0, 2.0],
            "e_TX": [0.5, 0.5, 0.3],
            "CRpsig": [100.0, 200.0, 50.0],
            "EdgeRsig": [0, 0, 0],
            "Beta": [0.8, 0.8, 0.7],
            "e_Beta": [0.1, 0.1, 0.1],
            "E_Beta": [0.1, 0.1, 0.1],
            "Rc": [1.0, 1.2, 0.8],
            "e_Rc": [0.1, 0.1, 0.1],
            "E_Rc": [0.1, 0.1, 0.1],
            "Com": [b"", b"", b""],
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


def test_load_xu2022_catalog_fits(tmp_path) -> None:
    path = tmp_path / "xu2022table3.dat"
    _write_mini_rxgcc_fits(path)
    df = load_xu2022_catalog(path)
    assert len(df) == 3
    assert "RA" in df.columns and "DEC" in df.columns
    assert "R500_arcmin" in df.columns
    assert float(df["BMAJ"].iloc[0]) == pytest.approx(9.0 / 60.0)
    assert df["Class"].iloc[0] == "G"


def test_load_xu2022_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="RXGCC catalog not found"):
        load_xu2022_catalog(tmp_path / "missing.dat")


def test_filter_xu2022_by_class(tmp_path) -> None:
    path = tmp_path / "xu2022table3.dat"
    _write_mini_rxgcc_fits(path)
    df = load_xu2022_catalog(path)
    gold = filter_xu2022_by_class(df, ("G",))
    assert len(gold) == 1
    assert gold["Class"].iloc[0] == "G"


def test_match_catalog_to_xu2022(tmp_path) -> None:
    path = tmp_path / "xu2022table3.dat"
    _write_mini_rxgcc_fits(path)
    ref = load_xu2022_catalog(path)
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=180.0, dec=40.0),
            _meta_row(meta_id=2, ra=0.0, dec=-20.0),
        ]
    )
    result = match_catalog_to_xu2022(
        meta,
        xu2022=ref,
        config=Xu2022MatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=RXGCC_REFERENCE_RADIUS_BEAM,
        ),
    )
    assert result.summary["n_xu2022_footprint"] == 3
    assert int(result.meta_flags.loc[0, "n_xu2022"]) >= 1
    assert int(result.meta_flags.loc[1, "n_xu2022"]) == 0
    unique = select_unique_xu2022_matches(result.meta_flags)
    assert len(unique) >= 1
    text = summarize_xu2022_match(result)
    assert "RXGCC" in text
    assert np.isfinite(float(result.xu2022_footprint["POS_ERR"].iloc[0]))
