"""Tests for NED-LVS cross-match QA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.nedlvs import (
    NedlvsMatchConfig,
    _diam_to_bmaj_deg,
    _footprint_filter_nedlvs,
    build_sfr_radio_luminosity_table,
    load_nedlvs_catalog,
    match_catalog_to_nedlvs,
    resolve_highest_frequency_peak_flux,
    select_metacatalog,
    summarize_nedlvs_match,
)
from lwa_catalog.constants import BAND_FREQ_HZ, NEDLVS_DEFAULT_BMAJ_DEG, band_frequency_hz


def _write_mini_nedlvs(path: Path) -> None:
    table = Table(
        {
            "objname": ["GAL1", "GAL2"],
            "ra": [10.0, 50.0],
            "dec": [20.0, 50.0],
            "objtype": ["G", "G"],
            "z": [0.01, 0.02],
            "DistMpc": [40.0, 80.0],
            "Diam": [40.0, np.nan],
            "Mstar": [1e10, 2e10],
            "SFR_hybrid": [1.5, np.nan],
            "SFR_W4": [1.0, 2.0],
        }
    )
    table.write(path, overwrite=True)


def _meta_rows(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meta_id": list(range(len(rows))),
            "RA": [r[0] for r in rows],
            "DEC": [r[1] for r in rows],
            "bands_present": ["Blue"] * len(rows),
            "BMAJ_match": [0.05] * len(rows),
        }
    )


def _nedlvs_rows(rows: list[tuple[float, float, float | None]]) -> pd.DataFrame:
    records = []
    for ra, dec, diam in rows:
        bmaj = NEDLVS_DEFAULT_BMAJ_DEG
        if diam is not None and np.isfinite(diam) and diam > 0:
            bmaj = diam / 3600.0
        records.append(
            {
                "RA": ra,
                "DEC": dec,
                "BMAJ": bmaj,
                "objname": "GAL",
                "objtype": "G",
                "z": 0.01,
                "DistMpc": 30.0,
                "Diam_arcsec": diam,
                "Mstar": 1e10,
                "SFR_hybrid": 1.0,
                "SFR_W4": 1.0,
            }
        )
    return pd.DataFrame(records)


def test_diam_to_bmaj_deg() -> None:
    diam = np.array([40.0, np.nan, 0.0])
    bmaj = _diam_to_bmaj_deg(diam, default_bmaj_deg=0.01)
    assert bmaj[0] == pytest.approx(40.0 / 3600.0)
    assert bmaj[1] == pytest.approx(0.01)
    assert bmaj[2] == pytest.approx(0.01)


def test_load_nedlvs_catalog(tmp_path) -> None:
    catalog_path = tmp_path / "nedlvs.fits"
    _write_mini_nedlvs(catalog_path)
    loaded = load_nedlvs_catalog(catalog_path)
    assert len(loaded) == 2
    assert loaded.loc[0, "BMAJ"] == pytest.approx(40.0 / 3600.0)
    assert loaded.loc[1, "BMAJ"] == pytest.approx(NEDLVS_DEFAULT_BMAJ_DEG)


def test_load_nedlvs_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.fits"
    with pytest.raises(FileNotFoundError, match="NED-LVS catalog not found"):
        load_nedlvs_catalog(missing)


def test_footprint_filter_nedlvs() -> None:
    nedlvs = pd.DataFrame({"DEC": [-10.0, 0.0, 15.0, 30.0]})
    lwa = pd.DataFrame({"DEC": [0.0, 20.0]})
    filtered = _footprint_filter_nedlvs(nedlvs, lwa)
    assert list(filtered["DEC"]) == [0.0, 15.0]


def test_match_single_hit() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.01, 20.0, 40.0)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    assert int(result.meta_flags.iloc[0]["n_nedlvs"]) == 1
    assert bool(result.meta_flags.iloc[0]["matched"])
    assert int(result.summary["n_nedlvs_matched"]) == 1
    assert result.summary["nedlvs_recovery"] == pytest.approx(1.0)


def test_match_oversplit_two_meta_one_nedlvs() -> None:
    meta = _meta_rows([(10.01, 20.0), (10.02, 20.0)])
    nedlvs = _nedlvs_rows([(10.015, 20.0, 40.0)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    assert int(result.summary["n_nedlvs_oversplit"]) == 1
    assert bool(result.nedlvs_flags.iloc[0]["oversplit"])


def test_summarize_nedlvs_match_includes_key_metrics() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.0, 20.0, 40.0)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    text = summarize_nedlvs_match(result)
    assert "NED-LVS recovery" in text
    assert "Meta with NED-LVS host" in text


def test_match_lst_merged_blue_target() -> None:
    lst_blue = _meta_rows([(10.0, 20.0)]).drop(columns=["bands_present"])
    nedlvs = _nedlvs_rows([(10.0, 20.0, 40.0)])
    config = NedlvsMatchConfig(target="lst_merged_blue")
    result = match_catalog_to_nedlvs(lst_blue, nedlvs=nedlvs, config=config)
    assert int(result.summary["n_meta_matched"]) == 1


def test_band_frequency_hz() -> None:
    assert band_frequency_hz("Blue") == pytest.approx(BAND_FREQ_HZ["Blue"])
    assert band_frequency_hz("82MHz") == pytest.approx(82e6)


def test_resolve_highest_frequency_peak_flux() -> None:
    row = pd.Series(
        {
            "bands_present": "Blue,Green,Red",
            "Peak_flux_Blue": 2.0,
            "Peak_flux_Green": 3.0,
            "Peak_flux_Red": 4.0,
            "origin_band": "Full",
        }
    )
    flux, freq, band = resolve_highest_frequency_peak_flux(row)
    assert band == "Blue"
    assert flux == pytest.approx(2.0)
    assert freq == pytest.approx(BAND_FREQ_HZ["Blue"])


def test_select_metacatalog_blue() -> None:
    meta = pd.DataFrame(
        {
            "meta_id": [0, 1],
            "bands_present": ["Blue", "Red"],
        }
    )
    out = select_metacatalog(meta, selection="blue")
    assert len(out) == 1
    assert int(out.iloc[0]["meta_id"]) == 0


def test_build_sfr_radio_luminosity_table() -> None:
    meta = pd.DataFrame(
        {
            "meta_id": [0],
            "RA": [10.0],
            "DEC": [20.0],
            "bands_present": ["Blue"],
            "Peak_flux_Blue": [0.5],
            "BMAJ_match": [0.05],
        }
    )
    nedlvs = _nedlvs_rows([(10.01, 20.0, 40.0)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    table = build_sfr_radio_luminosity_table(meta, result.nedlvs_footprint, result.meta_flags)
    assert len(table) == 1
    assert table.iloc[0]["SFR_column"] == "SFR_hybrid"
    assert table.iloc[0]["nuL_nu_erg_s"] > 0
