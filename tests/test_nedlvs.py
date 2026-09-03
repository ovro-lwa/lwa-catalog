"""Tests for NED-LVS cross-match QA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.nedlvs import (
    NedlvsMatchConfig,
    _associate_by_centroid_sigma,
    _filter_nedlvs_redshift,
    _footprint_filter_nedlvs,
    build_sfr_radio_luminosity_table,
    load_nedlvs_catalog,
    match_catalog_to_nedlvs,
    resolve_band_flux,
    resolve_centroid_sigma_deg,
    resolve_highest_frequency_flux,
    resolve_highest_frequency_peak_flux,
    resolve_row_flux,
    select_bijective_nedlvs_flags,
    select_metacatalog,
    select_unique_nedlvs_matches,
    summarize_nedlvs_match,
)
from lwa_catalog.constants import (
    BAND_FREQ_HZ,
    NEDLVS_DEFAULT_CENTROID_SIGMA_DEG,
    NEDLVS_DEFAULT_MAX_REDSHIFT,
    band_frequency_hz,
)


def _write_mini_nedlvs(path: Path) -> None:
    table = Table(
        {
            "objname": ["GAL1", "GAL2"],
            "ra": [10.0, 50.0],
            "dec": [20.0, 50.0],
            "objtype": ["G", "G"],
            "z": [0.01, 0.5],
            "DistMpc": [40.0, 80.0],
            "Diam": [40.0, np.nan],
            "Mstar": [1e10, 2e10],
            "SFR_hybrid": [1.5, np.nan],
            "SFR_W4": [1.0, 2.0],
        }
    )
    table.write(path, overwrite=True)


def _meta_rows(
    rows: list[tuple[float, float]],
    *,
    jitter_deg: float | None = 0.001,
) -> pd.DataFrame:
    jitter = [jitter_deg] * len(rows) if jitter_deg is not None else [0.0] * len(rows)
    return pd.DataFrame(
        {
            "meta_id": list(range(len(rows))),
            "RA": [r[0] for r in rows],
            "DEC": [r[1] for r in rows],
            "bands_present": ["Blue"] * len(rows),
            "cluster_jitter_rms_deg": jitter,
            "Peak_flux_Blue": [1.0] * len(rows),
        }
    )


def _nedlvs_rows(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    records = []
    for ra, dec, z in rows:
        records.append(
            {
                "RA": ra,
                "DEC": dec,
                "objname": "GAL",
                "objtype": "G",
                "z": z,
                "DistMpc": 30.0,
                "Diam_arcsec": 20.0,
                "Mstar": 1e10,
                "SFR_hybrid": 1.0,
                "SFR_W4": 1.0,
            }
        )
    return pd.DataFrame(records)


def test_load_nedlvs_catalog(tmp_path) -> None:
    catalog_path = tmp_path / "nedlvs.fits"
    _write_mini_nedlvs(catalog_path)
    loaded = load_nedlvs_catalog(catalog_path)
    assert len(loaded) == 2
    assert "Diam_arcsec" in loaded.columns
    assert "BMAJ" not in loaded.columns


def test_load_nedlvs_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.fits"
    with pytest.raises(FileNotFoundError, match="NED-LVS catalog not found"):
        load_nedlvs_catalog(missing)


def test_resolve_centroid_sigma_prefers_e_ra_dec() -> None:
    row = pd.Series({"E_RA": 0.002, "E_DEC": 0.001, "cluster_jitter_rms_deg": 0.05})
    assert resolve_centroid_sigma_deg(row) == pytest.approx(np.hypot(0.002, 0.001))


def test_resolve_centroid_sigma_uses_jitter() -> None:
    row = pd.Series({"cluster_jitter_rms_deg": 0.01})
    assert resolve_centroid_sigma_deg(row) == pytest.approx(0.01)


def test_resolve_centroid_sigma_default() -> None:
    row = pd.Series({"cluster_jitter_rms_deg": 0.0})
    assert resolve_centroid_sigma_deg(row) == pytest.approx(NEDLVS_DEFAULT_CENTROID_SIGMA_DEG)


def test_footprint_filter_nedlvs() -> None:
    nedlvs = pd.DataFrame({"DEC": [-10.0, 0.0, 15.0, 30.0]})
    lwa = pd.DataFrame({"DEC": [0.0, 20.0]})
    filtered = _footprint_filter_nedlvs(nedlvs, lwa)
    assert list(filtered["DEC"]) == [0.0, 15.0]


def test_filter_nedlvs_redshift() -> None:
    nedlvs = _nedlvs_rows([(0.0, 0.0, 0.01), (1.0, 1.0, 0.5)])
    filtered = _filter_nedlvs_redshift(nedlvs, 0.2)
    assert len(filtered) == 1
    assert float(filtered.iloc[0]["z"]) == pytest.approx(0.01)


def test_associate_by_centroid_sigma_single_hit() -> None:
    base = pd.DataFrame({"RA": [10.0], "DEC": [20.0], "SIGMA": [0.001]})
    ref = pd.DataFrame({"RA": [10.0001], "DEC": [20.0], "SIGMA": [0.0001]})
    hits, matched = _associate_by_centroid_sigma(base, ref, position_sigma_scale=3.0)
    assert hits == {0: [0]}
    assert matched == {0}


def test_match_single_hit() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.01)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    assert int(result.meta_flags.iloc[0]["n_nedlvs"]) == 1
    assert bool(result.meta_flags.iloc[0]["matched"])
    assert int(result.summary["n_meta_unique_nedlvs"]) == 1


def test_match_uses_match_ra_dec_over_native() -> None:
    """NED-LVS association should use cascaded match_* coords when present."""
    from lwa_catalog.analyze.nedlvs import resolve_match_coordinates

    meta = _meta_rows([(10.0, 20.0)], jitter_deg=1e-5)
    # Native LWA far from NED; match_* on the NED galaxy.
    meta["match_RA"] = [10.0001]
    meta["match_DEC"] = [20.0]
    meta["match_source"] = ["VLASS"]
    ra, dec = resolve_match_coordinates(meta.iloc[0])
    assert ra == pytest.approx(10.0001)
    assert dec == pytest.approx(20.0)

    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.01)])
    result = match_catalog_to_nedlvs(
        meta,
        nedlvs=nedlvs,
        config=NedlvsMatchConfig(position_sigma_scale=3.0, default_centroid_sigma_deg=1e-5),
    )
    assert int(result.meta_flags.iloc[0]["n_nedlvs"]) == 1
    assert float(result.meta_flags.iloc[0]["match_RA"]) == pytest.approx(10.0001)
    assert result.meta_flags.iloc[0]["match_source"] == "VLASS"

    # Without match_*, native RA is 1° away — should miss with tiny sigma.
    meta_native = _meta_rows([(9.0, 20.0)], jitter_deg=1e-5)
    miss = match_catalog_to_nedlvs(
        meta_native,
        nedlvs=nedlvs,
        config=NedlvsMatchConfig(position_sigma_scale=3.0, default_centroid_sigma_deg=1e-5),
    )
    assert int(miss.meta_flags.iloc[0]["n_nedlvs"]) == 0


def test_match_rejects_high_redshift() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.5)])
    config = NedlvsMatchConfig(max_redshift=0.2)
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs, config=config)
    assert int(result.summary["n_meta_matched"]) == 0


def test_match_oversplit_two_meta_one_nedlvs() -> None:
    meta = _meta_rows([(10.0001, 20.0), (10.0002, 20.0)], jitter_deg=0.01)
    nedlvs = _nedlvs_rows([(10.00015, 20.0, 0.01)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    assert int(result.summary["n_nedlvs_oversplit"]) == 1
    assert bool(result.nedlvs_flags.iloc[0]["oversplit"])


def test_summarize_nedlvs_match_includes_key_metrics() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.01)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    text = summarize_nedlvs_match(result)
    assert "Meta unique match" in text
    assert "NED-LVS recovery" in text


def test_match_lst_merged_blue_target() -> None:
    lst_blue = _meta_rows([(10.0, 20.0)]).drop(columns=["bands_present"])
    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.01)])
    config = NedlvsMatchConfig(target="lst_merged_blue")
    result = match_catalog_to_nedlvs(lst_blue, nedlvs=nedlvs, config=config)
    assert int(result.summary["n_meta_matched"]) == 1


def test_band_frequency_hz() -> None:
    assert band_frequency_hz("Blue") == pytest.approx(BAND_FREQ_HZ["Blue"])
    assert band_frequency_hz("82MHz") == pytest.approx(82e6)
    assert band_frequency_hz("VLSSR") == pytest.approx(BAND_FREQ_HZ["VLSSR"])
    assert band_frequency_hz("NVSS") == pytest.approx(BAND_FREQ_HZ["NVSS"])
    assert band_frequency_hz("VLASS") == pytest.approx(BAND_FREQ_HZ["VLASS"])


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


def test_resolve_highest_frequency_flux_prefers_total() -> None:
    row = pd.Series(
        {
            "bands_present": "82MHz,18MHz",
            "Peak_flux_82MHz": 2.0,
            "Total_flux_82MHz": 3.5,
            "Peak_flux_18MHz": 1.0,
            "Total_flux_18MHz": 1.2,
        }
    )
    flux, freq, band = resolve_highest_frequency_flux(row, prefer_total=True)
    assert band == "82MHz"
    assert flux == pytest.approx(3.5)
    assert freq == pytest.approx(82e6)


def test_resolve_highest_frequency_flux_falls_back_to_peak() -> None:
    row = pd.Series(
        {
            "bands_present": "73MHz",
            "Peak_flux_73MHz": 1.5,
        }
    )
    flux, freq, band = resolve_highest_frequency_flux(row, prefer_total=True)
    assert band == "73MHz"
    assert flux == pytest.approx(1.5)
    assert freq == pytest.approx(73e6)


def test_resolve_band_and_row_flux() -> None:
    meta = pd.Series(
        {
            "origin_band": "82MHz",
            "Total_flux_82MHz": 4.0,
            "Peak_flux_82MHz": 2.0,
        }
    )
    assert resolve_band_flux(meta, "82MHz", prefer_total=True) == pytest.approx(4.0)
    assert resolve_band_flux(meta, "82MHz", prefer_total=False) == pytest.approx(2.0)

    ref = pd.Series({"Peak_intensity": 0.05})
    assert resolve_row_flux(ref, prefer_total=True) == pytest.approx(0.05)
    ref_total = pd.Series({"Total_flux": 0.2, "Peak_flux": 0.1})
    assert resolve_row_flux(ref_total, prefer_total=True) == pytest.approx(0.2)


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


def test_select_unique_nedlvs_matches() -> None:
    flags = pd.DataFrame({"meta_id": [0, 1, 2], "n_nedlvs": [0, 1, 3]})
    unique = select_unique_nedlvs_matches(flags)
    assert list(unique["meta_id"]) == [1]


def test_build_sfr_radio_luminosity_table_unique_only() -> None:
    meta = _meta_rows([(10.0, 20.0)])
    nedlvs = _nedlvs_rows([(10.0001, 20.0, 0.01)])
    result = match_catalog_to_nedlvs(meta, nedlvs=nedlvs)
    table = build_sfr_radio_luminosity_table(
        meta, result.nedlvs_footprint, result.meta_flags, unique_only=True
    )
    assert len(table) == 1
    assert table.iloc[0]["SFR_column"] == "SFR_hybrid"
    assert table.iloc[0]["nuL_nu_erg_s"] > 0


def test_select_bijective_nedlvs_flags() -> None:
    meta_flags = pd.DataFrame(
        {
            "meta_id": [0, 1, 2],
            "n_nedlvs": [1, 1, 2],
            "matched": [True, True, True],
        }
    )
    nedlvs_flags = pd.DataFrame(
        {
            "nedlvs_pos": [0, 1, 2],
            "n_meta": [1, 2, 1],
            "meta_ids": [[0], [0, 1], [2]],
        }
    )
    bijective = select_bijective_nedlvs_flags(meta_flags, nedlvs_flags)
    assert list(bijective["nedlvs_pos"]) == [0]


def test_nedlvs_match_config_defaults() -> None:
    cfg = NedlvsMatchConfig()
    assert cfg.max_redshift == pytest.approx(NEDLVS_DEFAULT_MAX_REDSHIFT)
    assert cfg.position_sigma_scale == pytest.approx(3.0)
