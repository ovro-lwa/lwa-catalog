"""Tests for cross-band forced fill (gap insert + phot-dip replace)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.create.crossband_forced_fill import (
    CrossbandForcedFillConfig,
    classify_crossband_actions,
    fill_band_crossband_forced,
    summarize_crossband_forced_fill,
)


def _row(
    *,
    ra: float,
    dec: float,
    peak: float,
    band: str = "59MHz",
    bmaj: float = 0.5,
    total: float | None = None,
    source_id: int = 1,
) -> dict:
    tot = peak if total is None else total
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": tot,
        "E_Peak_flux": 0.1,
        "E_Total_flux": 0.1,
        "Maj": bmaj,
        "Min": bmaj * 0.8,
        "PA": 0.0,
        "BMAJ": bmaj,
        "BMIN": bmaj * 0.8,
        "BPA": 0.0,
        "band": band,
        "Source_id": source_id,
        "Isl_id": source_id,
        "S_Code": "S",
        "representative_lst": "healpix",
        "n_lst_contributions": 1,
    }


def _ok_fit(**kwargs):
    peak = float(kwargs.get("peak_guess", 20.0))
    if not np.isfinite(peak) or peak <= 0:
        peak = 20.0
    return {
        "ok": True,
        "forced_Peak": peak,
        "forced_Total": peak * 1.05,
        "forced_RA": float(kwargs["ra_deg"]),
        "forced_DEC": float(kwargs["dec_deg"]),
        "forced_resid_rms": 0.05,
    }


def _bad_resid_fit(**kwargs):
    out = _ok_fit(**kwargs)
    out["forced_resid_rms"] = 50.0
    return out


def test_classify_gap_insert_when_no_hit() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=11.0, dec=20.0, peak=18.0, band="55MHz", source_id=2)])
    actions = classify_crossband_actions(
        target,
        {"55MHz": seed},
        config=CrossbandForcedFillConfig(enable_phot_dip=False),
    )
    assert (actions["action"] == "insert").sum() == 1
    assert actions.iloc[0]["reason"] == "no_beam_hit"


def test_classify_gap_insert_when_only_offset_hit() -> None:
    # BMAJ=0.5° → beam match at 0.3°, but near gate 0.25*0.5=0.125° → offset_only
    target = pd.DataFrame([_row(ra=10.3, dec=20.0, peak=9.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=18.0, band="55MHz", source_id=2)])
    actions = classify_crossband_actions(
        target,
        {"55MHz": seed},
        config=CrossbandForcedFillConfig(enable_phot_dip=False, sep_frac_bmaj=0.25),
    )
    assert list(actions["action"]) == ["insert"]
    assert actions.iloc[0]["reason"] == "offset_only"


def test_classify_no_op_near_bright_when_phot_dip_off() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=17.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.01, dec=20.0, peak=18.0, band="55MHz", source_id=2)])
    actions = classify_crossband_actions(
        target,
        {"55MHz": seed},
        config=CrossbandForcedFillConfig(enable_phot_dip=False),
    )
    assert actions.empty or (actions["action"] != "insert").all()
    assert (actions["action"] == "replace").sum() == 0


def test_classify_phot_dip_replace_one_seed() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=10.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.01, dec=20.0, peak=18.0, band="73MHz", source_id=2)])
    actions = classify_crossband_actions(
        target,
        {"73MHz": seed},
        config=CrossbandForcedFillConfig(enable_phot_dip=True, peak_dip_ratio=0.70),
    )
    reps = actions.loc[actions["action"] == "replace"]
    assert len(reps) == 1
    assert float(reps.iloc[0]["peak_ratio"]) == pytest.approx(10.0 / 18.0)


def test_classify_phot_dip_two_neighbor_median() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=10.0, source_id=1)])
    lo = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=20.0, band="73MHz", source_id=2)])
    hi = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=16.0, band="82MHz", source_id=3)])
    actions = classify_crossband_actions(
        target,
        {"73MHz": lo, "82MHz": hi},
        config=CrossbandForcedFillConfig(enable_phot_dip=True, peak_dip_ratio=0.70),
    )
    reps = actions.loc[actions["action"] == "replace"]
    assert len(reps) == 1
    # median(20, 16) = 18 → 10/18 < 0.70
    assert float(reps.iloc[0]["seed_peak"]) == pytest.approx(18.0)
    assert float(reps.iloc[0]["peak_ratio"]) == pytest.approx(10.0 / 18.0)


def test_classify_peak_ok_skips_replace() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=17.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=18.0, band="73MHz", source_id=2)])
    actions = classify_crossband_actions(
        target,
        {"73MHz": seed},
        config=CrossbandForcedFillConfig(enable_phot_dip=True, peak_dip_ratio=0.70),
    )
    assert (actions["action"] == "replace").sum() == 0
    assert (actions["reason"] == "peak_ok").sum() == 1


def test_fill_gap_insert_applies_forced() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=11.0, dec=20.0, peak=18.0, band="55MHz", source_id=2)])
    result = fill_band_crossband_forced(
        target,
        {"55MHz": seed},
        target_band="59MHz",
        config=CrossbandForcedFillConfig(enable_phot_dip=False),
        fit_fn=_ok_fit,
    )
    assert result.summary["n_insert"] == 1
    assert len(result.target) == 2
    forced = result.target.loc[result.target["forced"] == True]  # noqa: E712
    assert len(forced) == 1
    assert float(forced.iloc[0]["Peak_flux"]) == pytest.approx(18.0)
    assert float(forced.iloc[0]["Total_flux"]) == pytest.approx(18.0 * 1.05)
    assert forced.iloc[0]["forced_mode"] == "insert"
    assert int(forced.iloc[0]["Source_id"]) < 0
    assert "n_insert=1" in summarize_crossband_forced_fill(result)


def test_fill_resid_reject_skips_insert() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=11.0, dec=20.0, peak=18.0, band="55MHz", source_id=2)])
    result = fill_band_crossband_forced(
        target,
        {"55MHz": seed},
        target_band="59MHz",
        config=CrossbandForcedFillConfig(enable_phot_dip=False),
        fit_fn=_bad_resid_fit,
    )
    assert result.summary["n_insert"] == 0
    assert result.summary["n_skip_resid"] == 1
    assert len(result.target) == 1


def test_fill_phot_dip_replace() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=10.0, total=7.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=18.0, band="73MHz", source_id=2)])
    result = fill_band_crossband_forced(
        target,
        {"73MHz": seed},
        target_band="78MHz",
        config=CrossbandForcedFillConfig(enable_phot_dip=True),
        fit_fn=_ok_fit,
    )
    assert result.summary["n_replace"] == 1
    assert len(result.target) == 1
    row = result.target.iloc[0]
    assert bool(row["forced"]) is True
    assert row["forced_mode"] == "replace"
    assert float(row["Peak_flux"]) == pytest.approx(18.0)
    assert float(row["Total_flux"]) == pytest.approx(18.0 * 1.05)


def test_fill_near_bright_unchanged_with_phot_dip() -> None:
    target = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=17.0, source_id=1)])
    seed = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=18.0, band="73MHz", source_id=2)])
    result = fill_band_crossband_forced(
        target,
        {"73MHz": seed},
        target_band="78MHz",
        config=CrossbandForcedFillConfig(enable_phot_dip=True),
        fit_fn=_ok_fit,
    )
    assert result.summary["n_replace"] == 0
    assert result.summary["n_insert"] == 0
    assert float(result.target.iloc[0]["Peak_flux"]) == pytest.approx(17.0)
    assert bool(result.target.iloc[0]["forced"]) is False
