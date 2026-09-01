"""Tests for metacatalog reliability predicates and filters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.reliability import (
    ReliabilityConfig,
    ReliabilityResult,
    SourceQualityFlag,
    assert_gold_subset_of_cleaned,
    assign_source_quality_flags,
    cluster_radec_jitter_rms,
    decode_quality_flag,
    filter_metacatalog_reliability,
    flag_has_nan,
    flag_jitter_exceeds,
    flag_residual_absolute,
    flag_residual_percentile,
    flag_scode_complex,
    flag_unphysical_flux,
    flux_sigma_total_minus_peak,
    pack_quality_flags,
    passes_multi_image,
    quality_flag_legend,
    seed_lst_rows,
)
from lwa_catalog.constants import CLUSTER_JITTER_RMS_COL
from lwa_catalog.create.merge import build_global_metacatalog, merge_lst_metacatalog
from lwa_catalog.io import write_lst_merged, write_metacatalog, write_sources_catalog
from lwa_catalog.paths import CatalogLayout


def test_reliability_config_defaults() -> None:
    cfg = ReliabilityConfig()
    assert cfg.resid_rms_thresh_jy == 1.0
    assert cfg.resid_mean_thresh_jy == 1.0
    assert cfg.resid_percentile_lo == 1.0
    assert cfg.resid_percentile_hi == 99.0
    assert cfg.jitter_bmaj_frac == 0.3
    assert cfg.require_unique_assoc_include is True
    assert cfg.require_unique_assoc_exclude is False


def test_flux_sigma_and_unphysical() -> None:
    df = pd.DataFrame(
        {
            "Total_flux": [1.0, 1.0, 0.0, 1.0],
            "Peak_flux": [2.0, 1.05, 2.0, 2.0],
            "E_Total_flux": [0.1, 0.1, 0.1, np.nan],
            "E_Peak_flux": [0.1, 0.1, 0.1, 0.1],
        }
    )
    sigma = flux_sigma_total_minus_peak(df)
    assert sigma.iloc[0] < -3
    assert abs(sigma.iloc[1]) < 1
    assert not np.isfinite(sigma.iloc[3])
    unphys = flag_unphysical_flux(df, nsigma=3.0)
    assert bool(unphys.iloc[0]) is True
    assert bool(unphys.iloc[1]) is False
    assert bool(unphys.iloc[3]) is False


def test_residual_absolute_dual_thresh() -> None:
    df = pd.DataFrame(
        {
            "Resid_Isl_rms": [0.5, 1.5, 0.2, np.nan],
            "Resid_Isl_mean": [0.1, 0.1, -1.5, 0.0],
        }
    )
    flags = flag_residual_absolute(df, rms_thresh_jy=1.0, mean_thresh_jy=1.0)
    assert bool(flags["resid_fail"].iloc[0]) is False
    assert bool(flags["resid_rms_fail"].iloc[1]) is True
    assert bool(flags["resid_mean_fail"].iloc[2]) is True
    assert bool(flags["resid_fail"].iloc[3]) is False


def test_passes_multi_image_unique_assoc() -> None:
    row = pd.Series(
        {
            "n_lst_contributions": 2,
            "bands_present": "Full",
            "n_assoc_Blue": 0,
        }
    )
    assert passes_multi_image(row) is True

    row2 = pd.Series(
        {
            "n_lst_contributions": 1,
            "bands_present": "Full,Blue",
            "n_assoc_Blue": 1,
        }
    )
    assert passes_multi_image(row2) is True

    row3 = pd.Series(
        {
            "n_lst_contributions": 1,
            "bands_present": "Full,Blue",
            "n_assoc_Blue": 2,
        }
    )
    assert passes_multi_image(row3) is False

    row4 = pd.Series(
        {
            "n_lst_contributions": 1,
            "bands_present": "Full",
        }
    )
    assert passes_multi_image(row4) is False


def test_cluster_jitter_rms() -> None:
    tight = pd.DataFrame({"RA": [10.0, 10.001], "DEC": [20.0, 20.001]})
    rms = cluster_radec_jitter_rms(tight)
    assert rms < 0.01
    assert cluster_radec_jitter_rms(pd.DataFrame({"RA": [10.0], "DEC": [20.0]})) == 0.0
    assert not np.isfinite(cluster_radec_jitter_rms(pd.DataFrame()))
    assert flag_jitter_exceeds(0.2, 0.5, frac=0.3) is True
    assert flag_jitter_exceeds(0.1, 0.5, frac=0.3) is False


def _src(
    *,
    ra: float,
    dec: float,
    peak: float,
    lst_hour: str,
    band: str,
    source_id: int,
    bmaj: float = 0.5,
    total: float | None = None,
    resid_rms: float = 0.1,
    resid_mean: float = 0.05,
    e_peak: float = 0.05,
    e_total: float = 0.05,
) -> dict:
    total_flux = peak if total is None else total
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": total_flux,
        "E_Peak_flux": e_peak,
        "E_Total_flux": e_total,
        "Maj": 0.1,
        "Min": 0.05,
        "PA": 0.0,
        "DC_Maj": 0.1,
        "DC_Min": 0.05,
        "DC_PA": 0.0,
        "BMAJ": bmaj,
        "Resid_Isl_rms": resid_rms,
        "Resid_Isl_mean": resid_mean,
        "lst_hour": lst_hour,
        "band": band,
        "Source_id": source_id,
        "source_file": f"{band}_{lst_hour}.fits",
    }


def _reliability_layout(tmp_path: Path) -> tuple[CatalogLayout, pd.DataFrame, dict]:
    layout = CatalogLayout(tmp_path)
    full_01 = pd.DataFrame(
        [_src(ra=30.0, dec=37.0, peak=2.0, lst_hour="01h", band="Full", source_id=101)]
    )
    full_02 = pd.DataFrame(
        [_src(ra=30.02, dec=37.0, peak=1.8, lst_hour="02h", band="Full", source_id=102)]
    )
    blue_01 = pd.DataFrame(
        [_src(ra=30.01, dec=37.0, peak=1.5, lst_hour="01h", band="Blue", source_id=201)]
    )
    blue_02 = pd.DataFrame(
        [_src(ra=30.03, dec=37.0, peak=1.4, lst_hour="02h", band="Blue", source_id=202)]
    )
    write_sources_catalog(full_01, layout, "01h", "Full")
    write_sources_catalog(full_02, layout, "02h", "Full")
    write_sources_catalog(blue_01, layout, "01h", "Blue")
    write_sources_catalog(blue_02, layout, "02h", "Blue")
    lst_full = merge_lst_metacatalog([full_01, full_02], band="Full")
    lst_blue = merge_lst_metacatalog([blue_01, blue_02], band="Blue")
    write_lst_merged(lst_full, layout, "Full")
    write_lst_merged(lst_blue, layout, "Blue")
    write_lst_merged(pd.DataFrame(), layout, "Green")
    write_lst_merged(pd.DataFrame(), layout, "Red")
    lst_merged = {
        "Full": lst_full,
        "Blue": lst_blue,
        "Green": pd.DataFrame(),
        "Red": pd.DataFrame(),
    }
    meta = build_global_metacatalog(lst_merged)
    write_metacatalog(meta, layout)
    return layout, meta, lst_merged


def test_reliability_uses_merge_time_jitter_without_source_rematch(tmp_path: Path) -> None:
    """Merge-time jitter avoids per-hour source Parquet reads."""
    layout = CatalogLayout(tmp_path)
    full_01 = pd.DataFrame(
        [_src(ra=30.0, dec=37.0, peak=2.0, lst_hour="01h", band="Full", source_id=101)]
    )
    full_02 = pd.DataFrame(
        [_src(ra=30.02, dec=37.0, peak=1.8, lst_hour="02h", band="Full", source_id=102)]
    )
    blue_01 = pd.DataFrame(
        [_src(ra=30.01, dec=37.0, peak=1.5, lst_hour="01h", band="Blue", source_id=201)]
    )
    lst_full = merge_lst_metacatalog([full_01, full_02], band="Full")
    lst_blue = merge_lst_metacatalog([blue_01], band="Blue")
    write_lst_merged(lst_full, layout, "Full")
    write_lst_merged(lst_blue, layout, "Blue")
    write_lst_merged(pd.DataFrame(), layout, "Green")
    write_lst_merged(pd.DataFrame(), layout, "Red")
    meta = build_global_metacatalog(
        {"Full": lst_full, "Blue": lst_blue, "Green": pd.DataFrame(), "Red": pd.DataFrame()}
    )
    assert CLUSTER_JITTER_RMS_COL in meta.columns
    write_metacatalog(meta, layout)

    lst_merged = {
        "Full": lst_full,
        "Blue": lst_blue,
        "Green": pd.DataFrame(),
        "Red": pd.DataFrame(),
    }
    cleaned, gold = filter_metacatalog_reliability(
        meta,
        layout,
        lst_merged=lst_merged,
        config=ReliabilityConfig(strict=True),
    )
    assert any("merge-time cluster jitter" in w for w in cleaned.warnings)
    assert len(cleaned.flags) == len(meta)
    assert cleaned.flags["jitter_rms_deg"].notna().any()
    assert int(cleaned.flags["n_rematch"].max()) >= 2
    meta_jitter = pd.to_numeric(meta[CLUSTER_JITTER_RMS_COL], errors="coerce").iloc[0]
    assert cleaned.flags["jitter_rms_deg"].iloc[0] == pytest.approx(float(meta_jitter))


def test_seed_lst_and_filters_nesting(tmp_path: Path) -> None:
    layout, meta, lst_merged = _reliability_layout(tmp_path)
    seed, _ = seed_lst_rows(meta, layout, lst_merged=lst_merged)
    assert seed["seed_matched"].any()

    cleaned, gold = filter_metacatalog_reliability(
        meta, layout, lst_merged=lst_merged, config=ReliabilityConfig(strict=True)
    )
    assert len(gold.catalog) <= len(cleaned.catalog) <= len(meta)
    assert set(gold.meta_ids).issubset(set(cleaned.meta_ids))
    assert len(cleaned.catalog) >= 1
    assert len(gold.catalog) >= 1


def test_soft_keep_hard_reject_missing_seed(tmp_path: Path) -> None:
    layout, meta, lst_merged = _reliability_layout(tmp_path)
    meta = meta.copy()
    meta["origin_band"] = "Green"
    cleaned, gold = filter_metacatalog_reliability(
        meta, layout, lst_merged=lst_merged, config=ReliabilityConfig(strict=True)
    )
    assert len(cleaned.catalog) >= 1
    assert len(gold.catalog) == 0


def test_confused_band_does_not_buy_multi_image(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    full = pd.DataFrame(
        [
            {
                **_src(ra=40.0, dec=37.0, peak=2.0, lst_hour="01h", band="Full", source_id=1),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    blue = pd.DataFrame(
        [
            {
                **_src(ra=40.01, dec=37.0, peak=1.0, lst_hour="01h", band="Blue", source_id=2),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    write_lst_merged(full, layout, "Full")
    write_lst_merged(blue, layout, "Blue")
    write_lst_merged(pd.DataFrame(), layout, "Green")
    write_lst_merged(pd.DataFrame(), layout, "Red")
    write_sources_catalog(
        pd.DataFrame([_src(ra=40.0, dec=37.0, peak=2.0, lst_hour="01h", band="Full", source_id=1)]),
        layout,
        "01h",
        "Full",
    )
    meta = build_global_metacatalog(
        {"Full": full, "Blue": blue, "Green": pd.DataFrame(), "Red": pd.DataFrame()}
    )
    meta = meta.copy()
    meta["n_assoc_Blue"] = 2
    meta["n_lst_contributions"] = 1
    write_metacatalog(meta, layout)
    cleaned, gold = filter_metacatalog_reliability(meta, layout, config=ReliabilityConfig())
    assert len(cleaned.catalog) == 0
    assert len(gold.catalog) == 0


def test_assert_gold_subset_warns() -> None:
    cleaned = ReliabilityResult(
        catalog=pd.DataFrame({"meta_id": [1]}),
        meta_ids=np.array([1]),
        tier_counts=pd.DataFrame(),
        flags=pd.DataFrame(),
    )
    gold = ReliabilityResult(
        catalog=pd.DataFrame({"meta_id": [1, 2]}),
        meta_ids=np.array([1, 2]),
        tier_counts=pd.DataFrame(),
        flags=pd.DataFrame(),
    )
    with pytest.warns(UserWarning, match="gold"):
        assert_gold_subset_of_cleaned(cleaned, gold, strict=False)
    with pytest.raises(ValueError, match="gold"):
        assert_gold_subset_of_cleaned(cleaned, gold, strict=True)


def test_quality_flag_pack_and_decode() -> None:
    legend = quality_flag_legend()
    assert len(legend) == 12
    assert set(legend["bit"]) == set(range(12))
    flags = pd.DataFrame(
        {
            "has_nan": [True, False],
            "invalid": [False, False],
            "single_lst": [False, True],
            "single_unique_band": [False, False],
            "unphysical_soft": [False, False],
            "resid_fail_soft": [False, False],
            "resid_pctl_rms": [False, False],
            "resid_pctl_mean": [False, False],
            "jitter_fail_soft": [False, False],
            "confused_assoc": [False, False],
            "no_vlssr": [False, True],
            "scode_complex": [False, False],
        }
    )
    packed = pack_quality_flags(flags)
    assert packed.dtype == np.uint32
    assert packed[0] == np.uint32(SourceQualityFlag.HAS_NAN)
    assert packed[1] == np.uint32(SourceQualityFlag.SINGLE_LST | SourceQualityFlag.NO_VLSSR)
    assert decode_quality_flag(int(packed[0])) == ["HAS_NAN"]
    assert decode_quality_flag(0) == []
    assert packed[0] != 0
    assert packed[1] != 0


def test_flag_has_nan_core_columns_only() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0, np.nan],
            "DEC": [20.0, 20.0],
            "Peak_flux": [1.0, 1.0],
            "Peak_flux_Blue": [np.nan, np.nan],
        }
    )
    flagged = flag_has_nan(df)
    assert bool(flagged.iloc[0]) is False
    assert bool(flagged.iloc[1]) is True


def test_flag_scode_and_residual_percentile() -> None:
    scode = flag_scode_complex(pd.Series(["S", "C", "M", np.nan, "s"]))
    assert scode.tolist() == [False, True, True, False, False]

    rms = np.array([0.1, 0.2, 0.3, 0.4, 10.0])
    mean = np.array([-5.0, 0.0, 0.01, 0.02, 0.03])
    pctl = flag_residual_percentile(rms, mean, lo=1.0, hi=99.0)
    assert bool(pctl["resid_pctl_rms"].iloc[-1]) is True
    assert bool(pctl["resid_pctl_rms"].iloc[1]) is False
    assert bool(pctl["resid_pctl_mean"].iloc[0]) is True
    assert bool(pctl["resid_pctl_mean"].iloc[2]) is False


def test_assign_source_quality_flags_keeps_all_rows(tmp_path: Path) -> None:
    layout, meta, lst_merged = _reliability_layout(tmp_path)
    vlssr = pd.DataFrame(
        {
            "RA": [30.0],
            "DEC": [37.0],
            "Peak_flux": [1.0],
            "BMAJ": [0.5],
            "BMIN": [0.5],
        }
    )
    result = assign_source_quality_flags(
        meta,
        layout,
        lst_merged=lst_merged,
        config=ReliabilityConfig(strict=True),
        vlssr=vlssr,
    )
    assert len(result.catalog) == len(meta)
    assert "quality_flag" in result.catalog.columns
    assert result.catalog["quality_flag"].to_numpy().dtype == np.uint32
    assert int((~result.flags["no_vlssr"]).sum()) >= 1

    meta_bad = meta.copy()
    meta_bad["S_Code"] = "C"
    meta_bad["n_lst_contributions"] = 1
    result_bad = assign_source_quality_flags(
        meta_bad,
        layout,
        lst_merged=lst_merged,
        vlssr=pd.DataFrame(
            {"RA": [0.0], "DEC": [0.0], "Peak_flux": [1.0], "BMAJ": [0.01], "BMIN": [0.01]}
        ),
    )
    flag = int(result_bad.catalog["quality_flag"].iloc[0])
    names = set(decode_quality_flag(flag))
    assert "SCODE_COMPLEX" in names
    assert "SINGLE_LST" in names
    assert "NO_VLSSR" in names
    assert flag != 0
