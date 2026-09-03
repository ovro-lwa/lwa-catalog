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
    filter_by_quality_flags,
    filter_by_quality_mask,
    filter_metacatalog_reliability,
    flag_has_nan,
    flag_extended,
    flag_high_ellipticity,
    flag_invalid_astrometry_flux,
    flag_jitter_exceeds,
    flag_low_elevation,
    flag_residual_absolute,
    flag_residual_percentile,
    flag_single_unique_band,
    flag_scode_complex,
    flag_unphysical_flux,
    flux_qa_frame,
    flux_sigma_total_minus_peak,
    is_subband_metacatalog,
    pack_quality_flags,
    passes_multi_image,
    qa_band_for_row,
    quality_flag_legend,
    representative_peak_flux,
    seed_lst_rows,
    unique_assoc_band_count,
)
from lwa_catalog.constants import CLUSTER_JITTER_RMS_COL, SUBBAND_METACATALOG_REQUIRED_COLUMNS
from lwa_catalog.create.merge import (
    build_global_metacatalog,
    build_subband_metacatalog,
    merge_lst_metacatalog,
)
from lwa_catalog.io import (
    read_metacatalog,
    seed_band_from_discovery,
    write_lst_merged,
    write_metacatalog,
    write_sources_catalog,
)
from lwa_catalog.paths import CatalogLayout


def test_reliability_config_defaults() -> None:
    cfg = ReliabilityConfig()
    assert cfg.resid_rms_thresh_jy == 1.0
    assert cfg.resid_mean_thresh_jy == 1.0
    assert cfg.resid_percentile_lo == 1.0
    assert cfg.resid_percentile_hi == 99.0
    assert cfg.jitter_bmaj_frac == 0.3
    assert cfg.min_elevation_deg == 10.0
    assert cfg.max_source_ellipticity == 3.0
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


def test_passes_multi_image_subband_seed_only() -> None:
    row = pd.Series(
        {
            "n_lst_contributions": 1,
            "origin_band": "82MHz",
            "bands_present": "82MHz",
        }
    )
    assert unique_assoc_band_count(row)[1] == 1
    assert bool(flag_single_unique_band(pd.DataFrame([row])).iloc[0]) is True
    assert passes_multi_image(row) is False


def test_subband_invalid_astrometry_and_flux_qa() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0, np.nan],
            "DEC": [20.0, 20.0],
            "origin_band": ["82MHz", "82MHz"],
            "astrometry_band": ["82MHz", "82MHz"],
            "Peak_flux_82MHz": [1.5, 1.0],
            "Total_flux_82MHz": [1.4, 1.0],
            "E_Peak_flux_82MHz": [0.1, 0.1],
            "E_Total_flux_82MHz": [0.1, 0.1],
        }
    )
    assert is_subband_metacatalog(df)
    invalid = flag_invalid_astrometry_flux(df)
    assert bool(invalid.iloc[0]) is False
    assert bool(invalid.iloc[1]) is True
    row = df.iloc[0]
    frame = flux_qa_frame(row, None, qa_band_for_row(row))
    assert float(frame.iloc[0]["Peak_flux"]) == 1.5
    assert not flag_has_nan(df).iloc[0]


def test_representative_peak_flux_subband() -> None:
    df = pd.DataFrame(
        {
            "bands_present": ["82MHz,18MHz", "18MHz"],
            "Peak_flux_82MHz": [2.0, np.nan],
            "Peak_flux_18MHz": [1.0, 0.5],
        }
    )
    rep = representative_peak_flux(df)
    assert float(rep.iloc[0]) == 2.0
    assert float(rep.iloc[1]) == 0.5


def test_read_metacatalog_auto_validates_subband(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    meta = pd.DataFrame(
        {
            "meta_id": [0],
            "RA": [10.0],
            "DEC": [20.0],
            "origin_band": ["82MHz"],
            "bands_present": ["82MHz"],
            "astrometry_band": ["82MHz"],
            "Peak_flux_82MHz": [1.0],
        }
    )
    write_metacatalog(meta, layout, required=SUBBAND_METACATALOG_REQUIRED_COLUMNS, schema=None)
    loaded = read_metacatalog(layout)
    assert "Peak_flux" not in loaded.columns
    assert float(loaded.iloc[0]["Peak_flux_82MHz"]) == 1.0


def test_seed_band_from_discovery_mhz() -> None:
    bands = ("18MHz", "55MHz", "82MHz")
    assert seed_band_from_discovery(bands) == "82MHz"
    assert seed_band_from_discovery(("Full", "Blue")) == "Full"


def test_assign_source_quality_flags_subband(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    high = pd.DataFrame(
        [
            _src(
                ra=30.0,
                dec=37.0,
                peak=2.0,
                lst_hour="01h",
                band="82MHz",
                source_id=101,
            )
        ]
    )
    low = pd.DataFrame(
        [
            _src(
                ra=30.01,
                dec=37.0,
                peak=1.5,
                lst_hour="01h",
                band="18MHz",
                source_id=201,
            )
        ]
    )
    write_sources_catalog(high, layout, "01h", "82MHz")
    write_sources_catalog(low, layout, "01h", "18MHz")
    lst_high = merge_lst_metacatalog([high], band="82MHz")
    lst_low = merge_lst_metacatalog([low], band="18MHz")
    write_lst_merged(lst_high, layout, "82MHz")
    write_lst_merged(lst_low, layout, "18MHz")
    meta = build_subband_metacatalog(
        {"82MHz": lst_high, "18MHz": lst_low},
        seed_band="82MHz",
        assoc_bands=("18MHz",),
        color_bands=("82MHz", "18MHz"),
        band_freq_hz={"82MHz": 82e6, "18MHz": 18e6},
    )
    write_metacatalog(meta, layout, required=SUBBAND_METACATALOG_REQUIRED_COLUMNS, schema=None)
    lst_merged = {"82MHz": lst_high, "18MHz": lst_low}
    result = assign_source_quality_flags(
        meta,
        layout,
        lst_merged=lst_merged,
        vlssr=pd.DataFrame(
            {"RA": [30.0], "DEC": [37.0], "Peak_flux": [1.0], "BMAJ": [0.5], "BMIN": [0.5]}
        ),
    )
    assert len(result.catalog) == len(meta)
    assert is_subband_metacatalog(result.catalog)
    assert not bool(result.flags["invalid"].all())


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
    assert len(legend) == 16
    assert set(legend["bit"]) == set(range(16))
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
            "low_elevation": [False, False],
            "high_ellipticity": [False, False],
            "extended": [False, False],
            "large_single": [False, False],
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


def test_filter_by_quality_flags_any_all_none() -> None:
    lst = int(SourceQualityFlag.SINGLE_LST)
    vlssr = int(SourceQualityFlag.NO_VLSSR)
    both = lst | vlssr
    df = pd.DataFrame(
        {
            "meta_id": [0, 1, 2, 3],
            "quality_flag": [0, lst, vlssr, both],
        }
    )
    names = ["SINGLE_LST", "NO_VLSSR"]
    any_hit = filter_by_quality_flags(df, names, match="any")
    assert set(any_hit["meta_id"]) == {1, 2, 3}
    all_hit = filter_by_quality_flags(df, names, match="all")
    assert set(all_hit["meta_id"]) == {3}
    none_hit = filter_by_quality_flags(df, names, match="none")
    assert set(none_hit["meta_id"]) == {0}
    assert len(filter_by_quality_flags(df, [], match="any")) == 4
    with pytest.raises(ValueError, match="Unknown quality flag"):
        filter_by_quality_flags(df, ["NOT_A_BIT"])


def test_filter_by_quality_mask_default_247() -> None:
    from lwa_catalog.analyze.reliability import quality_flag_mask_from_names
    from lwa_catalog.constants import DEFAULT_QUALITY_FLAG_MASK

    expected = quality_flag_mask_from_names(
        [
            "HAS_NAN",
            "INVALID_ASTROMETRY",
            "SINGLE_LST",
            "UNPHYSICAL_FLUX",
            "RESID_ABS_FAIL",
            "RESID_PCTL_RMS",
            "RESID_PCTL_MEAN",
        ]
    )
    assert DEFAULT_QUALITY_FLAG_MASK == expected == 247

    lst = int(SourceQualityFlag.SINGLE_LST)
    no_vlssr = int(SourceQualityFlag.NO_VLSSR)
    df = pd.DataFrame(
        {
            "meta_id": [0, 1, 2, 3],
            "quality_flag": [0, no_vlssr, lst, lst | no_vlssr],
        }
    )
    kept = filter_by_quality_mask(df)
    assert set(kept["meta_id"]) == {0, 1}
    lst_only = filter_by_quality_mask(df, mask=lst)
    assert set(lst_only["meta_id"]) == {0, 1}
    assert len(filter_by_quality_mask(pd.DataFrame({"meta_id": [0]}))) == 1


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


def test_flag_extended() -> None:
    compact = pd.DataFrame({"Maj": [0.5], "BMAJ_match": [0.5]})
    extended = pd.DataFrame({"Maj": [2.0], "BMAJ_match": [0.5]})
    assert bool(flag_extended(compact, bmaj_ratio=3.0).iloc[0]) is False
    assert bool(flag_extended(extended, bmaj_ratio=3.0).iloc[0]) is True

    packed = pack_quality_flags(pd.DataFrame({"extended": [True]}))
    assert int(packed[0]) == int(SourceQualityFlag.EXTENDED)
    assert decode_quality_flag(int(packed[0])) == ["EXTENDED"]


def test_large_single_composite_logic() -> None:
    flags = pd.DataFrame(
        {
            "extended": [False, True, False, True, False],
            "high_ellipticity": [False, False, True, True, True],
            "single_unique_band": [False, False, True, False, False],
            "single_lst": [False, True, False, False, False],
        }
    )
    flags["large_single"] = (
        (flags["extended"] | flags["high_ellipticity"])
        & (flags["single_unique_band"] | flags["single_lst"])
    )
    packed = pack_quality_flags(flags)

    # row 0: no large/elongated and no singleton marker
    assert int(packed[0]) == 0
    assert bool(flags["large_single"].iloc[0]) is False

    # row 1: extended + single_lst => LARGE_SINGLE set
    assert int(packed[1] & np.uint32(SourceQualityFlag.LARGE_SINGLE)) != 0
    assert bool(flags["large_single"].iloc[1]) is True

    # row 2: high_ellipticity + single_unique_band => LARGE_SINGLE set
    assert int(packed[2] & np.uint32(SourceQualityFlag.LARGE_SINGLE)) != 0
    assert bool(flags["large_single"].iloc[2]) is True

    # row 3: both large criteria but no singleton marker => LARGE_SINGLE clear
    assert int(packed[3] & np.uint32(SourceQualityFlag.LARGE_SINGLE)) == 0
    assert bool(flags["large_single"].iloc[3]) is False

    # row 4: high_ellipticity only => LARGE_SINGLE clear
    assert int(packed[4] & np.uint32(SourceQualityFlag.LARGE_SINGLE)) == 0
    assert bool(flags["large_single"].iloc[4]) is False


def test_flag_low_elevation_and_high_ellipticity() -> None:
    from lwa_catalog.analyze.reliability import OR_HESL_EXCLUDE_MASK, filter_or_hesl
    from lwa_catalog.create.merge import source_elevation_deg

    low_row = pd.DataFrame(
        {
            "RA": [180.0],
            "DEC": [-50.0],
            "representative_lst": ["12h"],
            "Maj": [0.5],
            "Min": [0.5],
        }
    )
    high_row = pd.DataFrame(
        {
            "RA": [180.0],
            "DEC": [37.0],
            "representative_lst": ["12h"],
            "Maj": [6.0],
            "Min": [1.0],
        }
    )
    assert float(source_elevation_deg(180.0, -50.0, "12h")) < 10.0
    assert bool(flag_low_elevation(low_row, min_deg=10.0).iloc[0]) is True
    assert bool(flag_low_elevation(high_row, min_deg=10.0).iloc[0]) is False
    assert bool(flag_high_ellipticity(high_row, max_ratio=3.0).iloc[0]) is True
    assert bool(flag_high_ellipticity(low_row, max_ratio=3.0).iloc[0]) is False

    packed = pack_quality_flags(
        pd.DataFrame(
            {
                "low_elevation": [True],
                "high_ellipticity": [True],
            }
        )
    )
    assert int(packed[0]) == int(SourceQualityFlag.LOW_ELEVATION | SourceQualityFlag.HIGH_ELLIPTICITY)


def test_filter_or_hesl_or_and_combo() -> None:
    from lwa_catalog.analyze.reliability import OR_HESL_EXCLUDE_MASK, filter_or_hesl

    df = pd.DataFrame(
        {
            "meta_id": [0, 1, 2, 3, 4],
            "quality_flag": np.uint32(
                [
                    0,
                    int(SourceQualityFlag.SINGLE_LST),
                    int(SourceQualityFlag.HIGH_ELLIPTICITY),
                    int(
                        SourceQualityFlag.SINGLE_LST | SourceQualityFlag.HIGH_ELLIPTICITY
                    ),
                    int(SourceQualityFlag.LOW_ELEVATION),
                ]
            ),
        }
    )
    kept = filter_or_hesl(df)
    assert kept["meta_id"].tolist() == [0, 1, 2]
    assert OR_HESL_EXCLUDE_MASK == 4595


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
