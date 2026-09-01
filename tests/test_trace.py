"""Tests for metacatalog source rematch / trace helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.trace import (
    _band_palette,
    plot_member_property_scatter,
    plot_peak_flux_vs_lst,
    rematch_meta_source,
)
from lwa_catalog.constants import BAND_OVERLAY_COLORS
from lwa_catalog.create.merge import build_global_metacatalog, merge_lst_metacatalog
from lwa_catalog.io import write_lst_merged, write_metacatalog, write_sources_catalog
from lwa_catalog.paths import CatalogLayout


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
) -> dict:
    total_flux = peak if total is None else total
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": total_flux,
        "E_Total_flux": 0.1 * total_flux,
        "Maj": 0.1,
        "Min": 0.05,
        "PA": 0.0,
        "DC_Maj": 0.1,
        "DC_Min": 0.05,
        "DC_PA": 0.0,
        "BMAJ": bmaj,
        "lst_hour": lst_hour,
        "band": band,
        "Source_id": source_id,
        "source_file": f"{band}_{lst_hour}.fits",
    }


def _build_trace_layout(tmp_path: Path) -> tuple[CatalogLayout, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Synthetic Full+Blue tree near RA=30° with two LST hours."""
    layout = CatalogLayout(tmp_path)

    full_01 = pd.DataFrame(
        [_src(ra=30.0, dec=37.0, peak=2.0, lst_hour="01h", band="Full", source_id=101)]
    )
    full_02 = pd.DataFrame(
        [_src(ra=30.05, dec=37.0, peak=1.5, lst_hour="02h", band="Full", source_id=102)]
    )
    blue_01 = pd.DataFrame(
        [_src(ra=30.02, dec=37.0, peak=3.0, lst_hour="01h", band="Blue", source_id=201)]
    )
    blue_02 = pd.DataFrame(
        [_src(ra=30.06, dec=37.0, peak=1.0, lst_hour="02h", band="Blue", source_id=202)]
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


def test_rematch_recovers_durable_keys(tmp_path: Path) -> None:
    layout, meta, lst_merged = _build_trace_layout(tmp_path)
    assert len(meta) >= 1
    mid = int(meta.iloc[0]["meta_id"])
    trace = rematch_meta_source(meta, layout, meta_id=mid, lst_merged=lst_merged)

    assert trace.meta_id == mid
    assert not trace.lst_matches.empty
    bands = set(trace.lst_matches["band"].astype(str))
    assert "Full" in bands
    assert "Blue" in bands

    keys = {
        (str(r.band), str(r.lst_hour), int(r.Source_id))
        for r in trace.source_matches.itertuples()
    }
    assert ("Full", "01h", 101) in keys
    assert ("Full", "02h", 102) in keys
    assert ("Blue", "01h", 201) in keys
    assert ("Blue", "02h", 202) in keys


def test_rematch_picks_seeded_flux_among_blue_lst_rows(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    # Two Blue LST rows in the Full beam. Merge elevation-seeds Peak_flux_Blue=1.0
    # (02h); rematch must recover that seeded flux, not the brighter closer 01h row.
    full = pd.DataFrame(
        [
            {
                **_src(ra=30.0, dec=37.0, peak=2.0, lst_hour="02h", band="Full", source_id=1),
                "n_lst_contributions": 1,
                "lst_hours": "02h",
                "representative_lst": "02h",
            }
        ]
    )
    blue = pd.DataFrame(
        [
            {
                **_src(ra=30.05, dec=37.0, peak=3.0, lst_hour="01h", band="Blue", source_id=10),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            },
            {
                **_src(ra=30.08, dec=37.0, peak=1.0, lst_hour="02h", band="Blue", source_id=11),
                "n_lst_contributions": 1,
                "lst_hours": "02h",
                "representative_lst": "02h",
            },
        ]
    )
    write_lst_merged(full, layout, "Full")
    write_lst_merged(blue, layout, "Blue")
    write_lst_merged(pd.DataFrame(), layout, "Green")
    write_lst_merged(pd.DataFrame(), layout, "Red")
    write_sources_catalog(
        pd.DataFrame([_src(ra=30.08, dec=37.0, peak=1.0, lst_hour="02h", band="Blue", source_id=11)]),
        layout,
        "02h",
        "Blue",
    )

    meta = build_global_metacatalog(
        {"Full": full, "Blue": blue, "Green": pd.DataFrame(), "Red": pd.DataFrame()}
    )
    write_metacatalog(meta, layout)
    mid = int(meta.iloc[0]["meta_id"])
    assert float(meta.iloc[0]["Peak_flux_Blue"]) == 1.0
    trace = rematch_meta_source(meta, layout, meta_id=mid)

    blue_lst = trace.lst_matches[trace.lst_matches["band"] == "Blue"]
    assert len(blue_lst) == 1
    assert float(blue_lst.iloc[0]["Peak_flux"]) == 1.0
    assert int(blue_lst.iloc[0]["Source_id"]) == 11


def test_rematch_prefers_seeded_source_over_bright_beam_neighbor(tmp_path: Path) -> None:
    """Regression: faint seed + bright confused neighbor inside BMAJ (meta_id 17776-like)."""
    layout = CatalogLayout(tmp_path)
    seed = {
        **_src(ra=281.205, dec=9.578, peak=11.81, lst_hour="23h", band="Red", source_id=865, bmaj=0.34),
        "n_lst_contributions": 1,
        "lst_hours": "23h",
        "representative_lst": "23h",
    }
    neighbor = {
        **_src(ra=281.402, dec=9.877, peak=72.98, lst_hour="19h", band="Red", source_id=325, bmaj=0.37),
        "n_lst_contributions": 9,
        "lst_hours": "14h,15h,16h,17h,18h,19h,20h,21h,22h",
        "representative_lst": "19h",
    }
    red = pd.DataFrame([seed, neighbor])
    write_lst_merged(pd.DataFrame(), layout, "Full")
    write_lst_merged(pd.DataFrame(), layout, "Blue")
    write_lst_merged(pd.DataFrame(), layout, "Green")
    write_lst_merged(red, layout, "Red")
    write_sources_catalog(
        pd.DataFrame(
            [_src(ra=281.205, dec=9.578, peak=11.81, lst_hour="23h", band="Red", source_id=865, bmaj=0.34)]
        ),
        layout,
        "23h",
        "Red",
    )

    meta = build_global_metacatalog(
        {
            "Full": pd.DataFrame(),
            "Blue": pd.DataFrame(),
            "Green": pd.DataFrame(),
            "Red": red,
        }
    )
    # Keep only the Red-only seeded row matching the faint source
    meta = meta.loc[meta["Peak_flux"].between(11.0, 12.0)].reset_index(drop=True)
    assert len(meta) == 1
    meta["meta_id"] = 17776
    write_metacatalog(meta, layout)

    trace = rematch_meta_source(meta, layout, meta_id=17776)
    assert len(trace.lst_matches) == 1
    assert float(trace.lst_matches.iloc[0]["Peak_flux"]) == pytest.approx(11.81, rel=1e-3)
    assert int(trace.lst_matches.iloc[0]["Source_id"]) == 865
    assert set(trace.source_matches["Peak_flux"].round(2)) == {11.81}


def test_rematch_missing_sources_file_warns(tmp_path: Path) -> None:
    layout, meta, lst_merged = _build_trace_layout(tmp_path)
    # Remove one hour so rematch must warn and still return the other.
    (tmp_path / "sources_01h_Full.parquet").unlink()
    mid = int(meta.iloc[0]["meta_id"])
    trace = rematch_meta_source(meta, layout, meta_id=mid, lst_merged=lst_merged)
    assert any("sources_01h_Full.parquet" in w for w in trace.warnings)
    keys = {
        (str(r.band), str(r.lst_hour), int(r.Source_id))
        for r in trace.source_matches.itertuples()
        if str(r.band) == "Full"
    }
    assert ("Full", "02h", 102) in keys
    assert ("Full", "01h", 101) not in keys


def test_rematch_unknown_meta_id_raises(tmp_path: Path) -> None:
    layout, meta, _ = _build_trace_layout(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        rematch_meta_source(meta, layout, meta_id=999_999)


def test_associate_skips_nan_coordinates() -> None:
    from lwa_catalog.create.merge import associate_catalogs

    base = pd.DataFrame({"RA": [30.0], "DEC": [37.0], "BMAJ": [0.5]})
    band = pd.DataFrame(
        {
            "RA": [np.nan, 30.05, 90.0],
            "DEC": [37.0, 37.0, np.nan],
            "BMAJ": [0.5, 0.5, 0.5],
        }
    )
    hits, matched = associate_catalogs(base, band)
    assert hits.get(0) == [1]
    assert matched == {1}


def test_plot_helpers_return_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from lwa_catalog.analyze import plot_maj_min_scatter, plot_ra_dec_scatter

    df = pd.DataFrame(
        {
            "band": ["Full", "Blue", "Full"],
            "lst_hour": ["01h", "01h", "02h"],
            "Peak_flux": [1.0, 0.8, 1.1],
            "E_Peak_flux": [0.1, 0.05, 0.08],
            "Total_flux": [1.2, 0.9, 1.3],
            "E_Total_flux": [0.12, 0.06, 0.09],
            "RA": [10.0, 10.01, 10.02],
            "DEC": [20.0, 20.01, 20.02],
            "E_RA": [0.01, 0.01, 0.01],
            "E_DEC": [0.01, 0.01, 0.01],
            "Maj": [0.2, 0.18, 0.22],
            "Min": [0.1, 0.09, 0.11],
            "E_Maj": [0.02, 0.02, 0.02],
            "E_Min": [0.01, 0.01, 0.01],
            "Source_id": [1, 2, 3],
        }
    )
    ax1 = plot_peak_flux_vs_lst(df)
    ax2 = plot_member_property_scatter(df)
    ax3 = plot_ra_dec_scatter(df)
    ax4 = plot_maj_min_scatter(df)
    assert isinstance(ax1, Axes)
    assert isinstance(ax2, Axes)
    assert isinstance(ax3, Axes)
    assert isinstance(ax4, Axes)
    # Empty frame should not raise
    ax5 = plot_peak_flux_vs_lst(pd.DataFrame())
    ax6 = plot_member_property_scatter(pd.DataFrame())
    assert isinstance(ax5, Axes)
    assert isinstance(ax6, Axes)


def test_band_palette_subband_red_to_blue() -> None:
    palette = _band_palette(["18MHz", "55MHz", "82MHz"])
    assert palette["18MHz"] == BAND_OVERLAY_COLORS["Red"]
    assert palette["82MHz"] == BAND_OVERLAY_COLORS["Blue"]
    assert palette["18MHz"] != palette["55MHz"] != palette["82MHz"]
