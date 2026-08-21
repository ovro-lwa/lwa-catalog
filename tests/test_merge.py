"""Tests for LST / band metacatalog merge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lwa_catalog.constants import BAND_FREQ_HZ
from lwa_catalog.create.merge import (
    add_spectral_indices,
    build_global_metacatalog,
    merge_lst_metacatalog,
)


def _src(
    *,
    ra: float,
    dec: float,
    peak: float,
    lst_hour: str,
    band: str,
    bmaj: float = 0.5,
    total: float | None = None,
    e_total: float | None = None,
) -> dict:
    total_flux = peak if total is None else total
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": total_flux,
        "E_Total_flux": 0.1 * total_flux if e_total is None else e_total,
        "Maj": 0.1,
        "Min": 0.05,
        "PA": 0.0,
        "DC_Maj": 0.1,
        "DC_Min": 0.05,
        "DC_PA": 0.0,
        "BMAJ": bmaj,
        "lst_hour": lst_hour,
        "band": band,
        "source_file": f"{band}_{lst_hour}.fits",
    }


def test_merge_lst_clusters_nearby_detections() -> None:
    catalogs = [
        pd.DataFrame([_src(ra=10.0, dec=20.0, peak=1.0, lst_hour="01h", band="Full")]),
        pd.DataFrame([_src(ra=10.01, dec=20.0, peak=1.2, lst_hour="02h", band="Full")]),
        pd.DataFrame([_src(ra=50.0, dec=0.0, peak=0.5, lst_hour="01h", band="Full")]),
    ]
    merged = merge_lst_metacatalog(catalogs, band="Full")
    assert len(merged) == 2
    assert int(merged.loc[merged["RA"].between(9, 11), "n_lst_contributions"].iloc[0]) == 2


def test_merge_lst_picks_highest_elevation_and_flux_std() -> None:
    # Source near RA=30° (LST 02h). Peak flux is higher at 01h so the old
    # median-flux rule would prefer 01h; elevation at transit prefers 02h.
    catalogs = [
        pd.DataFrame([_src(ra=30.0, dec=37.0, peak=3.0, lst_hour="01h", band="Full")]),
        pd.DataFrame([_src(ra=30.1, dec=37.0, peak=1.0, lst_hour="02h", band="Full")]),
        pd.DataFrame([_src(ra=30.0, dec=37.0, peak=2.0, lst_hour="03h", band="Full")]),
    ]
    merged = merge_lst_metacatalog(catalogs, band="Full")
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["representative_lst"] == "02h"
    assert float(row["Peak_flux"]) == 1.0
    assert abs(float(row["Peak_flux_std"]) - float(np.std([3.0, 1.0, 2.0], ddof=1))) < 1e-12


def test_merge_lst_single_detection_has_nan_flux_std() -> None:
    catalogs = [
        pd.DataFrame([_src(ra=50.0, dec=0.0, peak=0.5, lst_hour="01h", band="Full")]),
    ]
    merged = merge_lst_metacatalog(catalogs, band="Full")
    assert len(merged) == 1
    assert merged.iloc[0]["representative_lst"] == "01h"
    assert np.isnan(float(merged.iloc[0]["Peak_flux_std"]))


def test_build_global_associates_matching_bands() -> None:
    full = pd.DataFrame(
        [
            {
                **_src(ra=10.0, dec=20.0, peak=2.0, lst_hour="01h", band="Full"),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    blue = pd.DataFrame(
        [
            {
                **_src(ra=10.05, dec=20.0, peak=1.5, lst_hour="01h", band="Blue"),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    green = pd.DataFrame(
        [
            {
                **_src(ra=80.0, dec=-10.0, peak=0.8, lst_hour="01h", band="Green"),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    red = pd.DataFrame([])
    meta = build_global_metacatalog(
        {"Full": full, "Blue": blue, "Green": green, "Red": red}
    )
    # One Full+Blue association + one Green-only seed
    assert len(meta) == 2
    assoc = meta[meta["origin_band"] == "Full"].iloc[0]
    assert "Blue" in str(assoc["bands_present"])
    assert float(assoc["Peak_flux_Blue"]) == 1.5
    assert "alpha_RG" in meta.columns
    assert "alpha_GB" in meta.columns
    assert np.isnan(float(assoc["alpha_GB"]))  # Blue only — no Green pair
    green_only = meta[meta["origin_band"] == "Green"].iloc[0]
    assert green_only["bands_present"] == "Green"


def test_add_spectral_indices_rg_gb_and_errors() -> None:
    nu_r = BAND_FREQ_HZ["Red"]
    nu_g = BAND_FREQ_HZ["Green"]
    nu_b = BAND_FREQ_HZ["Blue"]
    # Choose fluxes for a known α = -0.7 on both pairs
    alpha_true = -0.7
    s_g = 10.0
    s_r = s_g * (nu_r / nu_g) ** alpha_true
    s_b = s_g * (nu_b / nu_g) ** alpha_true
    e_r, e_g, e_b = 0.1 * s_r, 0.1 * s_g, 0.1 * s_b

    df = pd.DataFrame(
        [
            {
                "Total_flux_Red": s_r,
                "Total_flux_Green": s_g,
                "Total_flux_Blue": s_b,
                "E_Total_flux_Red": e_r,
                "E_Total_flux_Green": e_g,
                "E_Total_flux_Blue": e_b,
            },
            {
                # Missing Blue → only RG
                "Total_flux_Red": s_r,
                "Total_flux_Green": s_g,
                "Total_flux_Blue": np.nan,
                "E_Total_flux_Red": e_r,
                "E_Total_flux_Green": e_g,
                "E_Total_flux_Blue": np.nan,
            },
            {
                # Non-positive flux → no α
                "Total_flux_Red": -1.0,
                "Total_flux_Green": s_g,
                "Total_flux_Blue": s_b,
                "E_Total_flux_Red": e_r,
                "E_Total_flux_Green": e_g,
                "E_Total_flux_Blue": e_b,
            },
        ]
    )
    out = add_spectral_indices(df)
    assert abs(float(out.loc[0, "alpha_RG"]) - alpha_true) < 1e-10
    assert abs(float(out.loc[0, "alpha_GB"]) - alpha_true) < 1e-10
    assert np.isfinite(float(out.loc[0, "E_alpha_RG"]))
    assert np.isfinite(float(out.loc[0, "E_alpha_GB"]))
    expected_e_rg = np.sqrt((e_r / s_r) ** 2 + (e_g / s_g) ** 2) / abs(np.log(nu_r / nu_g))
    assert abs(float(out.loc[0, "E_alpha_RG"]) - expected_e_rg) < 1e-12

    assert abs(float(out.loc[1, "alpha_RG"]) - alpha_true) < 1e-10
    assert np.isnan(float(out.loc[1, "alpha_GB"]))
    assert np.isnan(float(out.loc[2, "alpha_RG"]))
    assert abs(float(out.loc[2, "alpha_GB"]) - alpha_true) < 1e-10


def test_build_global_computes_alpha_when_rgb_associated() -> None:
    full = pd.DataFrame(
        [
            {
                **_src(ra=10.0, dec=20.0, peak=2.0, lst_hour="01h", band="Full"),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    blue = pd.DataFrame(
        [
            {
                **_src(
                    ra=10.02,
                    dec=20.0,
                    peak=1.5,
                    total=12.0,
                    lst_hour="01h",
                    band="Blue",
                ),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    green = pd.DataFrame(
        [
            {
                **_src(
                    ra=10.03,
                    dec=20.0,
                    peak=1.4,
                    total=10.0,
                    lst_hour="01h",
                    band="Green",
                ),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    red = pd.DataFrame(
        [
            {
                **_src(
                    ra=10.04,
                    dec=20.0,
                    peak=1.3,
                    total=8.0,
                    lst_hour="01h",
                    band="Red",
                ),
                "n_lst_contributions": 1,
                "lst_hours": "01h",
                "representative_lst": "01h",
            }
        ]
    )
    meta = build_global_metacatalog(
        {"Full": full, "Blue": blue, "Green": green, "Red": red}
    )
    assert len(meta) == 1
    row = meta.iloc[0]
    assert set(str(row["bands_present"]).split(",")) >= {"Full", "Blue", "Green", "Red"}
    assert np.isfinite(float(row["alpha_RG"]))
    assert np.isfinite(float(row["alpha_GB"]))
    assert np.isfinite(float(row["E_alpha_RG"]))
    assert float(row["Total_flux_Red"]) == 8.0
    assert float(row["E_Total_flux_Green"]) == 1.0
