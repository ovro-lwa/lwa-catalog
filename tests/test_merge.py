"""Tests for LST / band metacatalog merge."""

from __future__ import annotations

import pandas as pd

from lwa_catalog.create.merge import build_global_metacatalog, merge_lst_metacatalog


def _src(
    *,
    ra: float,
    dec: float,
    peak: float,
    lst_hour: str,
    band: str,
    bmaj: float = 0.5,
    total: float | None = None,
) -> dict:
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": peak if total is None else total,
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
    green_only = meta[meta["origin_band"] == "Green"].iloc[0]
    assert green_only["bands_present"] == "Green"
