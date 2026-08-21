"""Tests for RA wrapping helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lwa_catalog.coords import normalize_ra_columns, wrap_ra_degrees
from lwa_catalog.create.merge import merge_lst_metacatalog


def test_wrap_ra_degrees_maps_to_0_360() -> None:
    ra = np.array([-10.0, 0.0, 180.0, 350.0, 370.0, np.nan])
    out = wrap_ra_degrees(ra)
    assert out[0] == 350.0
    assert out[1] == 0.0
    assert out[2] == 180.0
    assert out[3] == 350.0
    assert out[4] == 10.0
    assert np.isnan(out[5])


def test_normalize_ra_columns_wraps_ra_and_ra_band_only() -> None:
    df = pd.DataFrame(
        {
            "RA": [-10.0, 370.0],
            "RA_Blue": [-350.0, 10.0],
            "E_RA": [0.01, 0.02],
            "DEC": [1.0, 2.0],
        }
    )
    out = normalize_ra_columns(df)
    assert float(out.loc[0, "RA"]) == 350.0
    assert float(out.loc[1, "RA"]) == 10.0
    assert float(out.loc[0, "RA_Blue"]) == 10.0
    assert float(out.loc[0, "E_RA"]) == 0.01


def test_merge_lst_wraps_negative_ra_from_sources() -> None:
    catalogs = [
        pd.DataFrame(
            [
                {
                    "RA": -10.0,
                    "DEC": 20.0,
                    "Peak_flux": 1.0,
                    "Total_flux": 1.0,
                    "Maj": 0.1,
                    "Min": 0.05,
                    "PA": 0.0,
                    "DC_Maj": 0.1,
                    "DC_Min": 0.05,
                    "DC_PA": 0.0,
                    "BMAJ": 0.5,
                    "lst_hour": "12h",
                    "band": "Blue",
                    "source_file": "x.fits",
                }
            ]
        )
    ]
    merged = merge_lst_metacatalog(catalogs, band="Blue")
    assert float(merged.iloc[0]["RA"]) == 350.0
