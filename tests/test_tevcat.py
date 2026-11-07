"""Tests for Wakely & Horan 2008 TeVCat cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.associations import (
    ASSOCIATION_SPEC_TEVCAT,
    association_rows_from_match,
    harmonize_class_native,
)
from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    TEVCAT_REFERENCE_RADIUS_FIXED,
)
from lwa_catalog.analyze.tevcat import (
    TevcatMatchConfig,
    _finalize_tevcat_frame,
    filter_tevcat_by_subcatalog,
    match_catalog_to_tevcat,
    select_unique_tevcat_matches,
    summarize_tevcat_match,
)
from lwa_catalog.constants import TEVCAT_DEFAULT_EXTENT_ARCMIN


def _tev_row(
    *,
    tev_id: int,
    name: str,
    ra: float,
    dec: float,
    source_type: str = "HBL",
    subcatalog: str = "Default Catalog",
    size_x: float | None = None,
    size_y: float | None = None,
) -> dict:
    return {
        "id": tev_id,
        "tevcat_name": name,
        "canonical_name": name.replace("TeV ", ""),
        "other_names": "",
        "ra": ra,
        "dec": dec,
        "glon": np.nan,
        "glat": np.nan,
        "source_type": source_type,
        "subcatalog": subcatalog,
        "discovery_date": "",
        "discovered_by": "",
        "distance_kpc": np.nan,
        "redshift": np.nan,
        "flux_crab": np.nan,
        "energy_threshold": np.nan,
        "spectral_index": np.nan,
        "extended": size_x is not None,
        "size_x": size_x if size_x is not None else np.nan,
        "size_y": size_y if size_y is not None else np.nan,
        "variability": pd.NA,
        "notes": "",
    }


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


def test_finalize_tevcat_sets_bmaj_from_size() -> None:
    raw = pd.DataFrame(
        [
            _tev_row(
                tev_id=1,
                name="TeV J0000+000",
                ra=10.0,
                dec=20.0,
                size_x=0.5,
                size_y=0.2,
            ),
            _tev_row(tev_id=2, name="TeV J0100+000", ra=15.0, dec=25.0),
        ]
    )
    df = _finalize_tevcat_frame(raw)
    assert len(df) == 2
    assert "RA" in df.columns and "DEC" in df.columns
    assert float(df.loc[0, "BMAJ"]) == pytest.approx(0.5)
    assert float(df.loc[1, "BMAJ"]) == pytest.approx(
        TEVCAT_DEFAULT_EXTENT_ARCMIN / 60.0
    )
    assert float(df.loc[0, "POS_ERR"]) == pytest.approx(0.5 * 3600.0)


def test_filter_tevcat_by_subcatalog() -> None:
    raw = pd.DataFrame(
        [
            _tev_row(
                tev_id=1, name="TeV A", ra=1.0, dec=1.0, subcatalog="Default Catalog"
            ),
            _tev_row(
                tev_id=2, name="TeV B", ra=2.0, dec=2.0, subcatalog="Other Sources"
            ),
        ]
    )
    df = _finalize_tevcat_frame(raw)
    kept = filter_tevcat_by_subcatalog(df, ("Default Catalog",))
    assert len(kept) == 1
    assert kept.iloc[0]["tevcat_name"] == "TeV A"


def test_match_catalog_to_tevcat_unique() -> None:
    tev = _finalize_tevcat_frame(
        pd.DataFrame(
            [
                _tev_row(
                    tev_id=74,
                    name="TeV J0534+220",
                    ra=83.63,
                    dec=22.01,
                    source_type="PWN",
                    size_x=0.05,
                    size_y=0.05,
                ),
                _tev_row(
                    tev_id=75,
                    name="TeV J1104+382",
                    ra=166.08,
                    dec=38.19,
                    source_type="HBL",
                ),
            ]
        )
    )
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=83.63, dec=22.01, bmaj=0.1),
            _meta_row(meta_id=2, ra=200.0, dec=0.0, bmaj=0.1),
        ]
    )
    result = match_catalog_to_tevcat(
        meta,
        tevcat=tev,
        config=TevcatMatchConfig(
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=TEVCAT_REFERENCE_RADIUS_FIXED,
        ),
    )
    assert result.summary["n_meta_matched"] == 1
    assert result.summary["n_meta_unique"] == 1
    assert result.summary["n_tevcat_matched"] == 1
    unique = select_unique_tevcat_matches(result.meta_flags)
    assert len(unique) == 1
    assert int(unique.iloc[0]["meta_id"]) == 1

    text = summarize_tevcat_match(result)
    assert "TeVCat footprint" in text

    rows = association_rows_from_match(
        result.meta_flags, result.tevcat_footprint, ASSOCIATION_SPEC_TEVCAT
    )
    assert len(rows) == 1
    assert rows.iloc[0]["catalog"] == "TeVCat"
    assert rows.iloc[0]["ref_id"] == "TeV J0534+220"
    assert rows.iloc[0]["class_native"] == "PWN"


def test_harmonize_tevcat_classes() -> None:
    assert harmonize_class_native("TeVCat", "HBL") == "agn"
    assert harmonize_class_native("TeVCat", "FSRQ") == "agn"
    assert harmonize_class_native("TeVCat", "PWN") == "tev"
    assert harmonize_class_native("TeVCat", "UNID") == "tev"
    assert harmonize_class_native("TeVCat", "") == "tev"
