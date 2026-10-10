"""Tests for cross-match radius configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    LWA_CROSSMATCH_RADIUS_BEAM,
    NEDLVS_LWA_RADIUS,
    NEDLVS_REFERENCE_RADIUS_FIXED,
    VLSSR_REFERENCE_RADIUS_FIXED,
    catalog_match_frame,
    describe_crossmatch_radius,
    match_radius_deg,
    resolve_localization_sigma_deg,
)
from lwa_catalog.constants import (
    NEDLVS_CATALOG_POSITION_SIGMA_ARCSEC,
    NEDLVS_DEFAULT_POSITION_SIGMA_SCALE,
    NVSS_BMAJ_ARCSEC,
    VLASS_BMAJ_ARCSEC,
    VLSSR_BMAJ_ARCSEC,
    VLSSR_POSITION_ERROR_ARCSEC,
)


def test_fixed_radius_requires_arcsec() -> None:
    with pytest.raises(ValueError, match="fixed_arcsec"):
        CrossmatchRadiusSpec(mode="fixed")


def test_match_radius_fixed_arcsec() -> None:
    spec = CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=30.0)
    row = pd.Series({"RA": 0.0, "DEC": 0.0})
    assert match_radius_deg(row, spec) == pytest.approx(30.0 / 3600.0)


def test_vlssr_default_radius_is_fixed_position_error() -> None:
    row = pd.Series({})
    assert match_radius_deg(row, VLSSR_REFERENCE_RADIUS_FIXED) == pytest.approx(
        VLSSR_POSITION_ERROR_ARCSEC / 3600.0
    )
    assert VLSSR_REFERENCE_RADIUS_FIXED.mode == "fixed"
    assert VLSSR_REFERENCE_RADIUS_FIXED.fixed_arcsec == VLSSR_POSITION_ERROR_ARCSEC


def test_match_radius_beam_from_bmaj_match() -> None:
    row = pd.Series({"BMAJ_match": 0.25, "BMAJ": 0.1})
    assert match_radius_deg(row, LWA_CROSSMATCH_RADIUS_BEAM) == pytest.approx(0.25)


def test_match_radius_beam_fallback_default_arcsec() -> None:
    spec = CrossmatchRadiusSpec(mode="beam", default_arcsec=VLSSR_BMAJ_ARCSEC)
    row = pd.Series({"RA": 0.0, "DEC": 0.0})
    assert match_radius_deg(row, spec) == pytest.approx(VLSSR_BMAJ_ARCSEC / 3600.0)


def test_match_radius_localization_from_errors() -> None:
    spec = CrossmatchRadiusSpec(mode="localization", default_arcsec=5.0)
    row = pd.Series({"E_RA": 0.001, "E_DEC": 0.0005})
    assert match_radius_deg(row, spec) == pytest.approx(float(np.hypot(0.001, 0.0005)))


def test_localization_prefers_match_sigma_deg() -> None:
    row = pd.Series(
        {
            "match_sigma_deg": 1e-4,
            "match_source": "VLASS",
            "E_RA": 0.01,
            "E_DEC": 0.01,
        }
    )
    assert resolve_localization_sigma_deg(row, default_deg=5.0 / 3600.0) == pytest.approx(1e-4)
    spec = CrossmatchRadiusSpec(mode="localization", sigma_scale=3.0, default_arcsec=5.0)
    assert match_radius_deg(row, spec) == pytest.approx(3.0 * 1e-4)


def test_localization_uses_match_source_default() -> None:
    row = pd.Series({"match_source": "VLASS", "E_RA": 0.01, "E_DEC": 0.01})
    assert resolve_localization_sigma_deg(row, default_deg=5.0 / 3600.0) == pytest.approx(
        VLASS_BMAJ_ARCSEC / 3600.0
    )


def test_nedlvs_radius_defaults() -> None:
    assert NEDLVS_LWA_RADIUS.mode == "localization"
    assert NEDLVS_LWA_RADIUS.sigma_scale == pytest.approx(NEDLVS_DEFAULT_POSITION_SIGMA_SCALE)
    assert NEDLVS_REFERENCE_RADIUS_FIXED.mode == "fixed"
    assert NEDLVS_REFERENCE_RADIUS_FIXED.fixed_arcsec == pytest.approx(
        NEDLVS_CATALOG_POSITION_SIGMA_ARCSEC
    )


def test_catalog_match_frame_applies_spec() -> None:
    catalog = pd.DataFrame({"RA": [10.0], "DEC": [20.0], "BMAJ_match": [0.5]})
    frame = catalog_match_frame(catalog, LWA_CROSSMATCH_RADIUS_BEAM)
    assert frame.iloc[0]["BMAJ"] == pytest.approx(0.5)


def test_describe_crossmatch_radius_fixed() -> None:
    spec = CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=45.0)
    assert "fixed 45" in describe_crossmatch_radius(spec)


def test_fixed_radius_with_sigma_scale() -> None:
    spec = CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=NVSS_BMAJ_ARCSEC, sigma_scale=2.0)
    row = pd.Series({})
    assert match_radius_deg(row, spec) == pytest.approx(2.0 * NVSS_BMAJ_ARCSEC / 3600.0)
