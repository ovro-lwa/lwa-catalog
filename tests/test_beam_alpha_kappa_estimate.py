"""Unit tests for joint (α, κ) beam-scale estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.beam_alpha_kappa_estimate import (
    BeamAlphaKappaEstimateConfig,
    estimate_beam_alpha_kappa,
    summarize_beam_alpha_kappa_estimate,
    write_beam_alpha_kappa_result,
)
from lwa_catalog.constants import OVRO_LATITUDE_DEG
from lwa_catalog.paths import CatalogLayout


def _synth_sources(
    *,
    alpha: float = 1.1,
    kappa: float = 0.01,
    bands: tuple[str, ...] = ("55MHz", "73MHz"),
    n_per_tile: int = 20,
) -> pd.DataFrame:
    """Compact S sources with Maj/Min matching α and E(δ;κ) vs header beam."""
    phi = OVRO_LATITUDE_DEG
    tiles = [
        (28, 54.0),
        (10, 42.0),
        (102, 0.0),
        (189, -19.5),
        (140, -30.0),
    ]
    bmaj = 0.18
    bmin = 0.16
    rows: list[dict] = []
    meta_id = 0
    for band in bands:
        for ipix, dec in tiles:
            u = max(0.0, phi - dec)
            e = 1.0 + kappa * u
            for _ in range(n_per_tile):
                rows.append(
                    {
                        "band": band,
                        "tile_ipix": ipix,
                        "DEC": dec,
                        "RA": 180.0,
                        "Peak_flux": 2.0,
                        "S_Code": "S",
                        "BMAJ": bmaj,
                        "BMIN": bmin,
                        "BPA": 90.0,
                        "Maj": alpha * e * bmaj,
                        "Min": alpha * bmin,
                        "meta_id": meta_id,
                    }
                )
                meta_id += 1
    return pd.DataFrame(rows)


def test_estimate_alpha_kappa_synthetic_dec_ladder(tmp_path) -> None:
    sources = _synth_sources(alpha=1.1, kappa=0.01)
    result = estimate_beam_alpha_kappa(
        sources,
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=1.0,
            apply_vlssr_hybrid=False,
            zenith_half_width_deg=20.0,
        ),
    )
    assert result.kappa == pytest.approx(0.01, rel=0.05, abs=0.001)
    for _, row in result.by_band.iterrows():
        assert row["alpha"] == pytest.approx(1.1, rel=0.02)
    text = summarize_beam_alpha_kappa_estimate(result)
    assert "shared κ=" in text

    layout = CatalogLayout(tmp_path)
    path = write_beam_alpha_kappa_result(result, layout)
    assert path.is_file()
    assert (layout.root / "beam_alpha_kappa_by_band.parquet").is_file()


def test_estimate_kappa_zero_when_all_zenith() -> None:
    # Only high-Dec tiles → u≈0 for Maj stretch term near zero residual after α
    phi = OVRO_LATITUDE_DEG
    rows = []
    for i in range(40):
        rows.append(
            {
                "band": "73MHz",
                "tile_ipix": 28,
                "DEC": phi + 5.0,
                "Peak_flux": 2.0,
                "S_Code": "S",
                "BMAJ": 0.2,
                "BMIN": 0.18,
                "Maj": 1.15 * 0.2,
                "Min": 1.15 * 0.18,
            }
        )
    df = pd.DataFrame(rows)
    result = estimate_beam_alpha_kappa(
        df,
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=1.0,
            apply_vlssr_hybrid=False,
        ),
    )
    assert result.kappa == pytest.approx(0.0)
    assert result.by_band.iloc[0]["alpha"] == pytest.approx(1.15, rel=0.02)


def test_estimate_alpha_floor_and_used_invert() -> None:
    # Catalog already at used=1.2 with Maj/BMAJ≈1 → true α=1.2
    rows = []
    for i in range(30):
        rows.append(
            {
                "band": "73MHz",
                "tile_ipix": 10,
                "DEC": OVRO_LATITUDE_DEG,
                "Peak_flux": 2.0,
                "S_Code": "S",
                "BMAJ": 0.2,
                "BMIN": 0.18,
                "Maj": 0.2,
                "Min": 0.18,
            }
        )
    df = pd.DataFrame(rows)
    result = estimate_beam_alpha_kappa(
        df,
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=1.2,
            apply_vlssr_hybrid=False,
        ),
    )
    assert result.by_band.iloc[0]["alpha"] == pytest.approx(1.2, rel=0.02)


def test_vlssr_hybrid_rescale() -> None:
    sources = _synth_sources(alpha=1.0, kappa=0.0, bands=("73MHz",))
    # Zenith VLSSR wants α=1.2 → s=1.2 when size α=1.0
    meta_rows = []
    for i in range(20):
        meta_rows.append(
            {
                "DEC": OVRO_LATITUDE_DEG,
                "Total_flux_73MHz": 1.44,
                "Peak_flux_VLSSR": 1.0,
                "n_assoc_VLSSR": 1,
            }
        )
    meta = pd.DataFrame(meta_rows)
    result = estimate_beam_alpha_kappa(
        sources,
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=1.0,
            apply_vlssr_hybrid=True,
            anchor_band="73MHz",
        ),
        metacatalog_vlssr=meta,
    )
    assert result.alpha_vlssr_anchor == pytest.approx(1.2)
    assert result.alpha_scale_factor == pytest.approx(1.2, rel=0.05)
    assert result.by_band.iloc[0]["alpha"] == pytest.approx(1.2, rel=0.05)


def test_vlssr_missing_columns_warning() -> None:
    sources = _synth_sources(alpha=1.0, kappa=0.0, bands=("73MHz",))
    meta = pd.DataFrame(
        {
            "DEC": [OVRO_LATITUDE_DEG] * 5,
            "Total_flux_73MHz": [1.0] * 5,
        }
    )
    result = estimate_beam_alpha_kappa(
        sources,
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=1.0,
            apply_vlssr_hybrid=True,
            anchor_band="73MHz",
        ),
        metacatalog_vlssr=meta,
    )
    assert result.n_vlssr_zenith == 0
    assert any("Peak_flux_VLSSR" in w for w in result.warnings)


def test_estimate_undoes_detect_time_alpha_kappa() -> None:
    """Catalog BMAJ already includes detect (α,κ); invert that model."""
    from lwa_catalog.create.beam_scale_model import BeamScaleModel

    phi = OVRO_LATITUDE_DEG
    alpha_used = 1.15
    kappa_used = 0.009
    used = {
        "73MHz": BeamScaleModel(alpha=alpha_used, kappa=kappa_used, phi_deg=phi),
    }
    bmaj_h, bmin_h = 0.18, 0.16
    tiles = [(28, 54.0), (10, 42.0), (102, 0.0), (189, -19.5), (140, -30.0)]
    rows = []
    for ipix, dec in tiles:
        u = max(0.0, phi - dec)
        e = 1.0 + kappa_used * u
        bmaj = alpha_used * e * bmaj_h
        bmin = alpha_used * bmin_h
        for _ in range(25):
            rows.append(
                {
                    "band": "73MHz",
                    "tile_ipix": ipix,
                    "DEC": dec,
                    "Peak_flux": 2.0,
                    "Total_flux": 2.0,
                    "S_Code": "S",
                    "BMAJ": bmaj,
                    "BMIN": bmin,
                    # Perfect size match to attached beam → recover used (α,κ).
                    "Maj": bmaj,
                    "Min": bmin,
                }
            )
    result = estimate_beam_alpha_kappa(
        pd.DataFrame(rows),
        config=BeamAlphaKappaEstimateConfig(
            beam_scale_used=used,
            apply_vlssr_hybrid=False,
        ),
    )
    assert result.by_band.iloc[0]["alpha"] == pytest.approx(alpha_used, rel=0.02)
    assert result.kappa == pytest.approx(kappa_used, rel=0.05, abs=0.001)


def test_predict_total_flux_scales_with_alpha_squared() -> None:
    from lwa_catalog.analyze.beam_alpha_kappa_estimate import (
        predict_total_flux_for_beam_scale,
    )

    df = pd.DataFrame(
        {
            "band": ["73MHz", "73MHz"],
            "DEC": [OVRO_LATITUDE_DEG, OVRO_LATITUDE_DEG],
            "Total_flux": [1.44, 2.88],
            "Peak_flux": [1.0, 2.0],
        }
    )
    pred = predict_total_flux_for_beam_scale(
        df, beam_scale_used=1.0, beam_scale_new=1.2
    )
    # Total' = Total * (1.0/1.2)^2
    assert pred.iloc[0] == pytest.approx(1.44 / 1.44)
    assert pred.iloc[1] == pytest.approx(2.88 / 1.44)
