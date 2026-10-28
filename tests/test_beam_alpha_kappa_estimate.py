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
