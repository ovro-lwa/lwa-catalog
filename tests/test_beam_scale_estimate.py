"""Unit tests for per-band beam-scale estimation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.create.beam_scale import (
    DEFAULT_BEAM_SCALE_ALPHA_GRID,
    DEFAULT_BEAM_SCALE_PROBE_TILES,
    BeamScaleEstimateConfig,
    estimate_beam_scale,
    pick_best_alpha,
    summarize_beam_scale_estimate,
    tile_pixel_mask,
)
from lwa_catalog.create.healpix_detect import DEFAULT_BEAM_SCALE


def test_default_constants() -> None:
    assert DEFAULT_BEAM_SCALE == pytest.approx(1.0)
    assert DEFAULT_BEAM_SCALE_PROBE_TILES == (28, 102, 189)
    assert DEFAULT_BEAM_SCALE_ALPHA_GRID == (1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0)


def test_config_rejects_bad_grids() -> None:
    with pytest.raises(ValueError, match="probe_tiles"):
        BeamScaleEstimateConfig(probe_tiles=())
    with pytest.raises(ValueError, match="alpha_grid"):
        BeamScaleEstimateConfig(alpha_grid=())
    with pytest.raises(ValueError, match="alpha_grid"):
        BeamScaleEstimateConfig(alpha_grid=(1.0, -0.5))
    with pytest.raises(ValueError, match="min_probe_tiles"):
        BeamScaleEstimateConfig(min_probe_tiles=0)


def test_tile_pixel_mask_nested_children() -> None:
    nside_map, nside_tile = 8, 2
    factor = (nside_map // nside_tile) ** 2
    pix = tile_pixel_mask(nside_map, nside_tile, 3, nested=True)
    assert len(pix) == factor
    assert int(pix[0]) == 3 * factor
    assert int(pix[-1]) == 3 * factor + factor - 1
    with pytest.raises(ValueError, match="nested"):
        tile_pixel_mask(8, 2, 0, nested=False)
    with pytest.raises(ValueError, match="divisible"):
        tile_pixel_mask(8, 3, 0, nested=True)


def test_pick_best_alpha_mean_and_tiebreak() -> None:
    # α=1.2 uniquely best mean; α=1.5 has one very low tile but worse mean
    per_tile = pd.DataFrame(
        [
            {"alpha": 1.0, "tile_ipix": 28, "resid_rms": 1.0},
            {"alpha": 1.0, "tile_ipix": 102, "resid_rms": 1.0},
            {"alpha": 1.0, "tile_ipix": 189, "resid_rms": 1.0},
            {"alpha": 1.2, "tile_ipix": 28, "resid_rms": 0.5},
            {"alpha": 1.2, "tile_ipix": 102, "resid_rms": 0.5},
            {"alpha": 1.2, "tile_ipix": 189, "resid_rms": 0.5},
            {"alpha": 1.5, "tile_ipix": 28, "resid_rms": 0.1},
            {"alpha": 1.5, "tile_ipix": 102, "resid_rms": 0.9},
            {"alpha": 1.5, "tile_ipix": 189, "resid_rms": 0.9},
        ]
    )
    best, per_alpha, warn = pick_best_alpha(per_tile, min_probe_tiles=1)
    assert best == pytest.approx(1.2)
    assert warn == []
    row = per_alpha.set_index("alpha").loc[1.2]
    assert row["mean_resid_rms"] == pytest.approx(0.5)
    assert int(row["n_finite"]) == 3

    # Tie on mean → smallest α
    tied = pd.DataFrame(
        [
            {"alpha": 1.3, "tile_ipix": 28, "resid_rms": 0.4},
            {"alpha": 1.3, "tile_ipix": 102, "resid_rms": 0.4},
            {"alpha": 1.1, "tile_ipix": 28, "resid_rms": 0.4},
            {"alpha": 1.1, "tile_ipix": 102, "resid_rms": 0.4},
        ]
    )
    best_tie, _, _ = pick_best_alpha(tied)
    assert best_tie == pytest.approx(1.1)


def test_pick_best_alpha_fallback_empty_and_nan() -> None:
    best, per_alpha, warn = pick_best_alpha(pd.DataFrame(), fallback_scale=1.0)
    assert best == pytest.approx(1.0)
    assert per_alpha.empty
    assert any("fallback" in w for w in warn)

    nan_df = pd.DataFrame(
        [
            {"alpha": 1.0, "tile_ipix": 28, "resid_rms": np.nan},
            {"alpha": 1.2, "tile_ipix": 28, "resid_rms": np.nan},
        ]
    )
    best2, _, warn2 = pick_best_alpha(nan_df, fallback_scale=1.0)
    assert best2 == pytest.approx(1.0)
    assert any("NaN" in w for w in warn2)


def test_estimate_beam_scale_requires_tier2() -> None:
    npix = 12 * 8**2
    with pytest.raises(ValueError, match="tier2_bdsf_kw"):
        estimate_beam_scale(
            np.zeros(npix),
            np.ones(npix),
            nside_map=8,
            bmaj=0.1,
            bmin=0.1,
            tier2_bdsf_kw=None,
            config=BeamScaleEstimateConfig(
                probe_tiles=(0,),
                alpha_grid=(1.0,),
                nside_tile=2,
            ),
        )


def test_estimate_beam_scale_fallback_zero_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lwa_healpix")
    nside_map, nside_tile = 8, 2
    npix = 12 * nside_map**2
    healpix_map = np.zeros(npix, dtype=float)
    weight = np.zeros(npix, dtype=float)  # all empty → no usable probes

    # Should not call PyBDSF when no probes have weight
    monkeypatch.setattr(
        "lwa_catalog.create.beam_scale.run_pybdsf_on_hdu",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("PyBDSF should not run")),
    )

    with pytest.warns(UserWarning, match="fallback"):
        result = estimate_beam_scale(
            healpix_map,
            weight,
            nside_map=nside_map,
            bmaj=0.1,
            bmin=0.08,
            band="73MHz",
            tier2_bdsf_kw={},
            config=BeamScaleEstimateConfig(
                probe_tiles=(0, 1),
                alpha_grid=(1.0, 1.2),
                nside_tile=nside_tile,
                min_probe_tiles=1,
            ),
        )
    assert result.beam_scale == pytest.approx(1.0)
    assert result.summary["fallback"] is True
    assert any("usable probe" in w for w in result.warnings)
    text = summarize_beam_scale_estimate(result)
    assert "beam_scale=1" in text
