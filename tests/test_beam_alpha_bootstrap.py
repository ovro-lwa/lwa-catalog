"""Unit tests for VLSSR + neighbor beam-α bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.beam_alpha_bootstrap import (
    BeamAlphaBootstrapConfig,
    bootstrap_beam_alpha,
    load_beam_alpha_mapping,
    summarize_beam_alpha_bootstrap,
    write_beam_alpha_table,
)
from lwa_catalog.create.healpix_detect import DEFAULT_BEAM_SCALE, resolve_beam_scale
from lwa_catalog.paths import CatalogLayout


def _row(
    *,
    meta_id: int,
    bands: dict[str, tuple[float, float]],
    vlssr_peak: float | None = None,
) -> dict:
    """Build one metacatalog-like row.

    *bands* maps band → ``(total, peak)`` with n_assoc=n_confused=1.
    """
    row: dict = {"meta_id": meta_id}
    for band, (total, peak) in bands.items():
        row[f"Total_flux_{band}"] = total
        row[f"Peak_flux_{band}"] = peak
        row[f"n_assoc_{band}"] = 1
        row[f"n_confused_{band}"] = 1
    if vlssr_peak is not None:
        row["Peak_flux_VLSSR"] = vlssr_peak
        row["n_assoc_VLSSR"] = 1
    return row


def test_resolve_beam_scale_scalar_and_mapping() -> None:
    assert resolve_beam_scale("73MHz", 1.2) == pytest.approx(1.2)
    assert resolve_beam_scale("73MHz", {"73MHz": 1.3, "78MHz": 1.25}) == pytest.approx(
        1.3
    )
    assert resolve_beam_scale("18MHz", {"73MHz": 1.3}, default=1.1) == pytest.approx(
        1.1
    )
    assert resolve_beam_scale("18MHz", {"73MHz": 1.3}) == pytest.approx(
        DEFAULT_BEAM_SCALE
    )
    with pytest.raises(ValueError, match="finite"):
        resolve_beam_scale("73MHz", 0.0)


def test_bootstrap_anchor_and_neighbor_step(tmp_path) -> None:
    # Catalog built at α_used=1.0; Total_73 / VLSSR = 1.44 → α_73 = 1.2
    # Neighbor 73→78: T ratio 1.0, P ratio 0.81 → α_78/α_73 = √(0.81/1)=0.9
    rows = [
        _row(
            meta_id=1,
            bands={
                "73MHz": (1.44, 1.0),
                "78MHz": (1.44, 0.81),
            },
            vlssr_peak=1.0,
        ),
        _row(
            meta_id=2,
            bands={
                "73MHz": (1.44, 1.0),
                "78MHz": (1.44, 0.81),
            },
            vlssr_peak=1.0,
        ),
    ]
    df = pd.DataFrame(rows)
    result = bootstrap_beam_alpha(
        df,
        config=BeamAlphaBootstrapConfig(
            anchor_band="73MHz",
            beam_scale_used=1.0,
        ),
    )
    assert result.alpha_anchor == pytest.approx(1.2)
    assert result.median_total_over_vlssr == pytest.approx(1.44)
    mapping = result.as_mapping()
    assert mapping["73MHz"] == pytest.approx(1.2)
    assert mapping["78MHz"] == pytest.approx(1.2 * 0.9)

    out = write_beam_alpha_table(result, CatalogLayout(tmp_path))
    assert out.is_file()
    loaded = load_beam_alpha_mapping(CatalogLayout(tmp_path))
    assert loaded["73MHz"] == pytest.approx(1.2)
    assert loaded["78MHz"] == pytest.approx(1.08)
    text = summarize_beam_alpha_bootstrap(result)
    assert "α=" in text


def test_bootstrap_requires_vlssr() -> None:
    df = pd.DataFrame(
        [
            _row(
                meta_id=1,
                bands={"73MHz": (1.0, 1.0), "78MHz": (1.0, 1.0)},
            )
        ]
    )
    with pytest.raises(ValueError, match="Peak_flux_VLSSR"):
        bootstrap_beam_alpha(df)


def test_confused_rows_excluded() -> None:
    good = _row(
        meta_id=1,
        bands={"73MHz": (1.0, 1.0), "78MHz": (1.0, 1.0)},
        vlssr_peak=1.0,
    )
    bad = _row(
        meta_id=2,
        bands={"73MHz": (10.0, 1.0), "78MHz": (100.0, 1.0)},
        vlssr_peak=1.0,
    )
    bad["n_confused_78MHz"] = 2
    df = pd.DataFrame([good, bad])
    result = bootstrap_beam_alpha(
        df,
        config=BeamAlphaBootstrapConfig(beam_scale_used=1.0),
    )
    # Only the good row enters the neighbor median → T and P ratios both 1
    assert result.as_mapping()["78MHz"] == pytest.approx(result.as_mapping()["73MHz"])
