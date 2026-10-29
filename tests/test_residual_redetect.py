"""Tests for stage-3 residual re-detect helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.healpix_cutout import healpix_residual_path
from lwa_catalog.analyze.reliability import (
    SourceQualityFlag,
    flag_resid_leftover,
    pack_quality_flags,
)
from lwa_catalog.create.residual_redetect import (
    DETECT_STAGE_COL,
    RESID_LEFTOVER_COL,
    build_residual_healpix_map,
    mark_resid_leftover_on_metacatalog,
    merge_residual_into_stage2,
    summarize_residual_stage3,
)
from lwa_catalog.paths import CatalogLayout


def _src(ra: float, dec: float, peak: float = 1.0, bmaj: float = 0.5, **extra) -> dict:
    row = {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": peak,
        "Maj": bmaj,
        "Min": bmaj * 0.8,
        "PA": 90.0,
        "BMAJ": bmaj,
        "BMIN": bmaj * 0.8,
        "BPA": 90.0,
    }
    row.update(extra)
    return row


def test_healpix_residual_path() -> None:
    assert healpix_residual_path("/tmp/out", "73MHz", nside=2048) == Path(
        "/tmp/out/healpix_residual_73MHz_nside2048.fits"
    )
    assert CatalogLayout(Path("/tmp/out")).residual_matched("73MHz") == Path(
        "/tmp/out/residual_matched_73MHz.parquet"
    )


def test_build_residual_clips_negatives() -> None:
    nside = 2
    npix = 12 * nside**2
    obs = np.ones(npix, dtype=float)
    # Empty catalog → resid == obs; clip keeps positives.
    resid = build_residual_healpix_map(obs, None, nside=nside, nest=True, clip_negative=True)
    assert resid.shape == (npix,)
    assert np.all(resid[np.isfinite(resid)] >= 0.0)

    # Model brighter than obs → negative before clip → zero after.
    cat = pd.DataFrame([_src(0.0, 0.0, peak=10.0, bmaj=20.0)])
    resid2 = build_residual_healpix_map(obs, cat, nside=nside, nest=True, clip_negative=True)
    assert np.nanmin(resid2) >= 0.0

    resid_signed = build_residual_healpix_map(
        obs, cat, nside=nside, nest=True, clip_negative=False
    )
    assert np.nanmin(resid_signed) < 0.0


def test_build_residual_preserves_nan() -> None:
    nside = 2
    npix = 12 * nside**2
    obs = np.ones(npix, dtype=float)
    obs[0] = np.nan
    resid = build_residual_healpix_map(obs, None, nside=nside, nest=True)
    assert np.isnan(resid[0])


def test_merge_appends_unmatched_excludes_matched() -> None:
    stage2 = pd.DataFrame(
        [
            _src(10.0, 20.0, peak=5.0, bmaj=1.0, Source_id=1),
            _src(50.0, -10.0, peak=3.0, bmaj=1.0, Source_id=2),
        ]
    )
    residual = pd.DataFrame(
        [
            _src(10.05, 20.02, peak=0.4, bmaj=1.0),  # near source 1 → matched leftover
            _src(120.0, 5.0, peak=1.2, bmaj=1.0),  # new sky → append
        ]
    )
    result = merge_residual_into_stage2(stage2, residual, band="73MHz")
    assert result.summary["n_append"] == 1
    assert result.summary["n_matched"] == 1
    assert len(result.matched) == 1
    assert len(result.catalog) == 3
    assert int(result.catalog[DETECT_STAGE_COL].eq(3).sum()) == 1
    assert bool(result.catalog.loc[0, RESID_LEFTOVER_COL]) is True
    assert bool(result.catalog.loc[1, RESID_LEFTOVER_COL]) is False
    new = result.catalog[result.catalog[DETECT_STAGE_COL] == 3].iloc[0]
    assert new["RA"] == pytest.approx(120.0)
    assert "n_append=1" in summarize_residual_stage3(result)


def test_merge_empty_residual() -> None:
    stage2 = pd.DataFrame([_src(1.0, 2.0)])
    result = merge_residual_into_stage2(stage2, pd.DataFrame(), band="55MHz")
    assert len(result.catalog) == 1
    assert result.matched.empty
    assert result.summary["n_append"] == 0


def test_mark_resid_leftover_on_metacatalog() -> None:
    meta = pd.DataFrame(
        [
            _src(10.0, 20.0, peak=5.0, bmaj=1.0, meta_id=1),
            _src(80.0, 0.0, peak=2.0, bmaj=1.0, meta_id=2),
        ]
    )
    matched = pd.DataFrame(
        {
            "band": ["73MHz"],
            "RA_stage2": [10.02],
            "DEC_stage2": [20.01],
            "BMAJ_stage2": [1.0],
            "RA_residual": [10.03],
            "DEC_residual": [20.0],
            "BMAJ_residual": [1.0],
        }
    )
    out = mark_resid_leftover_on_metacatalog(meta, matched)
    assert bool(out.loc[0, RESID_LEFTOVER_COL]) is True
    assert bool(out.loc[1, RESID_LEFTOVER_COL]) is False


def test_flag_resid_leftover_packs_bit() -> None:
    df = pd.DataFrame({RESID_LEFTOVER_COL: [True, False]})
    flagged = flag_resid_leftover(df)
    assert list(flagged) == [True, False]
    packed = pack_quality_flags(pd.DataFrame({"resid_leftover": flagged}))
    assert packed[0] & np.uint32(SourceQualityFlag.RESID_LEFTOVER)
    assert packed[1] == 0
    assert int(SourceQualityFlag.RESID_LEFTOVER) == (1 << 19)


def test_write_residual_healpix_fits_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("lwa_healpix")
    pytest.importorskip("healpy")
    from lwa_healpix import read_healpix_fits, write_healpix_fits

    from lwa_catalog.create.residual_redetect import write_residual_healpix_fits

    nside = 8
    npix = 12 * nside**2
    obs = np.full(npix, 0.5, dtype=np.float32)
    weight = np.ones(npix, dtype=np.float32)
    write_healpix_fits(
        tmp_path / "healpix_73MHz_nside8.fits",
        obs,
        weight,
        nside=nside,
        nested=True,
        overwrite=True,
    )
    cat = pd.DataFrame([_src(45.0, 10.0, peak=2.0, bmaj=5.0)])
    out = write_residual_healpix_fits(
        tmp_path,
        "73MHz",
        cat,
        nside=nside,
        nested=True,
        clip_negative=True,
        overwrite=True,
    )
    assert out.is_file()
    resid, w, meta = read_healpix_fits(out)
    assert meta["nside"] == nside
    assert resid.shape == (npix,)
    assert np.nanmin(resid) >= 0.0
    assert w.shape == (npix,)
