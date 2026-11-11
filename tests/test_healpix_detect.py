"""Tests for HEALPix tile detection glue."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

pytest.importorskip("lwa_healpix")

from lwa_healpix import iter_nested_tile_headers  # noqa: E402

from lwa_catalog.constants import OVRO_LATITUDE_DEG  # noqa: E402
from lwa_catalog.create.beam_scale_model import (  # noqa: E402
    HEALPIX_ATTACH_BPA_DEG,
    BeamScaleModel,
    resolve_beam_scale_model,
    zenith_south_elongation,
)
from lwa_catalog.create.healpix_detect import (  # noqa: E402
    DEFAULT_BEAM_SCALE,
    attach_beam_and_freq,
    detect_sources_on_healpix_tiles,
    median_beam_from_paths,
    restfreq_hz_from_header,
    scale_beam,
)


def _tiny_fits(path: Path, *, bmaj: float = 0.1, bmin: float = 0.08, bpa: float = 1.0) -> Path:
    hdr = fits.Header(
        {
            "BMAJ": bmaj,
            "BMIN": bmin,
            "BPA": bpa,
            "RESTFREQ": 5.5e7,
            "BUNIT": "JY/BEAM",
        }
    )
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(path)
    return path


def test_median_beam_from_paths(tmp_path: Path) -> None:
    p1 = _tiny_fits(tmp_path / "a.fits", bmaj=0.10, bmin=0.08, bpa=0.0)
    p2 = _tiny_fits(tmp_path / "b.fits", bmaj=0.20, bmin=0.12, bpa=10.0)
    bmaj, bmin, bpa = median_beam_from_paths([p1, p2])
    assert bmaj == pytest.approx(0.15)
    assert bmin == pytest.approx(0.10)
    assert bpa == pytest.approx(5.0)


def test_restfreq_hz_from_header() -> None:
    hdr = fits.Header({"RESTFRQ": 7.4e7})
    assert restfreq_hz_from_header(hdr) == pytest.approx(7.4e7)


def test_scale_beam_default() -> None:
    bmaj, bmin, bpa = scale_beam(0.10, 0.08, 12.0)
    assert bmaj == pytest.approx(0.10 * DEFAULT_BEAM_SCALE)
    assert bmin == pytest.approx(0.08 * DEFAULT_BEAM_SCALE)
    assert bpa == pytest.approx(12.0)
    assert DEFAULT_BEAM_SCALE == pytest.approx(1.0)


def test_scale_beam_identity_and_invalid() -> None:
    bmaj, bmin, bpa = scale_beam(0.10, 0.08, 5.0, scale=1.0)
    assert bmaj == pytest.approx(0.10)
    assert bmin == pytest.approx(0.08)
    assert bpa == pytest.approx(5.0)
    with pytest.raises(ValueError, match="beam scale"):
        scale_beam(0.1, 0.1, scale=0.0)
    with pytest.raises(ValueError, match="beam scale"):
        scale_beam(0.1, 0.1, scale=-1.0)
    with pytest.raises(ValueError, match="beam scale"):
        scale_beam(0.1, 0.1, scale=float("nan"))


def test_beam_scale_model_zenith_south() -> None:
    model = BeamScaleModel(alpha=1.2, kappa=0.01, phi_deg=OVRO_LATITUDE_DEG)
    assert model.elongation(OVRO_LATITUDE_DEG) == pytest.approx(1.0)
    assert model.elongation(OVRO_LATITUDE_DEG + 10.0) == pytest.approx(1.0)
    assert model.elongation(OVRO_LATITUDE_DEG - 50.0) == pytest.approx(1.5)
    s_maj, s_min = model.axis_scales(OVRO_LATITUDE_DEG - 50.0)
    assert s_min == pytest.approx(1.2)
    assert s_maj == pytest.approx(1.2 * 1.5)
    bmaj, bmin, bpa = model.effective_beam(0.2, 0.1, 3.0, dec_deg=OVRO_LATITUDE_DEG)
    assert bmaj == pytest.approx(0.24)
    assert bmin == pytest.approx(0.12)
    assert bpa == pytest.approx(HEALPIX_ATTACH_BPA_DEG)
    iso = BeamScaleModel.isotropic(1.1)
    assert iso.effective_beam(0.2, 0.1, dec_deg=0.0)[:2] == pytest.approx(
        scale_beam(0.2, 0.1, scale=1.1)[:2]
    )
    with pytest.raises(ValueError, match="kappa"):
        BeamScaleModel(alpha=1.0, kappa=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        BeamScaleModel(alpha=0.0, kappa=0.0)


def test_resolve_beam_scale_model() -> None:
    assert resolve_beam_scale_model("73MHz", 1.2).alpha == pytest.approx(1.2)
    assert resolve_beam_scale_model("73MHz", 1.2).kappa == pytest.approx(0.0)
    m = resolve_beam_scale_model(
        "73MHz",
        {"73MHz": {"alpha": 1.1, "kappa": 0.02}},
    )
    assert m.alpha == pytest.approx(1.1)
    assert m.kappa == pytest.approx(0.02)
    assert resolve_beam_scale_model("18MHz", {"73MHz": 1.3}).alpha == pytest.approx(
        DEFAULT_BEAM_SCALE
    )
    assert zenith_south_elongation(0.0, 0.01, phi_deg=37.0) == pytest.approx(1.37)


def test_attach_beam_and_freq() -> None:
    hdu = fits.PrimaryHDU(
        np.ones((8, 8), dtype=np.float32),
        header=fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}),
    )
    out = attach_beam_and_freq(
        hdu, bmaj=0.1, bmin=0.09, bpa=2.0, restfreq_hz=5e7, bunit="JY/BEAM",
    )
    assert out.header["BMAJ"] == 0.1
    assert out.header["BMIN"] == 0.09
    assert out.header["BPA"] == 2.0
    assert out.header["RESTFREQ"] == 5e7
    assert out.header["RESTFRQ"] == 5e7
    assert out.header["BUNIT"] == "JY/BEAM"


def test_iter_nested_tile_count() -> None:
    tiles = list(iter_nested_tile_headers(4, 16, overlap=0.0))
    assert len(tiles) == 12 * 4**2


def test_detect_sources_on_healpix_tiles_tags_ipix(monkeypatch: pytest.MonkeyPatch) -> None:
    nside_map = 8
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    weight[0:ratio] = 1.0  # only nested parent 0

    def _fake_pybdsf(hdu, *, bdsf_kw=None, return_background=False, **kwargs):
        table = Table(
            {
                "RA": [float(hdu.header["CRVAL1"])],
                "DEC": [float(hdu.header["CRVAL2"])],
                "Peak_flux": [1.0],
                "Total_flux": [1.0],
                "E_Peak_flux": [0.1],
                "E_Total_flux": [0.1],
                "Maj": [0.1],
                "Min": [0.1],
                "PA": [0.0],
                "DC_Maj": [0.1],
                "DC_Min": [0.1],
                "DC_PA": [0.0],
                "Resid_Isl_rms": [0.01],
                "Resid_Isl_mean": [0.0],
                "S_Code": ["S"],
            }
        )
        if return_background:
            return table, np.zeros(hdu.data.shape, dtype=np.float32)
        return table

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )

    df = detect_sources_on_healpix_tiles(
        healpix_map,
        weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        overlap=0.0,
        bmaj=0.1,
        bmin=0.1,
        bpa=0.0,
        restfreq_hz=5.5e7,
        band="Full",
        skip_empty=True,
        tier_thresholds=((3.0, 5.0),),
    )
    assert len(df) == 1
    assert int(df.iloc[0]["tile_ipix"]) == 0
    assert int(df.iloc[0]["nside_tile"]) == nside_tile
    assert df.iloc[0]["band"] == "Full"
    assert int(df.iloc[0]["detect_stage"]) == 1
    # Default beam_scale=1.0 (identity) on catalog beam columns
    assert df.iloc[0]["BMAJ"] == pytest.approx(0.1 * DEFAULT_BEAM_SCALE)
    assert df.iloc[0]["BMIN"] == pytest.approx(0.1 * DEFAULT_BEAM_SCALE)


def test_detect_sources_on_healpix_tiles_filters_ipix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside_map = 8
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    # Enable two nested parents (0 and 1).
    weight[0:ratio] = 1.0
    weight[ratio : 2 * ratio] = 1.0
    seen: list[int] = []

    def _fake_pybdsf(hdu, *, bdsf_kw=None, return_background=False, **kwargs):
        table = Table(
            {
                "RA": [float(hdu.header["CRVAL1"])],
                "DEC": [float(hdu.header["CRVAL2"])],
                "Peak_flux": [1.0],
                "Total_flux": [1.0],
                "E_Peak_flux": [0.1],
                "E_Total_flux": [0.1],
                "Maj": [0.1],
                "Min": [0.1],
                "PA": [0.0],
                "DC_Maj": [0.1],
                "DC_Min": [0.1],
                "DC_PA": [0.0],
                "Resid_Isl_rms": [0.01],
                "Resid_Isl_mean": [0.0],
                "S_Code": ["S"],
            }
        )
        if return_background:
            return table, np.zeros(hdu.data.shape, dtype=np.float32)
        return table

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )

    # Patch iterator to record which tiles would run, then filter to ipix=1 only.
    real_iter = iter_nested_tile_headers

    def _tracking_iter(*args, **kwargs):
        for ipix, hdr in real_iter(*args, **kwargs):
            seen.append(int(ipix))
            yield ipix, hdr

    monkeypatch.setattr(
        "lwa_healpix.iter_nested_tile_headers",
        _tracking_iter,
    )

    df = detect_sources_on_healpix_tiles(
        healpix_map,
        weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        overlap=0.0,
        bmaj=0.1,
        bmin=0.1,
        restfreq_hz=5.5e7,
        band="Full",
        skip_empty=True,
        tier_thresholds=((3.0, 5.0),),
        tile_ipix=(1,),
    )
    assert set(df["tile_ipix"].astype(int)) == {1}
    assert 0 in seen and 1 in seen  # iterator still yields both; filter skips 0
    with pytest.raises(ValueError, match="tile_ipix must be non-empty"):
        detect_sources_on_healpix_tiles(
            healpix_map,
            weight,
            nside_map=nside_map,
            nside_tile=nside_tile,
            bmaj=0.1,
            bmin=0.1,
            tile_ipix=(),
            tier_thresholds=((3.0, 5.0),),
        )


def test_detect_sources_on_healpix_tiles_beam_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside_map = 8
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    weight[0:ratio] = 1.0
    attached: list[tuple[float, float, float]] = []

    def _fake_pybdsf(hdu, *, bdsf_kw=None, return_background=False, **kwargs):
        attached.append(
            (float(hdu.header["BMAJ"]), float(hdu.header["BMIN"]), float(hdu.header["BPA"]))
        )
        table = Table(
            {
                "RA": [float(hdu.header["CRVAL1"])],
                "DEC": [float(hdu.header["CRVAL2"])],
                "Peak_flux": [1.0],
                "Total_flux": [1.0],
                "E_Peak_flux": [0.1],
                "E_Total_flux": [0.1],
                "Maj": [0.1],
                "Min": [0.1],
                "PA": [0.0],
                "DC_Maj": [0.1],
                "DC_Min": [0.1],
                "DC_PA": [0.0],
                "Resid_Isl_rms": [0.01],
                "Resid_Isl_mean": [0.0],
                "S_Code": ["S"],
            }
        )
        if return_background:
            return table, np.zeros(hdu.data.shape, dtype=np.float32)
        return table

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )

    common = dict(
        healpix_map=healpix_map,
        weight=weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        overlap=0.0,
        bmaj=0.2,
        bmin=0.1,
        bpa=3.0,
        restfreq_hz=5.5e7,
        band="Full",
        skip_empty=True,
        tier_thresholds=((3.0, 5.0),),
    )
    df_default = detect_sources_on_healpix_tiles(**common)
    assert df_default.iloc[0]["BMAJ"] == pytest.approx(0.2)
    assert df_default.iloc[0]["BMIN"] == pytest.approx(0.1)
    assert df_default.iloc[0]["BPA"] == pytest.approx(HEALPIX_ATTACH_BPA_DEG)
    assert attached[-1] == pytest.approx((0.2, 0.1, HEALPIX_ATTACH_BPA_DEG))

    df_scaled = detect_sources_on_healpix_tiles(**common, beam_scale=1.1)
    assert df_scaled.iloc[0]["BMAJ"] == pytest.approx(0.22)
    assert df_scaled.iloc[0]["BMIN"] == pytest.approx(0.11)
    assert attached[-1] == pytest.approx((0.22, 0.11, HEALPIX_ATTACH_BPA_DEG))

    # κ>0: southern tile gets larger BMAJ than a northern one at same α
    weight2 = np.zeros(npix, dtype=np.float64)
    weight2[0:ratio] = 1.0
    # nested parent 16 at nside_tile=2 has Dec ~ -19.5°
    weight2[16 * ratio : (16 + 1) * ratio] = 1.0
    model = BeamScaleModel(alpha=1.0, kappa=0.01, phi_deg=OVRO_LATITUDE_DEG)
    df_k = detect_sources_on_healpix_tiles(
        **{**common, "weight": weight2, "beam_scale": model, "bpa": 0.0}
    )
    assert set(df_k["tile_ipix"].astype(int)) == {0, 16}
    bmaj_by_ipix = {
        int(r.tile_ipix): float(r.BMAJ) for r in df_k.itertuples(index=False)
    }
    assert bmaj_by_ipix[16] > bmaj_by_ipix[0]
    assert df_k["BPA"].eq(HEALPIX_ATTACH_BPA_DEG).all()


def test_detect_sources_on_healpix_tiles_residual_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High tier locks center S; deep residual tier appends an offset source."""
    nside_map = 8
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    weight[0:ratio] = 1.0
    calls: list[dict] = []

    def _fake_pybdsf(hdu, *, bdsf_kw=None, return_background=False, **kwargs):
        kw = dict(bdsf_kw or {})
        calls.append(kw)
        ra = float(hdu.header["CRVAL1"])
        dec = float(hdu.header["CRVAL2"])
        isl = float(kw.get("thresh_isl", 0.0))
        if isl >= 9.0:
            table = Table(
                {
                    "RA": [ra],
                    "DEC": [dec],
                    "Peak_flux": [5.0],
                    "Total_flux": [5.0],
                    "E_Peak_flux": [0.1],
                    "E_Total_flux": [0.1],
                    "Maj": [0.1],
                    "Min": [0.1],
                    "PA": [0.0],
                    "DC_Maj": [0.1],
                    "DC_Min": [0.1],
                    "DC_PA": [0.0],
                    "Resid_Isl_rms": [0.01],
                    "Resid_Isl_mean": [0.0],
                    "S_Code": ["S"],
                    "Source_id": [1],
                    "Isl_id": [0],
                }
            )
            if return_background:
                return table, np.zeros(hdu.data.shape, dtype=np.float32)
            return table
        # Later tier on residual: new far source (no beam match to tier-1)
        table = Table(
            {
                "RA": [ra + 5.0],
                "DEC": [dec],
                "Peak_flux": [0.5],
                "Total_flux": [0.5],
                "E_Peak_flux": [0.1],
                "E_Total_flux": [0.1],
                "Maj": [0.1],
                "Min": [0.1],
                "PA": [0.0],
                "DC_Maj": [0.1],
                "DC_Min": [0.1],
                "DC_PA": [0.0],
                "Resid_Isl_rms": [0.01],
                "Resid_Isl_mean": [0.0],
                "S_Code": ["S"],
                "Source_id": [2],
                "Isl_id": [1],
            }
        )
        if return_background:
            return table, np.zeros(hdu.data.shape, dtype=np.float32)
        return table

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )

    df = detect_sources_on_healpix_tiles(
        healpix_map,
        weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        overlap=0.0,
        bmaj=0.5,
        bmin=0.5,
        bpa=0.0,
        beam_scale=1.0,
        restfreq_hz=5.5e7,
        band="Full",
        bdsf_kw={"quiet": True},
        tier_thresholds=((9.0, 11.0), (2.0, 3.0)),
        skip_empty=True,
    )
    assert len(calls) == 2
    assert float(calls[0]["thresh_isl"]) == 9.0
    assert float(calls[1]["thresh_isl"]) == 2.0
    assert len(df) == 2
    assert set(df["Source_id"].astype(int)) == {1, 2}
    assert set(df["detect_stage"].astype(int)) == {1, 2}


def test_detect_sources_on_healpix_tiles_background_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("healpy")
    nside_map = 16
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    weight[0:ratio] = 1.0  # only nested parent 0

    def _fake_pybdsf(hdu, *, bdsf_kw=None, return_background=False, **kwargs):
        table = Table({"RA": [float(hdu.header["CRVAL1"])], "DEC": [float(hdu.header["CRVAL2"])]})
        if return_background:
            return table, np.full(hdu.data.shape, -0.04, dtype=np.float32)
        return table

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )
    background = np.full(npix, np.nan)
    df = detect_sources_on_healpix_tiles(
        healpix_map,
        weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        bmaj=0.1,
        bmin=0.1,
        restfreq_hz=5.5e7,
        band="Full",
        skip_empty=True,
        background_out=background,
    )
    assert len(df) == 1
    assert background[:ratio] == pytest.approx(-0.04)
    assert np.isnan(background[ratio:]).all()
    with pytest.raises(ValueError, match="background_out shape"):
        detect_sources_on_healpix_tiles(
            healpix_map,
            weight,
            nside_map=nside_map,
            nside_tile=nside_tile,
            bmaj=0.1,
            bmin=0.1,
            background_out=np.zeros(npix + 1),
        )
