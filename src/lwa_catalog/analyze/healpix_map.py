"""Catalog → HEALPix map helpers, with HiPS export via ``lwa-healpix``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lwa_catalog.paths import CatalogLayout

# FWHM → Gaussian σ (degrees): σ = FWHM / (2 √(2 ln 2))
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Columns passed to :func:`metacatalog_to_healpix` for Gaussian painting.
_HEALPIX_GAUSSIAN_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "Peak_flux",
    "Maj",
    "Min",
    "PA",
)


def _bmaj_array_for_associate(df: pd.DataFrame) -> np.ndarray:
    """Per-row BMAJ (degrees) for :func:`associate_catalogs`, else 0."""
    out = np.zeros(len(df), dtype=float)
    for key in ("BMAJ_match", "BMAJ_full", "BMAJ"):
        if key not in df.columns:
            continue
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
        need = out <= 0.0
        take = need & np.isfinite(vals) & (vals > 0.0)
        out[take] = vals[take]
    return out


def lst_merged_catalog_for_healpix(
    metacatalog: pd.DataFrame,
    *,
    band: str,
    layout: CatalogLayout | None = None,
    lst_merged: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select unique LST-merged Gaussians for band-native HiPS painting.

    Subband fusion stores ``Peak_flux_{band}`` but top-level ``Maj``/``Min``/``PA``
    come from ``astrometry_band`` (often a different frequency). For a residual
    against ``healpix_{band}``, paint the LST-merged catalog for *band* instead:
    rematch each metacatalog row that has a finite ``Peak_flux_{band}`` onto
    ``metacatalog_lst_{band}`` with the same beam matcher / seeded-flux pick as
    :func:`~lwa_catalog.analyze.trace.rematch_meta_source`, then collapse
    duplicate LST rows (confused reverse claims) so each Gaussian is painted once.

    Parameters
    ----------
    metacatalog
        Fused rows to include (e.g. quality-filtered ``core_clean``). Needs
        ``RA``, ``DEC``, ``Peak_flux_{band}``, and a BMAJ column
        (``BMAJ_match`` / ``BMAJ_full`` / ``BMAJ``).
    band
        LST / coadd band label (e.g. ``\"78MHz\"``).
    layout
        Catalog tree used to load ``metacatalog_lst_{band}`` when *lst_merged*
        is omitted.
    lst_merged
        Optional in-memory LST-merged table for *band* (skips disk read).

    Returns
    -------
    DataFrame
        Unique LST rows with ``RA``, ``DEC``, ``Peak_flux``, ``Maj``, ``Min``,
        ``PA`` suitable for :func:`metacatalog_to_healpix`. Empty if nothing
        rematches.
    """
    from lwa_catalog.analyze.trace import _pick_rematch_lst_row
    from lwa_catalog.create.merge import associate_catalogs
    from lwa_catalog.io import read_lst_merged

    if metacatalog is None or metacatalog.empty:
        return pd.DataFrame(columns=list(_HEALPIX_GAUSSIAN_COLUMNS))

    flux_col = f"Peak_flux_{band}"
    if flux_col not in metacatalog.columns:
        msg = f"metacatalog missing {flux_col!r} (needed to rematch LST band {band!r})"
        raise KeyError(msg)
    for col in ("RA", "DEC"):
        if col not in metacatalog.columns:
            msg = f"metacatalog missing {col!r}"
            raise KeyError(msg)

    if lst_merged is None:
        if layout is None:
            msg = "pass layout= or lst_merged= to load the LST-merged band catalog"
            raise ValueError(msg)
        loaded = read_lst_merged(layout, band, as_pandas=True)
        if not isinstance(loaded, pd.DataFrame):
            msg = f"expected DataFrame from read_lst_merged({band!r})"
            raise TypeError(msg)
        lst_merged = loaded

    if lst_merged.empty:
        return pd.DataFrame(columns=list(_HEALPIX_GAUSSIAN_COLUMNS))
    for col in ("RA", "DEC", "Peak_flux"):
        if col not in lst_merged.columns:
            msg = f"LST-merged catalog for {band!r} missing {col!r}"
            raise KeyError(msg)

    flux = pd.to_numeric(metacatalog[flux_col], errors="coerce")
    base = metacatalog.loc[np.isfinite(flux)].copy()
    if base.empty:
        return pd.DataFrame(columns=list(_HEALPIX_GAUSSIAN_COLUMNS))
    base = base.reset_index(drop=True)
    base["BMAJ"] = _bmaj_array_for_associate(base)

    lst = lst_merged.reset_index(drop=True).copy()
    if "BMAJ" in lst.columns:
        lst["BMAJ"] = pd.to_numeric(lst["BMAJ"], errors="coerce").fillna(0.0)
    else:
        lst["BMAJ"] = 0.0

    matches, _matched = associate_catalogs(base, lst)
    chosen_ilocs: list[int] = []
    for base_i, band_ilocs in matches.items():
        ilocs = list(band_ilocs)
        if not ilocs:
            continue
        hits = lst.iloc[ilocs].reset_index(drop=True)
        pick = _pick_rematch_lst_row(hits, base.iloc[int(base_i)], band)
        chosen_ilocs.append(int(ilocs[int(pick.name)]))

    if not chosen_ilocs:
        return pd.DataFrame(columns=list(_HEALPIX_GAUSSIAN_COLUMNS))

    uniq = sorted(set(chosen_ilocs))
    out = lst.iloc[uniq].copy()
    keep = [c for c in _HEALPIX_GAUSSIAN_COLUMNS if c in out.columns]
    # Maj/Min/PA may be absent on odd tables; metacatalog_to_healpix floors them.
    for col in _HEALPIX_GAUSSIAN_COLUMNS:
        if col not in out.columns and col in {"Maj", "Min", "PA"}:
            out[col] = np.nan
            keep.append(col)
    return out.loc[:, keep].reset_index(drop=True)


def _require_healpy():
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised when healpy absent
        msg = (
            "healpy is required for HEALPix maps; "
            "install with: pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return hp


def _require_lwa_healpix():
    try:
        from lwa_healpix import healpix_to_hips
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa-healpix is required for HiPS export; "
            "install with: pip install 'lwa-catalog[analyze]' "
            "(or pip install /path/to/lwa-healpix)"
        )
        raise ImportError(msg) from exc
    return healpix_to_hips


def _point_healpix(
    ra: np.ndarray,
    dec: np.ndarray,
    amp: np.ndarray,
    *,
    nside: int,
    nest: bool,
) -> np.ndarray:
    """Deposit each source amplitude into a single HEALPix pixel."""
    hp = _require_healpy()
    out = np.zeros(hp.nside2npix(int(nside)), dtype=float)
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(int(nside), theta, phi, nest=bool(nest))
    np.add.at(out, pix, amp)
    return out


def _gaussian_healpix(
    ra: np.ndarray,
    dec: np.ndarray,
    amp: np.ndarray,
    maj_fwhm: np.ndarray,
    min_fwhm: np.ndarray,
    pa_deg: np.ndarray,
    *,
    nside: int,
    nest: bool,
    n_sigma: float = 5.0,
    min_fwhm_deg: float | None = None,
) -> np.ndarray:
    """Paint elliptical Gaussians (Peak amplitude, Maj/Min FWHM, PA) onto HEALPix.

    ``Maj`` / ``Min`` are FWHM in degrees (PyBDSF convention). ``PA`` is the
    position angle of the major axis in degrees, measured from North toward East.
    """
    hp = _require_healpy()
    nside_i = int(nside)
    out = np.zeros(hp.nside2npix(nside_i), dtype=float)
    pix_deg = float(hp.nside2resol(nside_i, arcmin=True) / 60.0)
    floor = float(min_fwhm_deg) if min_fwhm_deg is not None else pix_deg

    for i in range(len(ra)):
        peak = float(amp[i])
        if not np.isfinite(peak) or peak == 0.0:
            continue
        ra0 = float(ra[i])
        dec0 = float(dec[i])
        if not (np.isfinite(ra0) and np.isfinite(dec0)):
            continue

        maj = float(maj_fwhm[i]) if np.isfinite(maj_fwhm[i]) else floor
        minor = float(min_fwhm[i]) if np.isfinite(min_fwhm[i]) else maj
        maj = max(abs(maj), floor)
        minor = max(abs(minor), floor)
        if minor > maj:
            maj, minor = minor, maj
        pa = float(pa_deg[i]) if np.isfinite(pa_deg[i]) else 0.0

        sig_maj = maj * _FWHM_TO_SIGMA
        sig_min = minor * _FWHM_TO_SIGMA
        radius_deg = float(n_sigma) * sig_maj
        radius_deg = max(radius_deg, 2.0 * pix_deg)
        radius_rad = np.deg2rad(radius_deg)

        theta0 = np.deg2rad(90.0 - dec0)
        phi0 = np.deg2rad(ra0)
        vec = hp.ang2vec(theta0, phi0)
        pixs = hp.query_disc(nside_i, vec, radius_rad, nest=bool(nest))
        if pixs.size == 0:
            continue

        th, ph = hp.pix2ang(nside_i, pixs, nest=bool(nest))
        dec_p = 90.0 - np.rad2deg(th)
        ra_p = np.rad2deg(ph)
        # Local offsets (degrees): East, North
        cos_dec = np.cos(np.deg2rad(dec0))
        d_east = (ra_p - ra0) * cos_dec
        d_north = dec_p - dec0

        pa_rad = np.deg2rad(pa)
        sin_pa = np.sin(pa_rad)
        cos_pa = np.cos(pa_rad)
        # u along major axis, v along minor (PA from N toward E)
        u = d_east * sin_pa + d_north * cos_pa
        v = d_east * cos_pa - d_north * sin_pa
        val = peak * np.exp(-0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2))
        out[pixs] += val

    return out


def metacatalog_to_healpix(
    catalog: pd.DataFrame,
    *,
    nside: int = 64,
    weight_col: str | None = "Peak_flux",
    nest: bool = False,
    profile: str = "gaussian",
    maj_col: str = "Maj",
    min_col: str = "Min",
    pa_col: str = "PA",
    n_sigma: float = 5.0,
    min_fwhm_deg: float | None = None,
) -> np.ndarray:
    """Project metacatalog sources onto an equatorial HEALPix map.

    Default ``profile="gaussian"`` paints each source as an elliptical Gaussian
    with amplitude from *weight_col* (typically ``Peak_flux``), FWHM axes from
    ``Maj`` / ``Min`` (degrees), and major-axis PA from ``PA`` (degrees, N→E).
    Use ``profile="point"`` to deposit amplitude into a single pixel (legacy).

    Parameters
    ----------
    catalog
        Table with ``RA``, ``DEC`` in degrees; shape columns for Gaussians.
    nside
        HEALPix NSIDE.
    weight_col
        Amplitude column; ``None`` uses unit amplitude (counts for point mode).
    nest
        If True, use NESTED numbering; else RING.
    profile
        ``\"gaussian\"`` (default) or ``\"point\"``.
    maj_col, min_col, pa_col
        Gaussian FWHM / PA column names (ignored for ``point``).
    n_sigma
        Paint extent in units of major-axis σ.
    min_fwhm_deg
        Floor for Maj/Min; default is one HEALPix pixel.

    Returns
    -------
    ndarray
        Length ``12 * nside**2`` map (zeros where empty).
    """
    hp = _require_healpy()
    npix = hp.nside2npix(int(nside))
    if catalog is None or catalog.empty:
        return np.zeros(npix, dtype=float)
    if "RA" not in catalog.columns or "DEC" not in catalog.columns:
        msg = "catalog needs RA and DEC columns"
        raise ValueError(msg)

    profile_key = str(profile).lower()
    if profile_key not in {"gaussian", "point"}:
        msg = f"profile must be 'gaussian' or 'point', got {profile!r}"
        raise ValueError(msg)

    ra = pd.to_numeric(catalog["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(catalog["DEC"], errors="coerce").to_numpy(dtype=float)
    if weight_col is None:
        amp = np.ones(len(catalog), dtype=float)
    else:
        if weight_col not in catalog.columns:
            msg = f"weight column {weight_col!r} not in catalog"
            raise ValueError(msg)
        amp = pd.to_numeric(catalog[weight_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(amp)
    if not ok.any():
        return np.zeros(npix, dtype=float)
    ra = ra[ok]
    dec = dec[ok]
    amp = amp[ok]

    if profile_key == "point":
        return _point_healpix(ra, dec, amp, nside=nside, nest=nest)

    def _col_or_nan(name: str) -> np.ndarray:
        if name in catalog.columns:
            return pd.to_numeric(catalog[name], errors="coerce").to_numpy(dtype=float)[ok]
        return np.full(int(ok.sum()), np.nan, dtype=float)

    maj = _col_or_nan(maj_col)
    minor = _col_or_nan(min_col)
    pa = _col_or_nan(pa_col)
    return _gaussian_healpix(
        ra,
        dec,
        amp,
        maj,
        minor,
        pa,
        nside=nside,
        nest=nest,
        n_sigma=n_sigma,
        min_fwhm_deg=min_fwhm_deg,
    )


def write_healpix_hips(
    m: np.ndarray,
    output_directory: str | Path,
    *,
    nest: bool = False,
    coord_frame: str = "equatorial",
    threads: bool = True,
    cut_percentiles: tuple[float, float] | None = (1.0, 99.0),
    properties: dict[str, str] | None = None,
) -> Path:
    """Write a HEALPix map as a HiPS tile set via :func:`lwa_healpix.healpix_to_hips`.

    Catalog maps from :func:`metacatalog_to_healpix` use equatorial RA/Dec, so the
    default ``coord_frame`` is ``\"equatorial\"``.

    Parameters
    ----------
    m
        1-D HEALPix map.
    output_directory
        Directory for HiPS tiles, ``properties``, and Aladin Lite ``index.html``.
    nest
        If True, *m* uses NESTED ordering.
    coord_frame
        Coordinate frame of *m* (``\"equatorial\"``, ``\"galactic\"``, …).
    threads
        Multi-threaded reprojection (set ``False`` in tests).
    cut_percentiles, properties
        Forwarded to :func:`lwa_healpix.healpix_to_hips`.

    Returns
    -------
    Path
        Resolved HiPS output directory.
    """
    healpix_to_hips = _require_lwa_healpix()
    out = Path(output_directory)
    # reproject_to_hips creates the directory itself (exist_ok=False)
    if out.exists() and any(out.iterdir()):
        msg = f"HiPS output directory is not empty: {out}"
        raise FileExistsError(msg)
    healpix_to_hips(
        np.asarray(m, dtype=float),
        coord_frame=coord_frame,
        output_directory=out,
        nested=bool(nest),
        threads=bool(threads),
        cut_percentiles=cut_percentiles,
        properties=properties,
    )
    return out.resolve()


def metacatalog_to_hips(
    catalog: pd.DataFrame,
    output_directory: str | Path,
    *,
    nside: int = 64,
    weight_col: str | None = "Peak_flux",
    nest: bool = False,
    coord_frame: str = "equatorial",
    threads: bool = True,
    properties: dict[str, str] | None = None,
    profile: str = "gaussian",
    **healpix_kwargs,
) -> Path:
    """Project a metacatalog to HEALPix (default Gaussian shapes) and write HiPS."""
    m = metacatalog_to_healpix(
        catalog,
        nside=nside,
        weight_col=weight_col,
        nest=nest,
        profile=profile,
        **healpix_kwargs,
    )
    return write_healpix_hips(
        m,
        output_directory,
        nest=nest,
        coord_frame=coord_frame,
        threads=threads,
        properties=properties,
    )
