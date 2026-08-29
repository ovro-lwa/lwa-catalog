"""VLSSR cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lwa_catalog.analyze.reliability import parse_bands_present, resolve_bmaj
from lwa_catalog.constants import VLSSR_BMAJ_DEG, VLSSR_DEFAULT_PATH


def load_vlssr_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the VLSSR reference catalog from a whitespace-delimited text file.

    The file header is ``RA(2000) DEC(2000) "PEAK INT"``. Returned columns are
    ``RA``, ``DEC``, ``Peak_flux`` (Jy), ``BMAJ``, and ``BMIN`` (circular 80″
    PSF). Rows with non-finite RA/DEC are dropped.

    Parameters
    ----------
    path
        Catalog file path. Defaults to :data:`~lwa_catalog.constants.VLSSR_DEFAULT_PATH`.

    Returns
    -------
    pd.DataFrame
        VLSSR sources ready for :func:`~lwa_catalog.create.merge.associate_catalogs`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    catalog_path = Path(VLSSR_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"VLSSR catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(
        catalog_path,
        sep=r"\s+",
        skipinitialspace=True,
        quotechar='"',
    )
    df.columns = ["RA", "DEC", "Peak_flux"]
    df["BMAJ"] = VLSSR_BMAJ_DEG
    df["BMIN"] = VLSSR_BMAJ_DEG

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def select_blue_associated_rows(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return metacatalog rows with ``Blue`` in ``bands_present``.

    Original row indices are preserved for cross-match hit mapping.

    Raises
    ------
    ValueError
        If *catalog* lacks a ``bands_present`` column.
    """
    if "bands_present" not in catalog.columns:
        msg = "catalog missing bands_present column"
        raise ValueError(msg)

    keep: list[object] = []
    for idx, row in catalog.iterrows():
        if "Blue" in parse_bands_present(row):
            keep.append(idx)
    return catalog.loc[keep]


def _catalog_match_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """Build ``RA`` / ``DEC`` / ``BMAJ`` columns for beam-radius matching.

    Uses primary metacatalog ``RA``/``DEC`` and :func:`resolve_bmaj` for beam
    size. Rows with non-finite coordinates are omitted. Index matches *catalog*.
    """
    records: list[dict[str, float]] = []
    indices: list[object] = []
    for idx, row in catalog.iterrows():
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue
        bmaj = resolve_bmaj(row)
        if not np.isfinite(bmaj):
            bmaj = 0.0
        records.append({"RA": ra, "DEC": dec, "BMAJ": bmaj})
        indices.append(idx)

    if not records:
        return pd.DataFrame(columns=["RA", "DEC", "BMAJ"])
    return pd.DataFrame(records, index=indices)


def _footprint_filter_vlssr(vlssr: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep VLSSR rows whose Dec lies within the finite Dec range of *lwa*."""
    if vlssr.empty:
        return vlssr.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return vlssr.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return vlssr.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = float(finite_lwa.max())
    vlssr_dec = pd.to_numeric(vlssr["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(vlssr_dec) & (vlssr_dec >= dec_min) & (vlssr_dec <= dec_max)
    return vlssr.loc[keep].copy()
