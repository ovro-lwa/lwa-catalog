"""Fuse per-image catalogs into LST-band and global metacatalogs.

Stub. Notebook reference: ``merge_lst_metacatalog``,
``associate_band_into_metacatalog``, ``build_global_metacatalog`` in
``notebooks/ovro_lwa_metacatalog.ipynb``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def merge_lst_metacatalog(
    catalogs: Iterable[pd.DataFrame],
    *,
    band: str,
) -> pd.DataFrame:
    """Cross-match detections across LST hours within one band.

    Representative row per cluster: detection whose peak flux is nearest the
    median flux over the cluster (beam-sized matching).

    Raises
    ------
    NotImplementedError
        Until the notebook implementation is moved here.
    """
    raise NotImplementedError(
        f"merge_lst_metacatalog is a stub (band={band!r}); "
        "see notebooks/ovro_lwa_metacatalog.ipynb"
    )


def associate_band_into_metacatalog(
    metacatalog: pd.DataFrame,
    band_catalog: pd.DataFrame,
    band: str,
) -> pd.DataFrame:
    """Cross-match one color band onto the current metacatalog.

    Unmatched band-only sources are appended. Match radii are beam-sized.

    Raises
    ------
    NotImplementedError
        Until the notebook implementation is moved here.
    """
    raise NotImplementedError(
        f"associate_band_into_metacatalog is a stub (band={band!r}); "
        "see notebooks/ovro_lwa_metacatalog.ipynb"
    )


def build_global_metacatalog(
    lst_merged: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Fuse per-band LST-merged catalogs into one global metacatalog.

    Typical order: seed from Full, then associate Blue / Green / Red.

    Raises
    ------
    NotImplementedError
        Until the notebook implementation is moved here.
    """
    raise NotImplementedError(
        "build_global_metacatalog is a stub; "
        f"bands={sorted(lst_merged)}; see notebooks/ovro_lwa_metacatalog.ipynb"
    )
