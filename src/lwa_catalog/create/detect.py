"""Per-image source detection (PyBDSF).

Stub. Notebook reference: ``run_pybdsf_on_hdu`` / ``detect_sources`` in
``notebooks/ovro_lwa_metacatalog.ipynb``. Related conventions also live in
``image_plane_correction.source_detection``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def detect_sources(
    fits_path: str | Path,
    *,
    process_kw: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run PyBDSF on one FITS image and return a Gaussian catalog DataFrame.

    Parameters
    ----------
    fits_path
        Input image path.
    process_kw
        Extra keyword arguments for ``bdsf.process_image``.

    Returns
    -------
    pandas.DataFrame
        Gaussian catalog rows with provenance columns to be defined when
        extracted from the notebook.

    Raises
    ------
    NotImplementedError
        Until the notebook implementation is moved here.
    """
    raise NotImplementedError(
        "detect_sources is a stub; see notebooks/ovro_lwa_metacatalog.ipynb "
        f"(input would be {Path(fits_path)!s}, process_kw={process_kw})"
    )
