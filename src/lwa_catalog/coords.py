"""Sky-coordinate helpers for catalog tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def wrap_ra_degrees(ra: Any) -> np.ndarray:
    """Map right ascension to ``[0, 360)`` degrees.

    PyBDSF / FITS WCS often return longitude on a negative branch
    (e.g. ``RA - 360``). ``NaN`` values are preserved.
    """
    arr = np.asarray(ra, dtype=float)
    return np.mod(arr, 360.0)


def normalize_ra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``RA`` / ``RA_*`` columns wrapped to ``[0, 360)``.

    Error columns such as ``E_RA`` are left unchanged.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in out.columns:
        if col == "RA" or col.startswith("RA_"):
            out[col] = wrap_ra_degrees(out[col].to_numpy())
    return out
