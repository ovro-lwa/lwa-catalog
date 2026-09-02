"""Normalize PyBDSF GAUL catalog columns."""

from __future__ import annotations

import pandas as pd

from lwa_catalog.constants import GAUL_STRING_COLUMNS


def cast_s_code_value(val: object) -> object:
    """Return a stripped PyBDSF ``S_Code`` string, or pandas NA."""
    if val is None or val is pd.NA:
        return pd.NA
    if isinstance(val, float) and pd.isna(val):
        return pd.NA
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("ascii", errors="replace")
    text = str(val).strip()
    return text if text else pd.NA


def cast_gaul_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast PyBDSF string GAUL columns (e.g. ``S_Code``) to pandas string dtype."""
    out = df
    for col in GAUL_STRING_COLUMNS:
        if col not in out.columns:
            continue
        if out is df:
            out = df.copy()
        out[col] = out[col].map(cast_s_code_value).astype("string")
    return out
