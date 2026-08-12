"""Metacatalog summary helpers.

Stub / minimal implementations that can grow as notebook logic is extracted.
"""

from __future__ import annotations

import pandas as pd


def summarize_metacatalog(catalog: pd.DataFrame) -> dict[str, object]:
    """Return basic counts and band origin breakdown for a metacatalog.

    Parameters
    ----------
    catalog
        Global metacatalog table.

    Returns
    -------
    dict
        Summary fields suitable for logging or notebook display.
    """
    summary: dict[str, object] = {
        "n_sources": int(len(catalog)),
        "columns": list(catalog.columns),
    }
    if "origin_band" in catalog.columns:
        summary["origin_band_counts"] = (
            catalog["origin_band"].value_counts(dropna=False).to_dict()
        )
    return summary


def bands_present_counts(catalog: pd.DataFrame) -> pd.Series:
    """Count how often each band appears in ``bands_present``.

    Expects a comma-separated ``bands_present`` column as written by the
    metacatalog notebook.
    """
    if "bands_present" not in catalog.columns:
        msg = "catalog missing bands_present column"
        raise ValueError(msg)

    counts: dict[str, int] = {}
    for cell in catalog["bands_present"].astype(str):
        for band in (part.strip() for part in cell.split(",")):
            if band and band.lower() != "nan":
                counts[band] = counts.get(band, 0) + 1
    return pd.Series(counts, dtype=int).sort_index()
