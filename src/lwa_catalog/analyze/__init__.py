"""Analyze and summarize metacatalog tables.

Logic to extract from ``notebooks/metacatalog_sky_view.ipynb`` and related QA
workflows (band coverage, association stats, sky selection helpers).
"""

from __future__ import annotations

from lwa_catalog.analyze.summary import bands_present_counts, summarize_metacatalog

__all__ = ["bands_present_counts", "summarize_metacatalog"]
