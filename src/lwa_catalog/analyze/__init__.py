"""Analyze and summarize metacatalog tables.

Logic to extract from ``notebooks/metacatalog_sky_view.ipynb`` and related QA
workflows (band coverage, association stats, sky selection helpers).
"""

from __future__ import annotations

from lwa_catalog.analyze.healpix_map import (
    metacatalog_to_healpix,
    metacatalog_to_hips,
    write_healpix_hips,
)
from lwa_catalog.analyze.reliability import (
    ReliabilityConfig,
    ReliabilityResult,
    assert_gold_subset_of_cleaned,
    filter_metacatalog_exclude,
    filter_metacatalog_include,
    filter_metacatalog_reliability,
)
from lwa_catalog.analyze.summary import bands_present_counts, summarize_metacatalog
from lwa_catalog.analyze.trace import (
    SourceTrace,
    plot_maj_min_scatter,
    plot_member_property_scatter,
    plot_peak_flux_vs_lst,
    plot_ra_dec_scatter,
    preferred_trace_columns,
    rematch_meta_source,
)

__all__ = [
    "ReliabilityConfig",
    "ReliabilityResult",
    "SourceTrace",
    "assert_gold_subset_of_cleaned",
    "bands_present_counts",
    "filter_metacatalog_exclude",
    "filter_metacatalog_include",
    "filter_metacatalog_reliability",
    "metacatalog_to_healpix",
    "metacatalog_to_hips",
    "plot_maj_min_scatter",
    "plot_member_property_scatter",
    "plot_peak_flux_vs_lst",
    "plot_ra_dec_scatter",
    "preferred_trace_columns",
    "rematch_meta_source",
    "summarize_metacatalog",
    "write_healpix_hips",
]
