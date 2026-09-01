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
    QualityFlagResult,
    ReliabilityConfig,
    ReliabilityResult,
    SourceQualityFlag,
    assert_gold_subset_of_cleaned,
    assign_source_quality_flags,
    decode_quality_flag,
    filter_by_quality_flags,
    filter_metacatalog_exclude,
    filter_metacatalog_include,
    filter_metacatalog_reliability,
    quality_flag_bit_counts,
    quality_flag_legend,
    quality_flag_mask_from_names,
    quality_flag_names,
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
from lwa_catalog.analyze.nedlvs import (
    NedlvsMatchConfig,
    NedlvsMatchResult,
    build_sfr_radio_luminosity_table,
    load_nedlvs_catalog,
    match_catalog_to_nedlvs,
    radio_luminosity_nu,
    resolve_highest_frequency_peak_flux,
    select_metacatalog,
    summarize_nedlvs_match,
)
from lwa_catalog.analyze.vlssr import (
    VlssrMatchConfig,
    VlssrMatchResult,
    load_vlssr_catalog,
    match_catalog_to_vlssr,
    select_blue_associated_rows,
    summarize_vlssr_match,
)

__all__ = [
    "QualityFlagResult",
    "ReliabilityConfig",
    "ReliabilityResult",
    "SourceQualityFlag",
    "SourceTrace",
    "assign_source_quality_flags",
    "assert_gold_subset_of_cleaned",
    "bands_present_counts",
    "decode_quality_flag",
    "filter_by_quality_flags",
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
    "quality_flag_bit_counts",
    "quality_flag_legend",
    "quality_flag_mask_from_names",
    "quality_flag_names",
    "rematch_meta_source",
    "summarize_metacatalog",
    "summarize_nedlvs_match",
    "summarize_vlssr_match",
    "write_healpix_hips",
    "VlssrMatchConfig",
    "VlssrMatchResult",
    "NedlvsMatchConfig",
    "NedlvsMatchResult",
    "build_sfr_radio_luminosity_table",
    "load_nedlvs_catalog",
    "load_vlssr_catalog",
    "match_catalog_to_nedlvs",
    "match_catalog_to_vlssr",
    "radio_luminosity_nu",
    "resolve_highest_frequency_peak_flux",
    "select_metacatalog",
    "select_blue_associated_rows",
]
