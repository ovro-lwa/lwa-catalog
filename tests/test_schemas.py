"""Tests for Arrow catalog schemas."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

from lwa_catalog.constants import GAUL_COLUMNS, METACATALOG_REQUIRED_COLUMNS
from lwa_catalog.schemas import (
    lst_merged_schema,
    metacatalog_schema,
    sources_schema,
    table_from_dataframe,
)


def test_sources_schema_includes_gaul_and_provenance() -> None:
    names = set(sources_schema().names)
    assert set(GAUL_COLUMNS) <= names
    assert {"lst_hour", "band", "source_file", "BMAJ", "BMIN", "BPA", "time_key"} <= names
    from lwa_catalog.constants import DROPPED_GAUL_COLUMNS

    assert names.isdisjoint(DROPPED_GAUL_COLUMNS)


def test_lst_merged_schema_extends_sources() -> None:
    src = set(sources_schema().names)
    lst = set(lst_merged_schema().names)
    assert src <= lst
    assert {"n_lst_contributions", "lst_hours", "representative_lst", "Peak_flux_std"} <= lst


def test_metacatalog_schema_has_required_and_assoc_fields() -> None:
    names = set(metacatalog_schema().names)
    assert METACATALOG_REQUIRED_COLUMNS <= names
    assert "Peak_flux_Blue" in names
    assert "E_Total_flux_Blue" in names
    assert "n_assoc_Green" in names
    assert "source_file_Red" in names
    assert {"alpha_RG", "E_alpha_RG", "alpha_GB", "E_alpha_GB"} <= names
    assert "Peak_flux_std" in names


def test_table_from_dataframe_fills_missing_and_keeps_extras() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [1.0],
            "extra_flag": ["yes"],
        }
    )
    table = table_from_dataframe(df, sources_schema(), include_extras=True)
    assert table.schema.field("RA").type == pa.float64()
    assert "BMAJ" in table.column_names
    assert table.column("BMAJ").null_count == 1
    assert "extra_flag" in table.column_names
