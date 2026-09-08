"""Tests for catalog directory inventory and row-filter helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.catalog_index import (
    DEFAULT_DISPLAY_COLUMNS,
    apply_quality_mask,
    apply_radio_qa_filter,
    classify_catalog,
    discover_catalog_dirs,
    inventory_catalogs,
    is_metacatalog_parquet,
    load_display_column_prefs,
    save_display_column_prefs,
)
from lwa_catalog.analyze.reliability import SourceQualityFlag


def test_is_metacatalog_parquet() -> None:
    assert is_metacatalog_parquet("metacatalog.parquet")
    assert is_metacatalog_parquet("metacatalog_quality.parquet")
    assert is_metacatalog_parquet("metacatalog_spectral.parquet")
    assert is_metacatalog_parquet("metacatalog_radio.parquet")
    assert not is_metacatalog_parquet("metacatalog_lst_Blue.parquet")
    assert not is_metacatalog_parquet("sources_01h_Blue.parquet")


def test_classify_catalog_kinds() -> None:
    assert classify_catalog("metacatalog_lst_18MHz.parquet") == {
        "kind": "lst_merged",
        "lst_hour": None,
        "band": "18MHz",
    }
    assert classify_catalog("sources_01h_Blue.parquet")["kind"] == "sources"
    assert classify_catalog("sources_01h_Blue.parquet")["lst_hour"] == "01h"
    assert classify_catalog("sources_01h_Blue.parquet")["band"] == "Blue"
    assert classify_catalog("metacatalog_spectral.parquet")["kind"] == "metacatalog"
    assert classify_catalog("random.parquet")["kind"] == "other"


def test_discover_and_inventory(tmp_path: Path) -> None:
    tree = tmp_path / "metacatalog_demo"
    tree.mkdir()
    (tree / "metacatalog.parquet").write_bytes(b"PAR1")  # not a real parquet
    (tree / "sources_02h_Red.parquet").write_bytes(b"PAR1")
    empty = tmp_path / "metacatalog_empty"
    empty.mkdir()

    dirs = discover_catalog_dirs(tmp_path)
    assert tree.resolve() in dirs
    assert empty.resolve() not in dirs

    inv = inventory_catalogs(tree)
    assert set(inv["file"]) == {"metacatalog.parquet", "sources_02h_Red.parquet"}
    kinds = dict(zip(inv["file"], inv["kind"], strict=True))
    assert kinds["metacatalog.parquet"] == "metacatalog"
    assert kinds["sources_02h_Red.parquet"] == "sources"
    # Corrupt/non-parquet → n_rows None, size still reported
    assert inv["n_rows"].isna().all()
    assert (inv["size_mb"] >= 0).all()


def test_apply_radio_qa_filter_variants() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=[10, 20, 30])

    assert len(apply_radio_qa_filter(df, None)) == 3
    assert list(apply_radio_qa_filter(df, "a >= 2")["a"]) == [2, 3]

    mask = pd.Series([True, False, True], index=df.index)
    assert list(apply_radio_qa_filter(df, mask)["a"]) == [1, 3]

    assert list(apply_radio_qa_filter(df, df.index[[0, 2]])["a"]) == [1, 3]
    assert list(apply_radio_qa_filter(df, [20])["a"]) == [2]

    with pytest.raises(TypeError, match="boolean"):
        apply_radio_qa_filter(df, pd.Series([1, 2, 3], index=df.index))

    with pytest.raises(ValueError, match="not present"):
        apply_radio_qa_filter(df, pd.Series([True], index=[999]))


def test_apply_quality_mask_none_passthrough() -> None:
    df = pd.DataFrame(
        {
            "quality_flag": [
                0,
                int(SourceQualityFlag.HAS_NAN),
            ]
        }
    )
    out = apply_quality_mask(df, None)
    assert len(out) == 2
    filtered = apply_quality_mask(df, int(SourceQualityFlag.HAS_NAN))
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["quality_flag"]) == 0


def test_display_column_prefs_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cols.json"
    assert load_display_column_prefs(path) == list(DEFAULT_DISPLAY_COLUMNS)
    saved = save_display_column_prefs(["meta_id", "RA", "DEC"], path)
    assert saved == path
    assert load_display_column_prefs(path) == ["meta_id", "RA", "DEC"]
