"""Tests for long-form metacatalog associations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.associations import (
    ASSOCIATION_SPEC_FREUND2022,
    ASSOCIATION_SPEC_MILLIQUAS,
    ASSOCIATION_SPEC_XU2022,
    CatalogAssociationSpec,
    association_rows_from_match,
    association_rows_from_survey_attach,
    empty_associations_frame,
    flag_association_conflicts,
    harmonize_associations,
    harmonize_class_native,
    meta_ids_for_classes,
    metacatalog_with_erass3_association_columns,
    ra_dec_key,
    read_associations,
    replace_catalog_associations,
    summarize_classifications,
    validate_associations,
    write_associations,
)
from lwa_catalog.paths import CatalogLayout


def test_layout_associations_path(tmp_path) -> None:
    layout = CatalogLayout(tmp_path)
    assert layout.metacatalog_associations().name == "metacatalog_associations.parquet"


def test_empty_schema_and_validate() -> None:
    df = empty_associations_frame()
    validate_associations(df)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_associations(pd.DataFrame({"meta_id": [1]}))


def test_association_rows_from_match_unique_only() -> None:
    footprint = pd.DataFrame(
        {
            "Name": ["a", "b", "c"],
            "RA": [10.0, 11.0, 12.0],
            "DEC": [20.0, 21.0, 22.0],
            "class_primary": ["Q", "A", "S"],
            "class_label": ["QSO", "AGN", "star"],
        }
    )
    meta_flags = pd.DataFrame(
        {
            "meta_id": [1, 2, 3],
            "RA": [10.001, 11.0, 12.0],
            "DEC": [20.001, 21.0, 22.0],
            "n_milliquas": [1, 2, 0],
            "milliquas_positions": [[0], [1, 2], []],
            "matched": [True, True, False],
        }
    )
    rows = association_rows_from_match(meta_flags, footprint, ASSOCIATION_SPEC_MILLIQUAS)
    assert len(rows) == 1
    assert int(rows.iloc[0]["meta_id"]) == 1
    assert rows.iloc[0]["ref_id"] == "a"
    assert rows.iloc[0]["class_native"] == "Q"
    assert rows.iloc[0]["n_ref"] == 1
    assert np.isfinite(float(rows.iloc[0]["sep_arcsec"]))
    assert rows.iloc[0]["class_native_detail"] == "QSO"


def test_class_fallback_to_catalog_name() -> None:
    footprint = pd.DataFrame(
        {"2RXS": ["2RXS J1"], "RA": [180.0], "DEC": [40.0]}
    )
    meta_flags = pd.DataFrame(
        {
            "meta_id": [7],
            "RA": [180.0],
            "DEC": [40.0],
            "n_freund2022": [1],
            "freund2022_positions": [[0]],
        }
    )
    rows = association_rows_from_match(
        meta_flags, footprint, ASSOCIATION_SPEC_FREUND2022
    )
    assert list(rows["class_native"]) == ["Freund2022"]
    assert list(rows["ref_id"]) == ["2RXS J1"]


def test_ref_id_ra_dec_key_fallback() -> None:
    spec = CatalogAssociationSpec(
        catalog="VLSSR",
        id_column="missing_id",
        positions_col="vlssr_positions",
        n_col="n_vlssr",
        class_column=None,
    )
    footprint = pd.DataFrame({"RA": [1.5], "DEC": [-2.25]})
    meta_flags = pd.DataFrame(
        {
            "meta_id": [3],
            "RA": [1.5],
            "DEC": [-2.25],
            "n_vlssr": [1],
            "vlssr_positions": [[0]],
        }
    )
    rows = association_rows_from_match(meta_flags, footprint, spec)
    assert rows.iloc[0]["ref_id"] == ra_dec_key(1.5, -2.25)
    assert rows.iloc[0]["class_native"] == "VLSSR"


def test_association_rows_from_survey_attach() -> None:
    meta = pd.DataFrame(
        {
            "meta_id": [1, 2, 3],
            "RA": [10.0, 11.0, 12.0],
            "DEC": [20.0, 21.0, 22.0],
            "n_assoc_NVSS": [1, 2, 0],
            "RA_NVSS": [10.01, 11.01, np.nan],
            "DEC_NVSS": [20.01, 21.01, np.nan],
            "Peak_flux_NVSS": [1.0, 2.0, np.nan],
        }
    )
    rows = association_rows_from_survey_attach(meta, "NVSS")
    assert len(rows) == 1
    assert int(rows.iloc[0]["meta_id"]) == 1
    assert rows.iloc[0]["catalog"] == "NVSS"
    assert rows.iloc[0]["class_native"] == "NVSS"
    assert rows.iloc[0]["ref_id"] == ra_dec_key(10.01, 20.01)


def test_replace_catalog_associations_isolated(tmp_path) -> None:
    layout = CatalogLayout(tmp_path)
    assert read_associations(layout).empty

    a = pd.DataFrame(
        {
            "meta_id": [1, 2],
            "catalog": ["MilliQuas", "MilliQuas"],
            "ref_id": ["a", "b"],
            "ref_ra": [1.0, 2.0],
            "ref_dec": [3.0, 4.0],
            "class_native": ["Q", "A"],
            "sep_arcsec": [1.0, 2.0],
            "n_ref": [1, 1],
        }
    )
    b = pd.DataFrame(
        {
            "meta_id": [1],
            "catalog": ["Freund2022"],
            "ref_id": ["rxs"],
            "ref_ra": [1.0],
            "ref_dec": [3.0],
            "class_native": ["Freund2022"],
            "sep_arcsec": [0.5],
            "n_ref": [1],
        }
    )
    replace_catalog_associations(layout, "MilliQuas", a)
    replace_catalog_associations(layout, "Freund2022", b)
    all_rows = read_associations(layout)
    assert set(all_rows["catalog"]) == {"MilliQuas", "Freund2022"}
    assert len(all_rows) == 3

    a2 = a.iloc[[0]].copy()
    a2["class_native"] = ["Q"]
    replace_catalog_associations(layout, "MilliQuas", a2)
    updated = read_associations(layout)
    mq = updated.loc[updated["catalog"] == "MilliQuas"]
    fr = updated.loc[updated["catalog"] == "Freund2022"]
    assert len(mq) == 1
    assert int(mq.iloc[0]["meta_id"]) == 1
    assert len(fr) == 1
    assert fr.iloc[0]["ref_id"] == "rxs"

    # Empty replace drops that catalog.
    replace_catalog_associations(layout, "MilliQuas", empty_associations_frame())
    final = read_associations(layout)
    assert list(final["catalog"]) == ["Freund2022"]


def test_write_associations_full_rewrite(tmp_path) -> None:
    path = tmp_path / "metacatalog_associations.parquet"
    rows = pd.DataFrame(
        {
            "meta_id": [9],
            "catalog": ["2RXS"],
            "ref_id": ["x"],
            "ref_ra": [0.0],
            "ref_dec": [0.0],
            "class_native": ["2RXS"],
            "sep_arcsec": [0.0],
            "n_ref": [1],
        }
    )
    write_associations(path, rows)
    assert read_associations(path).iloc[0]["catalog"] == "2RXS"


def test_harmonize_and_conflicts() -> None:
    assert harmonize_class_native("Freund2022", "Freund2022") == "star"
    assert harmonize_class_native("MilliQuas", "Q") == "agn"
    assert harmonize_class_native("MilliQuas", "S") == "star"
    assert harmonize_class_native("Xu2022", "S") == "cluster"
    assert harmonize_class_native("Xu2022", "G") == "cluster"
    assert harmonize_class_native("eRASS3", "-1") == "star"
    assert harmonize_class_native("eRASS3", "3") == "galaxy"
    assert harmonize_class_native("eRASS3", "blazar") == "agn"
    assert harmonize_class_native("VLSSR", "VLSSR") == "radio"
    assert harmonize_class_native("NEDLVS", "G") == "galaxy"

    assoc = pd.DataFrame(
        {
            "meta_id": [1, 1, 2, 2],
            "catalog": ["Freund2022", "MilliQuas", "VLSSR", "NVSS"],
            "ref_id": ["r", "m", "v1", "v2"],
            "ref_ra": [0.0, 0.0, 1.0, 1.0],
            "ref_dec": [0.0, 0.0, 1.0, 1.0],
            "class_native": ["Freund2022", "Q", "VLSSR", "NVSS"],
            "sep_arcsec": [0.1, 0.2, 0.3, 0.4],
            "n_ref": [1, 1, 1, 1],
        }
    )
    harm = harmonize_associations(assoc)
    assert list(harm["class_harmonized"]) == ["star", "agn", "radio", "radio"]
    flags = flag_association_conflicts(assoc)
    by_id = flags.set_index("meta_id")
    assert bool(by_id.loc[1, "conflict"]) is True
    assert bool(by_id.loc[2, "conflict"]) is False
    summary = summarize_classifications(assoc)
    assert summary["n_conflict_meta_ids"] == 1
    assert summary["n_associations"] == 4


def test_meta_ids_for_classes() -> None:
    assoc = pd.DataFrame(
        {
            "meta_id": [1, 1, 2, 2, 3],
            "catalog": ["Freund2022", "MilliQuas", "VLSSR", "NVSS", "MilliQuas"],
            "ref_id": ["r", "m", "v1", "v2", "q"],
            "ref_ra": [0.0, 0.0, 1.0, 1.0, 2.0],
            "ref_dec": [0.0, 0.0, 1.0, 1.0, 2.0],
            "class_native": ["Freund2022", "Q", "VLSSR", "NVSS", "S"],
            "sep_arcsec": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_ref": [1, 1, 1, 1, 1],
        }
    )
    harm = harmonize_associations(assoc)

    # meta_id 1 has star+agn; 2 is radio-only; 3 is star-only.
    assert list(meta_ids_for_classes(harm, "star")) == [1, 3]
    assert list(meta_ids_for_classes(harm, "star", exclusive=True)) == [3]
    assert list(meta_ids_for_classes(harm, "star", exclude_conflicts=True)) == [3]
    assert list(meta_ids_for_classes(harm, ["radio"], exclusive=True)) == [2]
    assert list(
        meta_ids_for_classes(harm, "star", catalogs=("MilliQuas",))
    ) == [3]
    assert list(
        meta_ids_for_classes(harm, "S", column="class_native", catalogs=("MilliQuas",))
    ) == [3]
    assert meta_ids_for_classes(harm, "cluster").size == 0
    assert meta_ids_for_classes(pd.DataFrame(), "star").size == 0


def test_xu_class_does_not_collide_with_milliquas_s() -> None:
    footprint = pd.DataFrame(
        {"RXGCC": ["RXGCC J1"], "RA": [10.0], "DEC": [20.0], "Class": ["S"]}
    )
    meta_flags = pd.DataFrame(
        {
            "meta_id": [1],
            "RA": [10.0],
            "DEC": [20.0],
            "n_xu2022": [1],
            "xu2022_positions": [[0]],
        }
    )
    rows = association_rows_from_match(meta_flags, footprint, ASSOCIATION_SPEC_XU2022)
    assert rows.iloc[0]["class_native"] == "S"
    assert harmonize_class_native("Xu2022", "S") == "cluster"


def test_metacatalog_with_erass3_association_columns() -> None:
    meta = pd.DataFrame(
        {"meta_id": [1, 2], "RA": [10.0, 11.0], "DEC": [20.0, 21.0]}
    )
    assoc = pd.DataFrame(
        {
            "meta_id": [1],
            "catalog": ["eRASS3"],
            "ref_id": ["42"],
            "ref_ra": [10.01],
            "ref_dec": [20.01],
            "class_native": ["3"],
            "sep_arcsec": [1.5],
            "n_ref": [1],
        }
    )
    out = metacatalog_with_erass3_association_columns(meta, assoc)
    assert int(out.loc[0, "n_erass3"]) == 1
    assert int(out.loc[1, "n_erass3"]) == 0
    assert int(out.loc[0, "eRASS3_class_gal_exgal"]) == 3
    assert str(out.loc[0, "eRASS3_LS10_OBJID"]) == "42"


def test_meta_id_required() -> None:
    with pytest.raises(ValueError, match="meta_id"):
        association_rows_from_match(
            pd.DataFrame({"n_milliquas": [1], "milliquas_positions": [[0]]}),
            pd.DataFrame({"Name": ["a"], "RA": [0.0], "DEC": [0.0]}),
            ASSOCIATION_SPEC_MILLIQUAS,
        )
