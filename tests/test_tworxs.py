"""Tests for Boller et al. 2016 2RXS cross-match."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    TWORXS_REFERENCE_RADIUS_LOCALIZATION,
)
from lwa_catalog.analyze.tworxs import (
    TworxsMatchConfig,
    filter_tworxs_by_eximl,
    load_tworxs_catalog,
    match_catalog_to_tworxs,
    select_unique_tworxs_matches,
    summarize_tworxs_match,
)
from lwa_catalog.constants import TWORXS_IMAGE_PIXEL_ARCSEC


def _tworxs_line(
    *,
    name: str,
    eximl: float,
    ra: float,
    dec: float,
    crate: float = 0.02,
    e_xima: float = 0.2,
    e_yima: float = 0.2,
) -> str:
    """Build a minimal fixed-width row covering the columns we parse."""
    buf = [" "] * 780
    fields = {
        (1, 21): f"{name:<21}",
        (23, 28): f"{930101:6d}",
        (30, 32): f"{1:3d}",
        (34, 42): f"{eximl:9.2f}",
        (44, 52): f"{10.0:9.2f}",
        (54, 63): f"{3.0:10.6f}",
        (65, 72): f"{crate:8.4f}",
        (74, 80): f"{0.005:7.4f}",
        (82, 89): f"{900.0:8.2f}",
        (91, 99): f"{ra:9.5f}",
        (101, 109): f"{dec:9.5f}",
        (177, 185): f"{0.0:9.3f}",
        (187, 193): f"{0.0:7.3f}",
        (195, 201): f"{0.0:7.2f}",
        (203, 208): f"{0.1:6.3f}",
        (210, 217): f"{0.1:8.3f}",
        (219, 224): f"{0.0:6.3f}",
        (226, 234): f"{0.1:9.3f}",
        (242, 244): f"{0:3d}",
        (463, 472): f"{1e-12:10.5e}",
        (756, 763): f"{e_xima:8.6f}",
        (765, 772): f"{e_yima:8.6f}",
    }
    for (start, end), text in fields.items():
        width = end - start + 1
        chunk = text[:width].rjust(width)
        for i, ch in enumerate(chunk):
            buf[start - 1 + i] = ch
    return "".join(buf)


def _write_mini_tworxs(path, *, gzipped: bool = True) -> None:
    lines = [
        _tworxs_line(name="2RXS J120000.0+400000", eximl=20.0, ra=180.0, dec=40.0),
        _tworxs_line(name="2RXS J120010.0+400000", eximl=7.0, ra=180.1, dec=40.0),
        _tworxs_line(name="2RXS J003000.0-200000", eximl=15.0, ra=10.0, dec=-20.0),
    ]
    text = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def _meta_row(*, meta_id: int, ra: float, dec: float, bmaj: float = 0.2) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "73MHz",
        "Peak_flux": 1.0,
        "origin_band": "73MHz",
    }


def test_load_tworxs_catalog_gz(tmp_path) -> None:
    path = tmp_path / "cat2rxs.dat.gz"
    _write_mini_tworxs(path, gzipped=True)
    df = load_tworxs_catalog(path)
    assert len(df) == 3
    assert "E_RA" in df.columns and "POS_ERR" in df.columns
    expected = TWORXS_IMAGE_PIXEL_ARCSEC * np.hypot(0.2, 0.2)
    assert float(df["POS_ERR"].iloc[0]) == pytest.approx(expected)
    assert df["2RXS"].iloc[0].startswith("2RXS J120000")


def test_load_tworxs_resolves_dat_to_gz(tmp_path) -> None:
    gz = tmp_path / "cat2rxs.dat.gz"
    _write_mini_tworxs(gz, gzipped=True)
    df = load_tworxs_catalog(tmp_path / "cat2rxs.dat")
    assert len(df) == 3


def test_load_tworxs_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="2RXS catalog not found"):
        load_tworxs_catalog(tmp_path / "missing.dat")


def test_filter_tworxs_by_eximl(tmp_path) -> None:
    path = tmp_path / "cat2rxs.dat.gz"
    _write_mini_tworxs(path)
    df = load_tworxs_catalog(path)
    kept = filter_tworxs_by_eximl(df, eximl_min=9.0)
    assert len(kept) == 2
    assert float(kept["ExiML"].min()) >= 9.0


def test_match_catalog_to_tworxs(tmp_path) -> None:
    path = tmp_path / "cat2rxs.dat.gz"
    _write_mini_tworxs(path)
    ref = load_tworxs_catalog(path)
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=180.0, dec=40.0),
            _meta_row(meta_id=2, ra=0.0, dec=0.0),
        ]
    )
    result = match_catalog_to_tworxs(
        meta,
        tworxs=ref,
        config=TworxsMatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=TWORXS_REFERENCE_RADIUS_LOCALIZATION,
            eximl_min=9.0,
        ),
    )
    assert result.summary["n_2rxs_footprint"] == 2
    assert int(result.meta_flags.loc[0, "n_2rxs"]) >= 1
    assert int(result.meta_flags.loc[1, "n_2rxs"]) == 0
    unique = select_unique_tworxs_matches(result.meta_flags)
    assert len(unique) >= 1
    assert "2RXS" in summarize_tworxs_match(result)
