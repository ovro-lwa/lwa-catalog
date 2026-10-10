"""Tests for Freund et al. 2022 ROSAT stellar cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import (
    FREUND2022_REFERENCE_RADIUS_LOCALIZATION,
    LWA_CROSSMATCH_RADIUS_BEAM,
)
from lwa_catalog.analyze.freund2022 import (
    Freund2022MatchConfig,
    filter_freund2022_science_sample,
    load_freund2022_catalog,
    match_catalog_to_freund2022,
    select_unique_freund2022_matches,
    summarize_freund2022_match,
)


def _freund_line(
    rxs: str,
    match: str,
    *,
    epos: float,
    sep: float,
    pstellar: float,
    pij: float,
    ra: float,
    dec: float,
    fx: float,
    hr: float,
    gmag: float,
    bp_rp: float,
    plx: float,
    subdwarf: str,
) -> str:
    """Build one CDS main.dat-style fixed-width row (138 chars)."""
    return (
        f"{rxs:<21} {match:>22} {epos:6.2f} {sep:6.2f} {pstellar:6.4f} {pij:6.4f} "
        f"{ra:10.6f} {dec:11.6f} {fx:8.2e} {hr:5.2f} {gmag:6.2f} {bp_rp:6.2f} "
        f"{plx:7.2f} {subdwarf:<5}"
    )


def _write_mini_freund2022(path) -> None:
    lines = [
        _freund_line(
            "2RXS J120000.0+400000",
            "1234567890123456789",
            epos=10.0,
            sep=5.0,
            pstellar=0.9,
            pij=0.9,
            ra=180.0,
            dec=40.0,
            fx=1e-12,
            hr=0.1,
            gmag=10.0,
            bp_rp=1.0,
            plx=20.0,
            subdwarf="False",
        ),
        _freund_line(
            "2RXS J120010.0+400000",
            "1234567890123456790",
            epos=10.0,
            sep=5.0,
            pstellar=0.9,
            pij=0.9,
            ra=180.1,
            dec=40.0,
            fx=2e-12,
            hr=0.2,
            gmag=12.0,
            bp_rp=1.5,
            plx=5.0,
            subdwarf="True",
        ),
        _freund_line(
            "2RXS J120020.0+400000",
            "1234567890123456791",
            epos=10.0,
            sep=5.0,
            pstellar=0.3,
            pij=0.3,
            ra=180.2,
            dec=40.0,
            fx=3e-12,
            hr=-0.1,
            gmag=14.0,
            bp_rp=2.0,
            plx=2.0,
            subdwarf="False",
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


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


def test_load_freund2022_catalog_fwf(tmp_path) -> None:
    path = tmp_path / "freund2022.dat"
    _write_mini_freund2022(path)
    df = load_freund2022_catalog(path)
    assert len(df) == 3
    assert "E_RA" in df.columns and "E_DEC" in df.columns
    assert float(df["E_RA"].iloc[0]) == pytest.approx((10.0 / 3600.0) / np.sqrt(2.0))
    assert df["2RXS"].iloc[0].startswith("2RXS J120000")
    assert bool(df["subdwarf"].iloc[1]) is True
    assert float(df["POS_ERR"].iloc[0]) == pytest.approx(10.0)


def test_load_freund2022_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Freund\\+2022 catalog not found"):
        load_freund2022_catalog(tmp_path / "missing.dat")


def test_filter_freund2022_science_sample(tmp_path) -> None:
    path = tmp_path / "freund2022.dat"
    _write_mini_freund2022(path)
    df = load_freund2022_catalog(path)
    kept = filter_freund2022_science_sample(df)
    assert len(kept) == 1
    assert float(kept["RA"].iloc[0]) == pytest.approx(180.0)


def test_match_catalog_to_freund2022(tmp_path) -> None:
    path = tmp_path / "freund2022.dat"
    _write_mini_freund2022(path)
    ref = load_freund2022_catalog(path)
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=180.0, dec=40.0),
            _meta_row(meta_id=2, ra=10.0, dec=0.0),
        ]
    )
    result = match_catalog_to_freund2022(
        meta,
        freund2022=ref,
        config=Freund2022MatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=FREUND2022_REFERENCE_RADIUS_LOCALIZATION,
        ),
    )
    assert result.summary["n_freund2022_footprint"] == 1
    assert int(result.meta_flags.loc[0, "n_freund2022"]) == 1
    assert int(result.meta_flags.loc[1, "n_freund2022"]) == 0
    unique = select_unique_freund2022_matches(result.meta_flags)
    assert len(unique) == 1
    text = summarize_freund2022_match(result)
    assert "Meta matched" in text
