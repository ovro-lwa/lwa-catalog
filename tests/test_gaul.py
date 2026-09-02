"""Tests for PyBDSF GAUL column normalization."""

from __future__ import annotations

import pandas as pd

from lwa_catalog.gaul import cast_gaul_string_columns, cast_s_code_value


def test_cast_s_code_value_handles_bytes_and_whitespace() -> None:
    # PyBDSF GAUL FITS tables expose S_Code as byte strings.
    assert cast_s_code_value(b"S") == "S"
    assert cast_s_code_value(" c ") == "c"
    assert pd.isna(cast_s_code_value(None))
    assert pd.isna(cast_s_code_value(float("nan")))


def test_cast_gaul_string_columns() -> None:
    df = pd.DataFrame({"S_Code": [b"M", " S ", None], "Peak_flux": [1.0, 2.0, 3.0]})
    out = cast_gaul_string_columns(df)
    assert out["S_Code"].tolist() == ["M", "S", pd.NA]
    assert str(out["S_Code"].dtype) == "string"
