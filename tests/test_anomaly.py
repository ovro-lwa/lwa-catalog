"""Tests for classical Mahalanobis outlier scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.anomaly import (
    DEFAULT_MAHALANOBIS_COLUMNS,
    continuous_feature_columns,
    mahalanobis_outlier_scores,
    numeric_feature_matrix,
    resolve_mahalanobis_columns,
)


def test_resolve_mahalanobis_columns_intersection() -> None:
    df = pd.DataFrame(
        {
            "Maj": [1.0],
            "Peak_flux_82MHz": [2.0],
            "other": [3.0],
        }
    )
    got = resolve_mahalanobis_columns(df)
    assert got == ["Peak_flux_82MHz", "Maj"]
    assert "other" not in got
    assert all(c in DEFAULT_MAHALANOBIS_COLUMNS for c in got)


def test_continuous_feature_columns_excludes_origin_band() -> None:
    cols = ["origin_band", "Maj", "Min"]
    assert continuous_feature_columns(cols) == ["Maj", "Min"]


def test_empty_df_raises() -> None:
    df = pd.DataFrame({"Maj": pd.Series(dtype=float), "Min": pd.Series(dtype=float)})
    with pytest.raises(ValueError, match="empty"):
        mahalanobis_outlier_scores(df, ["Maj", "Min"])


def test_too_few_complete_vs_features_raises() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan],
            "b": [1.0, 2.0, 3.0],
            "c": [1.0, 2.0, 3.0],
        }
    )
    # Only 2 complete rows, 3 features → n_complete <= D
    with pytest.raises(ValueError, match="n_complete"):
        mahalanobis_outlier_scores(df, ["a", "b", "c"])


def test_percentile_out_of_range_raises() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20)})
    with pytest.raises(ValueError, match="percentile"):
        mahalanobis_outlier_scores(df, ["a", "b"], percentile=0.0)


def test_gaussian_cloud_extreme_flagged() -> None:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, size=n),
            "b": rng.normal(0.0, 1.0, size=n),
        }
    )
    # Far outlier
    df.loc[0, "a"] = 20.0
    df.loc[0, "b"] = 20.0
    result = mahalanobis_outlier_scores(df, ["a", "b"], percentile=99.0)
    assert result.n_scored == n
    assert result.outlier.loc[0]
    assert result.distances.loc[0] == result.distances.max()
    assert result.n_outliers >= 1
    # Threshold matches numpy percentile of finite d2
    finite = result.distances.dropna().to_numpy()
    assert result.threshold == pytest.approx(float(np.percentile(finite, 99.0)))


def test_nan_rows_unscored_in_drop_mode() -> None:
    df = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = mahalanobis_outlier_scores(df, ["a", "b"], percentile=100.0, incomplete="drop")
    assert result.n_scored == 5
    assert np.isnan(result.distances.iloc[-1])
    assert not bool(result.outlier.iloc[-1])
    assert not bool(result.complete_mask.iloc[-1])
    assert result.n_imputed == 0
    assert not result.imputed_mask.any()


def test_origin_band_factorize_and_missing() -> None:
    df = pd.DataFrame(
        {
            "origin_band": ["Blue", "Green", "Blue", "Red", "Green", None, "Blue"],
            "Maj": [1.0, 1.1, 0.9, 1.2, 1.0, 1.0, 0.95],
            "Min": [0.5, 0.55, 0.45, 0.6, 0.5, 0.5, 0.48],
        }
    )
    X, used, complete, warnings = numeric_feature_matrix(
        df, ["origin_band", "Maj", "Min"]
    )
    assert used == ("origin_band", "Maj", "Min")
    assert int(complete.sum()) == 6
    assert not complete.iloc[5]
    # Sorted labels: Blue=0, Green=1, Red=2
    assert X[0, 0] == 0.0
    assert X[1, 0] == 1.0
    assert X[3, 0] == 2.0

    result = mahalanobis_outlier_scores(
        df, ["origin_band", "Maj", "Min"], percentile=99.0
    )
    assert result.n_scored == 6
    assert np.isnan(result.distances.iloc[5])


def test_impute_median_scores_incomplete_rows() -> None:
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    df.loc[0, "a"] = np.nan  # incomplete, has b
    df.loc[1, ["a", "b"]] = np.nan  # all missing → unscored

    drop = mahalanobis_outlier_scores(df, ["a", "b"], percentile=95.0, incomplete="drop")
    imp = mahalanobis_outlier_scores(
        df, ["a", "b"], percentile=95.0, incomplete="impute_median"
    )

    assert drop.n_scored == n - 2
    assert imp.n_scored == n - 1  # row 0 scored, row 1 not
    assert np.isfinite(imp.distances.loc[0])
    assert np.isnan(imp.distances.loc[1])
    assert bool(imp.imputed_mask.loc[0])
    assert not bool(imp.imputed_mask.loc[1])
    # Threshold from complete cases only — same as drop mode threshold
    assert imp.threshold == pytest.approx(drop.threshold)
    assert not drop.imputed_mask.any()


def test_scored_frame_sorted_and_identity_cols() -> None:
    rng = np.random.default_rng(2)
    n = 30
    df = pd.DataFrame(
        {
            "meta_id": np.arange(n),
            "RA": rng.uniform(0, 360, size=n),
            "DEC": rng.uniform(0, 90, size=n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    df.loc[5, "a"] = 50.0
    result = mahalanobis_outlier_scores(df, ["a", "b"], percentile=90.0)
    assert list(result.scored_frame.columns[:3]) == ["meta_id", "RA", "DEC"]
    d2 = result.scored_frame["mahalanobis_d2"].to_numpy()
    assert np.all(d2[:-1] >= d2[1:])
    outliers = result.scored_frame.loc[result.scored_frame["mahalanobis_outlier"]]
    assert len(outliers) == result.n_outliers
