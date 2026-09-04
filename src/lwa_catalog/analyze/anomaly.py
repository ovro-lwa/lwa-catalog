"""Classical Mahalanobis outlier scoring on metacatalog feature columns.

Fits empirical mean/covariance on **complete cases** of selected columns and
flags high squared Mahalanobis distance via a percentile threshold. Non-numeric
columns such as ``origin_band`` are encoded with sorted ``pandas.factorize`` codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

IncompleteMode = Literal["drop", "impute_median"]

DEFAULT_MAHALANOBIS_COLUMNS: tuple[str, ...] = (
    "origin_band",
    "n_lst_contributions",
    "Peak_flux_82MHz",
    "Total_flux_82MHz",
    "Maj",
    "Min",
    "spec_model_a0",
    "spec_model_a1",
    "spec_model_a2",
    "spec_model_a3",
    "spec_model_n_flux",
    "spec_model_n_terms",
)

# Columns encoded via factorize rather than numeric coerce.
CATEGORICAL_FEATURE_COLUMNS: frozenset[str] = frozenset({"origin_band"})

_IDENTITY_COLUMNS: tuple[str, ...] = ("meta_id", "RA", "DEC")


@dataclass(frozen=True)
class MahalanobisConfig:
    """Configuration for :func:`mahalanobis_outlier_scores`."""

    percentile: float = 99.0
    incomplete: IncompleteMode = "drop"


@dataclass
class MahalanobisResult:
    """Classical Mahalanobis outlier scores for a table."""

    columns: tuple[str, ...]
    percentile: float
    threshold: float
    n_input: int
    n_scored: int
    n_outliers: int
    distances: pd.Series
    outlier: pd.Series
    complete_mask: pd.Series
    imputed_mask: pd.Series
    mean: np.ndarray
    covariance: np.ndarray
    scored_frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    n_imputed: int = 0
    incomplete: IncompleteMode = "drop"


def is_categorical_feature(column: str) -> bool:
    """Return True when *column* is factorized rather than numeric-coerced."""
    return column in CATEGORICAL_FEATURE_COLUMNS


def resolve_mahalanobis_columns(
    df: pd.DataFrame,
    preferred: Sequence[str] = DEFAULT_MAHALANOBIS_COLUMNS,
) -> list[str]:
    """Return *preferred* columns that exist on *df*, preserving order."""
    present = set(df.columns)
    return [c for c in preferred if c in present]


def continuous_feature_columns(columns: Sequence[str]) -> list[str]:
    """Selected feature names suitable for scatter axes (exclude categoricals)."""
    return [c for c in columns if not is_categorical_feature(c)]


def _encode_feature_column(series: pd.Series, *, name: str) -> pd.Series:
    """Return a float Series; NaN where the feature is unusable."""
    if is_categorical_feature(name) or not pd.api.types.is_numeric_dtype(series):
        cleaned = series.astype("string")
        cleaned = cleaned.mask(cleaned.isna() | (cleaned.str.strip() == ""), pd.NA)
        non_null = cleaned.dropna()
        if non_null.empty:
            return pd.Series(np.nan, index=series.index, dtype=float)
        uniques = sorted(non_null.unique().tolist())
        mapping = {label: float(i) for i, label in enumerate(uniques)}
        return cleaned.map(mapping).astype(float)

    return pd.to_numeric(series, errors="coerce").astype(float)


def numeric_feature_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], pd.Series, list[str]]:
    """Build an ``(N, D)`` float matrix and a complete-case mask.

    Columns missing from *df* are skipped with a warning. Columns that are
    entirely non-finite after encoding are dropped with a warning.

    Returns
    -------
    X
        Shape ``(len(df), D)`` with NaNs for incomplete cells.
    used_columns
        Columns retained after dropping missing/all-NaN fields.
    complete_mask
        True where every used column is finite for that row.
    warnings
        Human-readable notes about dropped columns, etc.
    """
    warnings: list[str] = []
    if not columns:
        raise ValueError("columns must be a non-empty sequence")

    frames: list[pd.Series] = []
    used: list[str] = []
    for col in columns:
        if col not in df.columns:
            warnings.append(f"column {col!r} not in DataFrame; skipped")
            continue
        encoded = _encode_feature_column(df[col], name=col)
        if not np.isfinite(encoded.to_numpy(dtype=float)).any():
            warnings.append(f"column {col!r} has no finite values; skipped")
            continue
        frames.append(encoded.rename(col))
        used.append(col)

    if not used:
        raise ValueError("no usable feature columns after encoding")

    mat = pd.concat(frames, axis=1)
    X = mat.to_numpy(dtype=float)
    complete = pd.Series(np.isfinite(X).all(axis=1), index=df.index, name="complete")
    return X, tuple(used), complete, warnings


def _squared_mahalanobis(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Return squared Mahalanobis distances for rows of *X*."""
    delta = X - mean
    # Pseudo-inverse handles near-singular covariances (correlated fluxes).
    precision = np.linalg.pinv(cov, hermitian=True)
    # (delta @ precision * delta).sum(axis=1)
    return np.einsum("ij,jk,ik->i", delta, precision, delta)


def _validate_percentile(percentile: float) -> float:
    p = float(percentile)
    if not np.isfinite(p) or not (0.0 < p <= 100.0):
        raise ValueError(f"percentile must be in (0, 100], got {percentile!r}")
    return p


def _fit_mean_cov(X_complete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = X_complete.shape
    if n <= d:
        raise ValueError(
            f"need n_complete > n_features for covariance "
            f"(got n_complete={n}, n_features={d})"
        )
    if n < 2:
        raise ValueError(f"need at least 2 complete rows to score (got {n})")
    mean = np.mean(X_complete, axis=0)
    # np.cov uses N-1; classical sample covariance.
    cov = np.cov(X_complete, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    return mean, np.asarray(cov, dtype=float)


def _build_scored_frame(
    df: pd.DataFrame,
    *,
    scored_index: pd.Index,
    columns: Sequence[str],
    distances: pd.Series,
    outlier: pd.Series,
    imputed_mask: pd.Series,
) -> pd.DataFrame:
    if len(scored_index) == 0:
        return pd.DataFrame()
    parts: list[pd.Series | pd.DataFrame] = []
    for col in _IDENTITY_COLUMNS:
        if col in df.columns:
            parts.append(df.loc[scored_index, col])
    for col in columns:
        if col in df.columns:
            parts.append(df.loc[scored_index, col])
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=scored_index)
    out = out.copy()
    out["mahalanobis_d2"] = distances.loc[scored_index].to_numpy(dtype=float)
    out["mahalanobis_outlier"] = outlier.loc[scored_index].to_numpy(dtype=bool)
    out["mahalanobis_imputed"] = imputed_mask.loc[scored_index].to_numpy(dtype=bool)
    return out.sort_values("mahalanobis_d2", ascending=False)


def mahalanobis_outlier_scores(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    percentile: float = 99.0,
    incomplete: IncompleteMode = "drop",
    config: MahalanobisConfig | None = None,
) -> MahalanobisResult:
    """Score rows with classical squared Mahalanobis distance.

    Parameters
    ----------
    df
        Input table (typically ``browser._df`` after quality / RADIO_QA filters).
    columns
        Feature columns. Defaults to
        :func:`resolve_mahalanobis_columns` with
        :data:`DEFAULT_MAHALANOBIS_COLUMNS`.
    percentile
        High-tail percentile of complete-case ``d^2`` used as the outlier
        threshold (``outlier`` when ``d^2 >= threshold``).
    incomplete
        ``"drop"`` — score complete cases only (default).
        ``"impute_median"`` — fit on complete cases; median-impute holes for
        rows with at least one observed feature, then score (Phase 5).
    config
        Optional config; overrides *percentile* / *incomplete* when given.

    Returns
    -------
    MahalanobisResult
        Distances aligned to ``df.index`` (NaN where unscored), masks, and a
        ``scored_frame`` of scored rows sorted by descending ``d^2``.
    """
    if config is not None:
        percentile = config.percentile
        incomplete = config.incomplete

    if incomplete not in ("drop", "impute_median"):
        raise ValueError(f"incomplete must be 'drop' or 'impute_median', got {incomplete!r}")

    percentile = _validate_percentile(percentile)

    if df is None or len(df) == 0:
        raise ValueError("df is empty; cannot compute Mahalanobis scores")

    if columns is None:
        cols: list[str] = resolve_mahalanobis_columns(df)
    else:
        cols = list(columns)
    if not cols:
        raise ValueError("no feature columns selected")

    X, used_columns, complete_mask, warnings = numeric_feature_matrix(df, cols)
    n_input = len(df)
    complete_idx = df.index[complete_mask.to_numpy()]
    n_complete = int(complete_mask.sum())
    d = len(used_columns)

    if n_complete < 2:
        raise ValueError(f"need at least 2 complete rows to score (got {n_complete})")
    if n_complete <= d:
        raise ValueError(
            f"need n_complete > n_features for covariance "
            f"(got n_complete={n_complete}, n_features={d})"
        )

    X_complete = X[complete_mask.to_numpy()]
    mean, cov = _fit_mean_cov(X_complete)
    try:
        cond = float(np.linalg.cond(cov))
        if np.isfinite(cond) and cond > 1e12:
            warnings.append(f"covariance condition number is large ({cond:.2e})")
    except np.linalg.LinAlgError:
        warnings.append("could not estimate covariance condition number")

    d2_complete = _squared_mahalanobis(X_complete, mean, cov)
    if not np.isfinite(d2_complete).all():
        raise ValueError("non-finite Mahalanobis distances for complete cases")

    threshold = float(np.percentile(d2_complete, percentile))

    distances = pd.Series(np.nan, index=df.index, dtype=float, name="mahalanobis_d2")
    outlier = pd.Series(False, index=df.index, dtype=bool, name="mahalanobis_outlier")
    imputed_mask = pd.Series(False, index=df.index, dtype=bool, name="mahalanobis_imputed")

    distances.loc[complete_idx] = d2_complete
    outlier.loc[complete_idx] = d2_complete >= threshold

    n_imputed = 0
    if incomplete == "impute_median":
        medians = np.nanmedian(X_complete, axis=0)
        incomplete_rows = ~complete_mask.to_numpy()
        observed = np.isfinite(X)
        # Rows with at least one observed feature among used columns.
        any_obs = observed.any(axis=1) & incomplete_rows
        all_missing = incomplete_rows & ~observed.any(axis=1)
        if int(all_missing.sum()):
            warnings.append(
                f"{int(all_missing.sum())} row(s) missing all selected features; unscored"
            )

        X_imp = X.copy()
        # Fill NaN holes on incomplete rows that have ≥1 observation.
        for j in range(d):
            hole = any_obs & ~observed[:, j]
            X_imp[hole, j] = medians[j]

        filled_any = any_obs & (~observed).any(axis=1)
        imputed_mask.loc[df.index[filled_any]] = True

        if int(any_obs.sum()):
            d2_imp = _squared_mahalanobis(X_imp[any_obs], mean, cov)
            imp_index = df.index[any_obs]
            distances.loc[imp_index] = d2_imp
            outlier.loc[imp_index] = d2_imp >= threshold
            n_imputed = int(imputed_mask.sum())
            if n_input > 0 and (n_imputed / n_input) > 0.5:
                warnings.append(
                    f"imputed {n_imputed} / {n_input} rows "
                    f"({100.0 * n_imputed / n_input:.1f}%); interpret with care"
                )
    elif incomplete != "drop":
        raise ValueError(f"unsupported incomplete mode: {incomplete!r}")

    scored_mask = distances.notna()
    scored_index = df.index[scored_mask.to_numpy()]
    n_scored = int(scored_mask.sum())
    n_outliers = int(outlier.to_numpy().sum())

    scored_frame = _build_scored_frame(
        df,
        scored_index=scored_index,
        columns=used_columns,
        distances=distances,
        outlier=outlier,
        imputed_mask=imputed_mask,
    )

    return MahalanobisResult(
        columns=used_columns,
        percentile=percentile,
        threshold=threshold,
        n_input=n_input,
        n_scored=n_scored,
        n_outliers=n_outliers,
        distances=distances,
        outlier=outlier,
        complete_mask=complete_mask,
        imputed_mask=imputed_mask,
        mean=mean,
        covariance=cov,
        scored_frame=scored_frame,
        warnings=warnings,
        n_imputed=n_imputed,
        incomplete=incomplete,
    )


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Column names suitable for the notebook MultiChoice (numeric + categoricals)."""
    names: list[str] = []
    for col in df.columns:
        if is_categorical_feature(col):
            names.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            names.append(col)
    # Prefer default order first, then remaining alphabetical
    preferred = resolve_mahalanobis_columns(df)
    preferred_set = set(preferred)
    rest = sorted(c for c in names if c not in preferred_set)
    return preferred + rest
