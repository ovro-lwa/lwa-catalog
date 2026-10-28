"""Bootstrap per-band restoring-beam α from VLSSR + neighbor Total/Peak.

``Total_flux_*`` already embeds the detect-time beam scale used to build the
metacatalog. The VLSSR anchor therefore estimates a **correction** relative to
that scale, and the absolute α is the product:

``α_true = beam_scale_used × √(median Total_anchor / Peak_VLSSR)``

Operators must pass ``beam_scale_used`` (``1.0`` if the tree was detected
unscaled). Omitting it is an error — a wrong prior silently biases every band.

After anchoring at a VLSSR-matched LWA band (default 73 MHz), walk neighbors
with ``(T_hi/T_lo) / (P_hi/P_lo) ≈ (α_lo/α_hi)²`` so spectral slope is not
absorbed into Δα. Output tables feed a re-detect with per-band ``beam_scale``
(see ``resolve_beam_scale``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lwa_catalog.constants import (
    COLOR_BANDS,
    SUBBAND_BANDS_MHZ,
    band_frequency_hz,
)
from lwa_catalog.create.beam_scale_model import (
    BeamScaleModel,
    beam_scale_model_from_value,
)
from lwa_catalog.io import read_table, write_table
from lwa_catalog.paths import CatalogLayout

__all__ = [
    "BEAM_ALPHA_BY_BAND_COLUMNS",
    "BEAM_ALPHA_STEP_COLUMNS",
    "BeamAlphaBootstrapConfig",
    "BeamAlphaBootstrapResult",
    "bootstrap_beam_alpha",
    "lwa_bands_with_total_flux",
    "load_beam_alpha_mapping",
    "load_beam_scale",
    "median_neighbor_flux_ratio",
    "read_beam_alpha_table",
    "read_beam_scale_json",
    "summarize_beam_alpha_bootstrap",
    "unconfused_unique_mask",
    "write_beam_alpha_table",
    "write_beam_scale_json",
]


BEAM_ALPHA_BY_BAND_COLUMNS: tuple[str, ...] = (
    "band",
    "alpha",
    "nu_mhz",
    "alpha_over_alpha_anchor",
)

BEAM_ALPHA_STEP_COLUMNS: tuple[str, ...] = (
    "band_lo",
    "band_hi",
    "direction",
    "n",
    "median_T_hi_over_lo",
    "median_P_hi_over_lo",
    "alpha_hi_over_lo",
    "alpha_lo",
    "alpha_hi",
)


@dataclass(frozen=True, slots=True)
class BeamAlphaBootstrapConfig:
    """Configuration for :func:`bootstrap_beam_alpha`.

    ``beam_scale_used`` is required: the scalar detect-time α that produced
    ``Total_flux_*`` on this tree (``1.0`` if unscaled / library default).
    """

    beam_scale_used: float
    anchor_band: str = "73MHz"
    bands: tuple[str, ...] | None = None
    require_unconfused_unique: bool = True


@dataclass(frozen=True, slots=True)
class BeamAlphaBootstrapResult:
    """Outcome of :func:`bootstrap_beam_alpha`.

    ``alpha_correction`` is ``√(median Total/VLSSR)``; absolute anchor α is
    ``beam_scale_used × alpha_correction``.
    """

    by_band: pd.DataFrame
    steps: pd.DataFrame
    alpha_anchor: float
    alpha_correction: float
    median_total_over_vlssr: float
    n_vlssr_unique: int
    anchor_band: str
    beam_scale_used: float
    warnings: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, float]:
        """Return ``{band: alpha}`` in frequency order."""
        out: dict[str, float] = {}
        for _, row in self.by_band.iterrows():
            a = float(row["alpha"])
            if np.isfinite(a) and a > 0.0:
                out[str(row["band"])] = a
        return out


def lwa_bands_with_total_flux(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] | None = None,
) -> list[str]:
    """Return LWA bands with ``Total_flux_{band}``, sorted by frequency."""
    pool = tuple(candidates) if candidates is not None else (*SUBBAND_BANDS_MHZ, *COLOR_BANDS)
    seen: list[str] = []
    for band in pool:
        if band not in seen and f"Total_flux_{band}" in df.columns:
            seen.append(band)

    def _key(b: str) -> float:
        nu = band_frequency_hz(b)
        return float(nu) if np.isfinite(nu) else float("inf")

    return sorted(seen, key=_key)


def unconfused_unique_mask(df: pd.DataFrame, band: str) -> pd.Series:
    """True where ``n_assoc==1`` and ``n_confused==1`` (missing ⇒ False)."""
    assoc_col = f"n_assoc_{band}"
    conf_col = f"n_confused_{band}"
    if assoc_col not in df.columns or conf_col not in df.columns:
        return pd.Series(False, index=df.index)
    n_assoc = pd.to_numeric(df[assoc_col], errors="coerce")
    n_conf = pd.to_numeric(df[conf_col], errors="coerce")
    return (n_assoc == 1) & (n_conf == 1)


def median_neighbor_flux_ratio(
    df: pd.DataFrame,
    band_lo: str,
    band_hi: str,
    *,
    flux_prefix: str,
    require_unconfused_unique: bool = True,
) -> tuple[float, int]:
    """Median ``S_hi/S_lo`` for positive fluxes (optionally unconfused unique)."""
    lo_col = f"{flux_prefix}_{band_lo}"
    hi_col = f"{flux_prefix}_{band_hi}"
    if lo_col not in df.columns or hi_col not in df.columns:
        return float("nan"), 0
    if require_unconfused_unique:
        mask = unconfused_unique_mask(df, band_lo) & unconfused_unique_mask(df, band_hi)
    else:
        mask = pd.Series(True, index=df.index)
    x = pd.to_numeric(df[lo_col], errors="coerce")
    y = pd.to_numeric(df[hi_col], errors="coerce")
    ok = mask & (x > 0) & (y > 0)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.median((y / x).loc[ok])), n


def _require_beam_scale_used(beam_scale_used: float) -> float:
    """Validate detect-time α; Totals already embed this scale."""
    scale = float(beam_scale_used)
    if not np.isfinite(scale) or scale <= 0.0:
        msg = (
            "beam_scale_used must be finite and > 0 "
            f"(detect-time α that built Total_flux_*); got {beam_scale_used!r}. "
            "Use 1.0 if the metacatalog was detected unscaled. "
            "True α = beam_scale_used × √(med Total/VLSSR Peak)."
        )
        raise ValueError(msg)
    return scale


def _alpha_anchor_from_vlssr(
    meta: pd.DataFrame,
    *,
    anchor_band: str,
    beam_scale_used: float,
) -> tuple[float, float, float, int]:
    """Return ``(alpha_anchor, alpha_correction, median Total/VLSSR Peak, n)``.

    ``alpha_correction = √(med Total/VLSSR)``;
    ``alpha_anchor = beam_scale_used × alpha_correction``.
    """
    scale = _require_beam_scale_used(beam_scale_used)
    total_col = f"Total_flux_{anchor_band}"
    if total_col not in meta.columns:
        msg = f"metacatalog missing {total_col}"
        raise ValueError(msg)
    if "Peak_flux_VLSSR" not in meta.columns:
        msg = (
            "metacatalog missing Peak_flux_VLSSR; attach VLSSR "
            "(e.g. radio_crossmatch → metacatalog_spectral.parquet) first"
        )
        raise ValueError(msg)

    t = pd.to_numeric(meta[total_col], errors="coerce")
    v = pd.to_numeric(meta["Peak_flux_VLSSR"], errors="coerce")
    if "n_assoc_VLSSR" in meta.columns:
        n_assoc = pd.to_numeric(meta["n_assoc_VLSSR"], errors="coerce")
        ok = (n_assoc == 1) & (t > 0) & (v > 0)
    else:
        ok = (t > 0) & (v > 0)

    n = int(ok.sum())
    if n == 0:
        msg = "no unique VLSSR matches with positive Total and Peak_VLSSR"
        raise ValueError(msg)

    med = float(np.median((t / v).loc[ok]))
    if not np.isfinite(med) or med <= 0.0:
        msg = f"median Total/VLSSR Peak must be finite and > 0, got {med!r}"
        raise ValueError(msg)

    correction = float(np.sqrt(med))
    return scale * correction, correction, med, n


def bootstrap_beam_alpha(
    metacatalog: pd.DataFrame,
    *,
    config: BeamAlphaBootstrapConfig,
) -> BeamAlphaBootstrapResult:
    """Estimate per-band beam α from VLSSR anchor + neighbor Total/Peak ratios.

    Parameters
    ----------
    metacatalog
        Quality-filtered (or spectral) rows with ``Total_flux_*``,
        ``Peak_flux_*``, ``n_assoc_*`` / ``n_confused_*``, and unique VLSSR
        columns ``Peak_flux_VLSSR`` (+ ``n_assoc_VLSSR`` when present).
        Totals must come from a known detect-time α (see
        ``config.beam_scale_used``).
    config
        Required ``beam_scale_used`` (detect-time α; ``1.0`` if unscaled),
        anchor band, optional band list.
    """
    cfg = config
    warnings: list[str] = []
    _require_beam_scale_used(cfg.beam_scale_used)

    if cfg.bands is not None:
        bands = [b for b in cfg.bands if f"Total_flux_{b}" in metacatalog.columns]
    else:
        bands = lwa_bands_with_total_flux(metacatalog)

    if cfg.anchor_band not in bands:
        msg = f"anchor_band {cfg.anchor_band!r} not among bands with Total_flux: {bands}"
        raise ValueError(msg)

    alpha_anchor, alpha_correction, med_vlssr, n_vlssr = _alpha_anchor_from_vlssr(
        metacatalog,
        anchor_band=cfg.anchor_band,
        beam_scale_used=cfg.beam_scale_used,
    )

    alpha_by_band: dict[str, float] = {b: float("nan") for b in bands}
    alpha_by_band[cfg.anchor_band] = alpha_anchor
    i_anchor = bands.index(cfg.anchor_band)
    step_rows: list[dict] = []

    def _step(lo: str, hi: str, *, direction: str) -> None:
        med_t, n_t = median_neighbor_flux_ratio(
            metacatalog,
            lo,
            hi,
            flux_prefix="Total_flux",
            require_unconfused_unique=cfg.require_unconfused_unique,
        )
        med_p, n_p = median_neighbor_flux_ratio(
            metacatalog,
            lo,
            hi,
            flux_prefix="Peak_flux",
            require_unconfused_unique=cfg.require_unconfused_unique,
        )
        n = min(n_t, n_p)
        if not (np.isfinite(med_t) and np.isfinite(med_p) and med_t > 0 and med_p > 0):
            warnings.append(f"skip {lo}–{hi}: insufficient Total/Peak pairs")
            return
        # (T_hi/T_lo)/(P_hi/P_lo) = (α_lo/α_hi)^2  ⇒  α_hi/α_lo = √(P/T)
        factor = float(np.sqrt(med_p / med_t))
        if direction == "up":
            alpha_by_band[hi] = alpha_by_band[lo] * factor
        else:
            alpha_by_band[lo] = alpha_by_band[hi] / factor
        step_rows.append(
            {
                "band_lo": lo,
                "band_hi": hi,
                "direction": direction,
                "n": n,
                "median_T_hi_over_lo": med_t,
                "median_P_hi_over_lo": med_p,
                "alpha_hi_over_lo": factor,
                "alpha_lo": alpha_by_band[lo],
                "alpha_hi": alpha_by_band[hi],
            }
        )

    for i in range(i_anchor, len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        _step(lo, hi, direction="up")

    for i in range(i_anchor, 0, -1):
        lo, hi = bands[i - 1], bands[i]
        _step(lo, hi, direction="down")

    by_band = pd.DataFrame(
        {
            "band": bands,
            "alpha": [alpha_by_band[b] for b in bands],
            "nu_mhz": [band_frequency_hz(b) / 1e6 for b in bands],
        }
    )
    by_band["alpha_over_alpha_anchor"] = by_band["alpha"] / alpha_anchor
    steps = pd.DataFrame(step_rows, columns=list(BEAM_ALPHA_STEP_COLUMNS))

    return BeamAlphaBootstrapResult(
        by_band=by_band,
        steps=steps,
        alpha_anchor=float(alpha_anchor),
        alpha_correction=float(alpha_correction),
        median_total_over_vlssr=float(med_vlssr),
        n_vlssr_unique=int(n_vlssr),
        anchor_band=str(cfg.anchor_band),
        beam_scale_used=float(cfg.beam_scale_used),
        warnings=tuple(warnings),
    )


def summarize_beam_alpha_bootstrap(result: BeamAlphaBootstrapResult) -> str:
    """Short text summary of a bootstrap result."""
    alphas = pd.to_numeric(result.by_band["alpha"], errors="coerce").to_numpy(dtype=float)
    finite = alphas[np.isfinite(alphas) & (alphas > 0)]
    lines = [
        f"anchor {result.anchor_band}: α_true={result.alpha_anchor:.4f}  "
        f"= beam_scale_used={result.beam_scale_used:g} "
        f"× correction={result.alpha_correction:.4f}  "
        f"(√(med Total/VLSSR={result.median_total_over_vlssr:.4f}), "
        f"n_unique={result.n_vlssr_unique:,})",
    ]
    if finite.size:
        lines.append(
            f"α range {float(np.min(finite)):.4f} … {float(np.max(finite)):.4f}  "
            f"(spread {float(np.max(finite) / np.min(finite)):.3f}×; "
            f"median {float(np.median(finite)):.4f})"
        )
    if result.warnings:
        lines.append(f"warnings: {len(result.warnings)}")
    return "\n".join(lines)


def _beam_scale_payload_from_result(result: BeamAlphaBootstrapResult) -> dict:
    """JSON-serializable document for ``beam_scale.json``."""
    mapping = result.as_mapping()
    return {
        "BEAM_SCALE": mapping,
        "beam_scale_used": float(result.beam_scale_used),
        "anchor_band": str(result.anchor_band),
        "alpha_anchor": float(result.alpha_anchor),
        "alpha_correction": float(result.alpha_correction),
        "median_total_over_vlssr": float(result.median_total_over_vlssr),
        "n_vlssr_unique": int(result.n_vlssr_unique),
        "source": "beam_alpha_bootstrap",
    }


def _beam_scale_json_value(raw: object) -> float | dict[str, float | str]:
    """Serialize one band entry as a JSON number or ``{alpha, kappa, ...}``."""
    if isinstance(raw, BeamScaleModel):
        return raw.as_mapping()
    if isinstance(raw, Mapping):
        model = beam_scale_model_from_value(raw)  # type: ignore[arg-type]
        if model.kappa == 0.0 and "kappa" not in raw and "model" not in raw:
            return float(model.alpha)
        return model.as_mapping()
    scale = float(raw)  # type: ignore[arg-type]
    if not np.isfinite(scale) or scale <= 0.0:
        msg = f"BEAM_SCALE entry must be finite and > 0, got {raw!r}"
        raise ValueError(msg)
    return scale


def write_beam_scale_json(
    beam_scale: float
    | Mapping[str, float | BeamScaleModel | Mapping[str, object]]
    | BeamAlphaBootstrapResult,
    path: Path | str | CatalogLayout,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write ``beam_scale.json`` next to catalog Parquets.

    *beam_scale* may be a scalar, ``{band: α}``, ``{band: {alpha, kappa}}``,
    ``{band: BeamScaleModel}``, or a bootstrap
    :class:`BeamAlphaBootstrapResult` (includes VLSSR anchor metadata).
    """
    if isinstance(path, CatalogLayout):
        out = path.beam_scale_json()
    else:
        out = Path(path)
        if out.suffix.lower() != ".json":
            out = out / "beam_scale.json"

    if isinstance(beam_scale, BeamAlphaBootstrapResult):
        payload: dict = _beam_scale_payload_from_result(beam_scale)
    elif isinstance(beam_scale, Mapping):
        mapping: dict[str, float | dict[str, float | str]] = {}
        any_kappa = False
        for band, value in beam_scale.items():
            serialized = _beam_scale_json_value(value)
            mapping[str(band)] = serialized
            if isinstance(serialized, Mapping) and float(serialized.get("kappa", 0.0)) != 0.0:
                any_kappa = True
        source = "alpha_kappa_zenith_south" if any_kappa else "mapping"
        payload = {"BEAM_SCALE": mapping, "source": source}
        if any_kappa:
            kappas = [
                float(v["kappa"])
                for v in mapping.values()
                if isinstance(v, Mapping) and "kappa" in v
            ]
            if kappas and len(set(np.round(kappas, 12))) == 1:
                payload["kappa"] = float(kappas[0])
    else:
        scale = float(beam_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            msg = f"BEAM_SCALE must be finite and > 0, got {beam_scale!r}"
            raise ValueError(msg)
        payload = {"BEAM_SCALE": scale, "source": "scalar"}

    if extra:
        for key, value in extra.items():
            if key == "BEAM_SCALE":
                continue
            payload[str(key)] = value

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def read_beam_scale_json(path: Path | str | CatalogLayout) -> dict:
    """Read a ``beam_scale.json`` document (full payload, not only the mapping)."""
    if isinstance(path, CatalogLayout):
        p = path.beam_scale_json()
    else:
        p = Path(path)
    if not p.is_file():
        msg = f"beam_scale.json not found: {p}"
        raise FileNotFoundError(msg)
    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        msg = f"beam_scale.json root must be an object: {p}"
        raise ValueError(msg)
    return payload


def write_beam_alpha_table(
    result: BeamAlphaBootstrapResult,
    path: Path | str | CatalogLayout,
) -> Path:
    """Write ``beam_alpha_by_band.parquet``, steps sidecar, and ``beam_scale.json``."""
    if isinstance(path, CatalogLayout):
        layout = path
        out = layout.beam_alpha_by_band()
        json_path: Path | CatalogLayout = layout
    else:
        layout = None
        out = Path(path)
        json_path = out.parent

    write_table(result.by_band.loc[:, list(BEAM_ALPHA_BY_BAND_COLUMNS)], out)
    steps_path = out.with_name(out.stem + "_steps.parquet")
    if len(result.steps):
        write_table(result.steps.loc[:, list(BEAM_ALPHA_STEP_COLUMNS)], steps_path)

    if layout is not None:
        write_beam_scale_json(result, layout)
    else:
        write_beam_scale_json(result, Path(json_path) / "beam_scale.json")
    return out


def read_beam_alpha_table(path: Path | str | CatalogLayout) -> pd.DataFrame:
    """Read a ``beam_alpha_by_band`` Parquet written by :func:`write_beam_alpha_table`."""
    if isinstance(path, CatalogLayout):
        p = path.beam_alpha_by_band()
    else:
        p = Path(path)
    df = read_table(p)
    if "band" not in df.columns or "alpha" not in df.columns:
        msg = f"beam alpha table missing band/alpha columns: {p}"
        raise ValueError(msg)
    return df


def _mapping_from_band_models(
    raw: Mapping[str, object],
) -> dict[str, BeamScaleModel]:
    """Validate ``{band: α | {alpha, kappa, ...}}`` → models."""
    out: dict[str, BeamScaleModel] = {}
    for k, v in raw.items():
        out[str(k)] = beam_scale_model_from_value(v)  # type: ignore[arg-type]
    if not out:
        msg = "no finite positive alpha values in mapping"
        raise ValueError(msg)
    return out


def _mapping_from_band_alphas(raw: Mapping[str, object]) -> dict[str, float]:
    """Validate a plain ``{band: α}`` mapping (α only; strips κ)."""
    models = _mapping_from_band_models(raw)
    return {band: float(m.alpha) for band, m in models.items()}


def load_beam_scale(
    path: Path | str | CatalogLayout,
) -> float | dict[str, BeamScaleModel]:
    """Load ``BEAM_SCALE`` from ``beam_scale.json``.

    Scalar documents return a bare ``float`` (isotropic α, ``κ=0`` implied).
    Per-band mappings return ``dict[str, BeamScaleModel]`` (float entries
    promote to ``κ=0``).
    """
    payload = read_beam_scale_json(path)
    if "BEAM_SCALE" not in payload:
        msg = f"beam_scale.json missing BEAM_SCALE key: {path}"
        raise ValueError(msg)
    raw = payload["BEAM_SCALE"]
    if isinstance(raw, Mapping):
        return _mapping_from_band_models(raw)
    scale = float(raw)
    if not np.isfinite(scale) or scale <= 0.0:
        msg = f"BEAM_SCALE must be finite and > 0, got {raw!r}"
        raise ValueError(msg)
    return scale


def load_beam_alpha_mapping(
    path: Path | str | CatalogLayout | pd.DataFrame | Mapping[str, float],
) -> dict[str, float]:
    """Load ``{band: α}`` from JSON, parquet, layout, DataFrame, or mapping.

    For a :class:`CatalogLayout`, prefers ``beam_scale.json`` when present
    (and ``BEAM_SCALE`` is a mapping), else ``beam_alpha_by_band.parquet``.
    ``{alpha, kappa}`` JSON entries contribute their ``alpha`` only.
    """
    if isinstance(path, Mapping) and not isinstance(path, pd.DataFrame):
        if "BEAM_SCALE" in path:
            raw = path["BEAM_SCALE"]
            if isinstance(raw, Mapping):
                return _mapping_from_band_alphas(raw)
            msg = (
                "BEAM_SCALE in payload is scalar; use load_beam_scale() "
                "or pass a per-band mapping"
            )
            raise TypeError(msg)
        return _mapping_from_band_alphas(path)

    if isinstance(path, pd.DataFrame):
        df = path
    elif isinstance(path, CatalogLayout):
        json_path = path.beam_scale_json()
        if json_path.is_file():
            loaded = load_beam_scale(path)
            if isinstance(loaded, Mapping):
                return {band: float(m.alpha) for band, m in loaded.items()}
            msg = (
                f"{json_path} has scalar BEAM_SCALE={loaded!r}; "
                "need a per-band mapping (or use load_beam_scale)"
            )
            raise TypeError(msg)
        df = read_beam_alpha_table(path)
    else:
        p = Path(path)
        if p.suffix.lower() == ".json" or p.name == "beam_scale.json":
            loaded = load_beam_scale(p)
            if isinstance(loaded, Mapping):
                return {band: float(m.alpha) for band, m in loaded.items()}
            msg = (
                f"{p} has scalar BEAM_SCALE={loaded!r}; "
                "need a per-band mapping (or use load_beam_scale)"
            )
            raise TypeError(msg)
        df = read_beam_alpha_table(p)

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        a = float(pd.to_numeric(row["alpha"], errors="coerce"))
        if not np.isfinite(a) or a <= 0.0:
            continue
        out[str(row["band"])] = a
    if not out:
        msg = "no finite positive alpha values in beam alpha table"
        raise ValueError(msg)
    return out
