"""Dec-dependent restoring-beam scale ``(α, κ)`` for HEALPix-tile detect.

Physical model (zenith-south Maj elongation)::

    u(δ) = max(0, φ − δ)          # deg south of OVRO zenith
    E(δ; κ) = 1 + κ · u(δ)        # Maj stretch; κ in deg⁻¹
    BMIN_eff = α · BMIN_header
    BMAJ_eff = α · BMAJ_header · E(δ; κ)
    BPA_eff  = 90°                # E–W; HEALPix formation

See ``.agents/plan-anisotropic-header-beam-alpha.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from lwa_catalog.constants import OVRO_LATITUDE_DEG

__all__ = [
    "HEALPIX_ATTACH_BPA_DEG",
    "MODEL_ZENITH_SOUTH",
    "BeamScaleModel",
    "beam_scale_model_from_value",
    "resolve_beam_scale_model",
    "zenith_south_elongation",
    "zenith_south_u",
]

MODEL_ZENITH_SOUTH: str = "zenith_south"
HEALPIX_ATTACH_BPA_DEG: float = 90.0

# Avoid circular import of healpix_detect.DEFAULT_BEAM_SCALE at module load.
_DEFAULT_ALPHA: float = 1.0


def zenith_south_u(dec_deg: float, *, phi_deg: float = OVRO_LATITUDE_DEG) -> float:
    """Degrees south of zenith; ``0`` at or above ``phi_deg``."""
    return float(max(0.0, float(phi_deg) - float(dec_deg)))


def zenith_south_elongation(
    dec_deg: float,
    kappa: float,
    *,
    phi_deg: float = OVRO_LATITUDE_DEG,
) -> float:
    """Maj stretch ``E = 1 + κ · u(δ)`` (``≥ 1`` when ``κ ≥ 0``)."""
    return float(1.0 + float(kappa) * zenith_south_u(dec_deg, phi_deg=phi_deg))


@dataclass(frozen=True, slots=True)
class BeamScaleModel:
    """Per-band isotropic α plus zenith-south Maj elongation κ."""

    alpha: float
    kappa: float = 0.0
    model: str = MODEL_ZENITH_SOUTH
    phi_deg: float = OVRO_LATITUDE_DEG

    def __post_init__(self) -> None:
        a = float(self.alpha)
        k = float(self.kappa)
        if not np.isfinite(a) or a <= 0.0:
            msg = f"alpha must be finite and > 0, got {self.alpha!r}"
            raise ValueError(msg)
        if not np.isfinite(k) or k < 0.0:
            msg = f"kappa must be finite and >= 0, got {self.kappa!r}"
            raise ValueError(msg)
        if self.model != MODEL_ZENITH_SOUTH:
            msg = f"unsupported beam scale model {self.model!r}; expected {MODEL_ZENITH_SOUTH!r}"
            raise ValueError(msg)
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "kappa", k)
        object.__setattr__(self, "phi_deg", float(self.phi_deg))

    @classmethod
    def isotropic(cls, alpha: float = _DEFAULT_ALPHA, *, phi_deg: float = OVRO_LATITUDE_DEG) -> BeamScaleModel:
        """``κ = 0`` model (scalar beam scale)."""
        return cls(alpha=float(alpha), kappa=0.0, phi_deg=phi_deg)

    def u(self, dec_deg: float) -> float:
        return zenith_south_u(dec_deg, phi_deg=self.phi_deg)

    def elongation(self, dec_deg: float) -> float:
        return zenith_south_elongation(dec_deg, self.kappa, phi_deg=self.phi_deg)

    def axis_scales(self, dec_deg: float) -> tuple[float, float]:
        """Return ``(scale_maj, scale_min)`` at *dec_deg*."""
        e = self.elongation(dec_deg)
        return float(self.alpha * e), float(self.alpha)

    def effective_beam(
        self,
        bmaj: float,
        bmin: float,
        bpa: float = 0.0,
        *,
        dec_deg: float,
        attach_bpa_deg: float = HEALPIX_ATTACH_BPA_DEG,
    ) -> tuple[float, float, float]:
        """Scaled ``(BMAJ, BMIN, BPA)``; *bpa* ignored (HEALPix attach uses *attach_bpa_deg*)."""
        del bpa  # header / caller BPA is not used on the HEALPix-tile path
        s_maj, s_min = self.axis_scales(dec_deg)
        out_maj = float(s_maj * float(bmaj))
        out_min = float(s_min * float(bmin))
        # PyBDSF expects BMAJ >= BMIN; swap if a pathological header inverts them.
        if out_min > out_maj:
            out_maj, out_min = out_min, out_maj
        return (out_maj, out_min, float(attach_bpa_deg))

    def as_mapping(self) -> dict[str, float | str]:
        """JSON-friendly band value (plus model metadata)."""
        return {
            "alpha": float(self.alpha),
            "kappa": float(self.kappa),
            "model": str(self.model),
            "phi_deg": float(self.phi_deg),
        }


def beam_scale_model_from_value(
    raw: float | BeamScaleModel | Mapping[str, Any],
    *,
    default_phi_deg: float = OVRO_LATITUDE_DEG,
) -> BeamScaleModel:
    """Coerce a scalar, model, or ``{alpha, kappa, ...}`` mapping to :class:`BeamScaleModel`."""
    if isinstance(raw, BeamScaleModel):
        return raw
    if isinstance(raw, Mapping):
        if "alpha" not in raw:
            msg = f"beam scale mapping must include 'alpha', got keys {sorted(raw)!r}"
            raise ValueError(msg)
        alpha = float(raw["alpha"])  # type: ignore[arg-type]
        kappa = float(raw.get("kappa", 0.0))  # type: ignore[arg-type]
        model = str(raw.get("model", MODEL_ZENITH_SOUTH))
        phi = float(raw["phi_deg"]) if "phi_deg" in raw else float(default_phi_deg)  # type: ignore[index]
        return BeamScaleModel(alpha=alpha, kappa=kappa, model=model, phi_deg=phi)
    return BeamScaleModel.isotropic(float(raw), phi_deg=default_phi_deg)


def resolve_beam_scale_model(
    band: str,
    beam_scale: float | BeamScaleModel | Mapping[str, Any],
    *,
    default: float | BeamScaleModel | None = None,
) -> BeamScaleModel:
    """Return a :class:`BeamScaleModel` for *band*.

    *beam_scale* may be a scalar α (``κ=0``), a single :class:`BeamScaleModel`,
    or a ``{band: α | {alpha, kappa, ...}}`` mapping. Missing *band* uses
    *default* or isotropic ``1.0``.
    """
    if isinstance(beam_scale, BeamScaleModel):
        return beam_scale
    if isinstance(beam_scale, Mapping) and "alpha" not in beam_scale:
        # Per-band mapping (values are scalars or ``{alpha, kappa}`` dicts).
        if band in beam_scale:
            return beam_scale_model_from_value(beam_scale[band])  # type: ignore[arg-type]
        if default is None:
            return BeamScaleModel.isotropic(_DEFAULT_ALPHA)
        return beam_scale_model_from_value(default)
    return beam_scale_model_from_value(beam_scale)  # type: ignore[arg-type]
