"""Shared catalog constants (bands, column names)."""

from __future__ import annotations

import re
from pathlib import Path

COLOR_BANDS: tuple[str, ...] = ("Full", "Blue", "Green", "Red")
ASSOC_BANDS: tuple[str, ...] = ("Blue", "Green", "Red")

# HiPS / Aladin overlay colors (dark-sky friendly).
BAND_OVERLAY_COLORS: dict[str, str] = {
    "Red": "#ff5252",
    "Green": "#66bb6a",
    "Blue": "#42a5f5",
    "Full": "#fdd835",
}
BAND_OVERLAY_COLOR_UNKNOWN = "#bdbdbd"

# Owens Valley Radio Observatory / OVRO-LWA (geodetic latitude, degrees).
OVRO_LATITUDE_DEG: float = 37.239777

# VLSSR reference catalog (external; circular 80″ PSF, peak flux in Jy).
VLSSR_BMAJ_ARCSEC: float = 80.0
VLSSR_BMAJ_DEG: float = VLSSR_BMAJ_ARCSEC / 3600.0
VLSSR_DEFAULT_PATH: Path = Path("/fast/claw/vlssr_radecpeak.txt")

# NED Local Volume Sample (Cook et al. 2023; latest FITS from NED-LVS page).
NEDLVS_DEFAULT_PATH: Path = Path("/fast/claw/NEDLVS_current.fits")
# Fallback angular radius when ``Diam`` is missing (arcsec; NED-LVS median ~20″).
NEDLVS_DEFAULT_BMAJ_ARCSEC: float = 20.0
NEDLVS_DEFAULT_BMAJ_DEG: float = NEDLVS_DEFAULT_BMAJ_ARCSEC / 3600.0

_SUBBAND_FREQ_RE = re.compile(r"^(\d+)MHz$", re.IGNORECASE)

# OVRO-LWA frequency-subband range for red (low) → blue (high) overlay colors.
SUBBAND_FREQ_MHZ_MIN: float = 18.0
SUBBAND_FREQ_MHZ_MAX: float = 82.0


def normalize_band_label(label: str) -> str | None:
    """Return canonical color-band or ``{n}MHz`` subband label, else ``None``."""
    text = str(label).strip()
    if not text:
        return None
    for band in COLOR_BANDS:
        if text.lower() == band.lower():
            return band
    match = _SUBBAND_FREQ_RE.match(text)
    if match:
        return f"{int(match.group(1))}MHz"
    return None


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def lerp_hex(hex_low: str, hex_high: str, fraction: float) -> str:
    """Linearly interpolate between two ``#rrggbb`` colors."""
    t = max(0.0, min(1.0, float(fraction)))
    r0, g0, b0 = _hex_to_rgb(hex_low)
    r1, g1, b1 = _hex_to_rgb(hex_high)
    return _rgb_to_hex(
        round(r0 + (r1 - r0) * t),
        round(g0 + (g1 - g0) * t),
        round(b0 + (b1 - b0) * t),
    )


def subband_frequency_color(
    band: str,
    *,
    freq_mhz_min: float | None = None,
    freq_mhz_max: float | None = None,
    color_low: str = BAND_OVERLAY_COLORS["Red"],
    color_high: str = BAND_OVERLAY_COLORS["Blue"],
) -> str | None:
    """Map an ``{n}MHz`` subband label to a red (low) → blue (high) hex color."""
    normalized = normalize_band_label(band)
    if normalized is None:
        return None
    match = _SUBBAND_FREQ_RE.match(normalized)
    if match is None:
        return None
    mhz = float(match.group(1))
    lo = SUBBAND_FREQ_MHZ_MIN if freq_mhz_min is None else freq_mhz_min
    hi = SUBBAND_FREQ_MHZ_MAX if freq_mhz_max is None else freq_mhz_max
    if hi <= lo:
        fraction = 0.5
    else:
        fraction = (mhz - lo) / (hi - lo)
    return lerp_hex(color_low, color_high, fraction)


def band_frequency_hz(band: str) -> float:
    """Return rest-frame center frequency (Hz) for a color or subband label."""
    key = str(band).strip()
    if key in BAND_FREQ_HZ:
        return float(BAND_FREQ_HZ[key])
    match = _SUBBAND_FREQ_RE.match(key)
    if match:
        return float(match.group(1)) * 1e6
    return float("nan")

# Rest-frame center frequencies (Hz) from RESTFRQ on OVRO-LWA deep color products.
BAND_FREQ_HZ: dict[str, float] = {
    "Blue": 73_956_883.683421,
    "Green": 51_675_007.041984,
    "Red": 34_599_742.150104,
    "Full": 57_200_637.276411,
}

# (label, band_lo_or_a, band_hi_or_b) for α = log(S_a/S_b) / log(ν_a/ν_b).
SPECTRAL_INDEX_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("RG", "Red", "Green"),
    ("GB", "Green", "Blue"),
)

GAUL_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "E_Peak_flux",
    "Maj",
    "Min",
    "PA",
    "DC_Maj",
    "DC_Min",
    "DC_PA",
    "Resid_Isl_rms",
    "Resid_Isl_mean",
)

# PyBDSF Gaussian numeric columns (currently the full GAUL list).
GAUL_FLOAT_COLUMNS: tuple[str, ...] = tuple(GAUL_COLUMNS)

# Former GAUL columns dropped from the detection default; rewrite helpers strip these.
DROPPED_GAUL_COLUMNS: frozenset[str] = frozenset(
    {
        "E_RA",
        "E_DEC",
        "E_Maj",
        "E_Min",
        "E_PA",
        "S_Code",
        "Gaus_id",
        "Isl_id",
        "Source_id",
    }
)

BAND_FIELDS: tuple[str, ...] = (
    "Peak_flux",
    "Total_flux",
    "E_Total_flux",
    "RA",
    "DEC",
    "Maj",
    "Min",
    "PA",
    "DC_Maj",
    "DC_Min",
    "DC_PA",
)

METACATALOG_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "RA",
        "DEC",
        "Peak_flux",
        "origin_band",
        "bands_present",
    }
)

SOURCES_REQUIRED_COLUMNS: frozenset[str] = frozenset({"RA", "DEC"})

# LST-merge QA columns propagated to metacatalog (origin-band row).
CLUSTER_JITTER_RMS_COL: str = "cluster_jitter_rms_deg"
LST_MERGE_QA_COLUMNS: tuple[str, ...] = (
    CLUSTER_JITTER_RMS_COL,
    "Resid_Isl_rms",
    "Resid_Isl_mean",
    "E_Peak_flux",
    "E_Total_flux",
)

SOURCES_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "lst_hour",
    "band",
    "source_file",
    "BMAJ",
    "BMIN",
    "BPA",
    "time_key",
)
