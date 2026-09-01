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
