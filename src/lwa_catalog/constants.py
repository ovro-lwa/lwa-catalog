"""Shared catalog constants (bands, column names)."""

from __future__ import annotations

COLOR_BANDS: tuple[str, ...] = ("Full", "Blue", "Green", "Red")
ASSOC_BANDS: tuple[str, ...] = ("Blue", "Green", "Red")

GAUL_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "E_RA",
    "E_DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "E_Peak_flux",
    "Maj",
    "E_Maj",
    "Min",
    "E_Min",
    "PA",
    "E_PA",
    "DC_Maj",
    "DC_Min",
    "DC_PA",
    "S_Code",
    "Gaus_id",
    "Isl_id",
    "Source_id",
)

# PyBDSF Gaussian numeric columns (everything in GAUL except codes/ids).
GAUL_FLOAT_COLUMNS: tuple[str, ...] = tuple(
    c for c in GAUL_COLUMNS if c not in {"S_Code", "Gaus_id", "Isl_id", "Source_id"}
)

BAND_FIELDS: tuple[str, ...] = (
    "Peak_flux",
    "Total_flux",
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

SOURCES_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "lst_hour",
    "band",
    "source_file",
    "BMAJ",
    "BMIN",
    "BPA",
    "time_key",
)
