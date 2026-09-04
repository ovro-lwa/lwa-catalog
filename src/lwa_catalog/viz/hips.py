"""HiPS survey discovery helpers for ipyaladin viewers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

# Public remote HiPS roots / Aladin IDs for radio-survey visual QA (browser-fetched).
# VLASS is the Quicklook Median Stack (CDS/Aladin ID ``NRAO/P/VLASS-Quicklook-MedianStack``).
# NRAO's VLASS image cache is sometimes offline; when tile fetches fail, Aladin may keep
# showing the previous survey (e.g. NVSS). Override with a local mirror if needed.
SURVEY_HIPS_URLS: dict[str, str] = {
    "VLSSR": "https://alasky.cds.unistra.fr/VLSSr/",
    "NVSS": "https://alasky.cds.unistra.fr/NVSS/intensity/",
    "VLASS": "https://vlass-dl.nrao.edu/vlass/HiPS/MedianStack/Quicklook/",
}

# Canonical Aladin Lite / CDS HiPS identifier for the VLASS median stack.
VLASS_MEDIAN_HIPS_ID: str = "NRAO/P/VLASS-Quicklook-MedianStack"


def _normalize_hips_ref(ref: str) -> str:
    """Ensure HTTP HiPS roots end with ``/``; leave Aladin survey IDs unchanged."""
    text = str(ref).strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text if text.endswith("/") else f"{text}/"
    return text


def survey_hips_url(
    survey: str,
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Return the HiPS root URL or Aladin ID for *survey* (``VLSSR`` / ``NVSS`` / ``VLASS``).

    Parameters
    ----------
    survey
        Survey name (case-insensitive).
    overrides
        Optional mapping of survey name → HiPS URL or Aladin ID. Keys are
        matched case-insensitively and replace the built-in defaults.
    """
    key = str(survey).strip().upper()
    table = dict(SURVEY_HIPS_URLS)
    if overrides:
        table.update({str(k).strip().upper(): str(v).strip() for k, v in overrides.items()})
    try:
        ref = table[key]
    except KeyError as exc:
        known = ", ".join(sorted(table))
        msg = f"unknown survey HiPS {survey!r}; expected one of: {known}"
        raise ValueError(msg) from exc
    return _normalize_hips_ref(ref)


def discover_local_hips_surveys(catalog_dir: Path) -> list[str]:
    """Find HiPS tile directories under *catalog_dir* (dirs containing ``properties``)."""
    root = Path(catalog_dir)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "properties").is_file()
    )


def fetch_hips_surveys(
    list_base: str,
    *,
    catalog_dir: Path | None = None,
    default_survey: str,
    timeout: float = 5.0,
) -> list[str]:
    """Return HiPS survey names from *list_base*, with optional local fallback."""
    local = discover_local_hips_surveys(catalog_dir) if catalog_dir is not None else []
    url = f"{list_base.rstrip('/')}/cgi-bin/list-hips.py"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            remote = [str(s) for s in json.loads(resp.read().decode())]
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        if local:
            return local
        return [default_survey]
    return sorted({*remote, *local})


def default_hips_survey(surveys: list[str], *, default_survey: str) -> str:
    """Pick a sensible default HiPS survey from the server list."""
    if default_survey in surveys:
        return default_survey
    for key in (
        "metacatalog_coadd2_full",
        "metacatalog_coadd2_gold",
        "metacatalog_coadd2_cleaned",
        "Full_I_deep",
    ):
        hit = next((s for s in surveys if key in s), None)
        if hit is not None:
            return hit
    return surveys[0] if surveys else default_survey


def hips_survey_url(survey: str, *, base: str) -> str:
    """Build the HiPS root URL passed to ipyaladin (trailing slash for Aladin Lite)."""
    survey = survey.strip().strip("/")
    if survey.startswith("http://") or survey.startswith("https://"):
        return survey if survey.endswith("/") else f"{survey}/"
    return f"{base.rstrip('/')}/{survey}/"
