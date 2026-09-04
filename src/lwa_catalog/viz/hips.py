"""HiPS survey discovery helpers for ipyaladin viewers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

# Public remote HiPS roots for radio-survey visual QA (browser-fetched).
SURVEY_HIPS_URLS: dict[str, str] = {
    "VLSSR": "https://alasky.cds.unistra.fr/VLSSr/",
    "NVSS": "https://alasky.cds.unistra.fr/NVSS/intensity/",
    "VLASS": "https://vlass-dl.nrao.edu/vlass/HiPS/MedianStack/Quicklook/",
}


def survey_hips_url(survey: str) -> str:
    """Return the public HiPS root URL for *survey* (``VLSSR`` / ``NVSS`` / ``VLASS``)."""
    key = str(survey).strip().upper()
    try:
        url = SURVEY_HIPS_URLS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SURVEY_HIPS_URLS))
        msg = f"unknown survey HiPS {survey!r}; expected one of: {known}"
        raise ValueError(msg) from exc
    return url if url.endswith("/") else f"{url}/"


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
