"""On-disk path layout for Parquet catalog artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lwa_catalog.constants import (
    BEAM_ALPHA_BY_BAND_FILENAME,
    BEAM_SCALE_JSON_FILENAME,
    METACATALOG_ASSOCIATIONS_FILENAME,
    METACATALOG_QUALITY_FLAGS_FILENAME,
    METACATALOG_RADIO_FILENAME,
    METACATALOG_SPECTRAL_FILENAME,
    METACATALOG_XRAY_FILENAME,
    RESIDUAL_MATCHED_FILENAME_TEMPLATE,
)


@dataclass(frozen=True, slots=True)
class CatalogLayout:
    """Directory layout for sources / LST-merged / global metacatalog Parquet files."""

    root: Path
    metacatalog_file: str = "metacatalog.parquet"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        meta = str(self.metacatalog_file).strip()
        if not meta:
            raise ValueError("metacatalog_file must be a non-empty path")
        object.__setattr__(self, "metacatalog_file", meta)

    def sources(self, lst_hour: str, band: str) -> Path:
        """Per-image sources catalog: ``sources_{lst}_{band}.parquet``."""
        return self.root / f"sources_{lst_hour}_{band}.parquet"

    def lst_merged(self, band: str) -> Path:
        """LST-merged per-band catalog: ``metacatalog_lst_{band}.parquet``."""
        return self.root / f"metacatalog_lst_{band}.parquet"

    def metacatalog(self) -> Path:
        """Global metacatalog Parquet path.

        Defaults to ``{root}/metacatalog.parquet``. Set :attr:`metacatalog_file`
        (or :meth:`with_metacatalog`) for an alternate filename under *root* or
        an absolute path elsewhere.
        """
        path = Path(self.metacatalog_file)
        if path.is_absolute():
            return path
        return self.root / path

    def metacatalog_parquet(self) -> Path:
        """Alias for :meth:`metacatalog`."""
        return self.metacatalog()

    def metacatalog_quality_flags(self) -> Path:
        """Per-bit quality boolean table: ``{root}/metacatalog_quality_flags.parquet``."""
        return self.root / METACATALOG_QUALITY_FLAGS_FILENAME

    def metacatalog_spectral(self) -> Path:
        """Analysis subset: optional survey cols + optional ``spec_*`` fits."""
        return self.root / METACATALOG_SPECTRAL_FILENAME

    def metacatalog_radio(self) -> Path:
        """Legacy radio-sidecar path; prefer :meth:`metacatalog_spectral`."""
        return self.root / METACATALOG_RADIO_FILENAME

    def metacatalog_xray(self) -> Path:
        """Legacy x-ray sidecar path; new writes use :meth:`metacatalog_associations`."""
        return self.root / METACATALOG_XRAY_FILENAME

    def metacatalog_associations(self) -> Path:
        """Long-form unique cross-match associations table."""
        return self.root / METACATALOG_ASSOCIATIONS_FILENAME

    def beam_alpha_by_band(self) -> Path:
        """Per-band beam α table from ``beam_alpha_bootstrap.ipynb``."""
        return self.root / BEAM_ALPHA_BY_BAND_FILENAME

    def beam_scale_json(self) -> Path:
        """``BEAM_SCALE`` JSON sidecar next to catalog Parquets."""
        return self.root / BEAM_SCALE_JSON_FILENAME

    def residual_matched(self, band: str) -> Path:
        """Stage-3 matched leftover sidecar: ``residual_matched_{band}.parquet``."""
        return self.root / RESIDUAL_MATCHED_FILENAME_TEMPLATE.format(band=band)

    def with_metacatalog(self, metacatalog_file: str | Path) -> CatalogLayout:
        """Return a copy of this layout pointing at a different metacatalog file."""
        return CatalogLayout(self.root, metacatalog_file=str(metacatalog_file))
