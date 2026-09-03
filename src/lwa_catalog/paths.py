"""On-disk path layout for Parquet catalog artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lwa_catalog.constants import (
    METACATALOG_QUALITY_FILENAME,
    METACATALOG_RADIO_FILENAME,
    METACATALOG_SPECTRAL_FILENAME,
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

    def metacatalog_quality(self) -> Path:
        """QA metacatalog with ``quality_flag``: ``{root}/metacatalog_quality.parquet``."""
        return self.root / METACATALOG_QUALITY_FILENAME

    def metacatalog_spectral(self) -> Path:
        """Metacatalog with post-hoc ``spec_model_*`` columns."""
        return self.root / METACATALOG_SPECTRAL_FILENAME

    def metacatalog_radio(self) -> Path:
        """Radio-survey-attached metacatalog: ``{root}/metacatalog_radio.parquet``."""
        return self.root / METACATALOG_RADIO_FILENAME

    def with_metacatalog(self, metacatalog_file: str | Path) -> CatalogLayout:
        """Return a copy of this layout pointing at a different metacatalog file."""
        return CatalogLayout(self.root, metacatalog_file=str(metacatalog_file))
