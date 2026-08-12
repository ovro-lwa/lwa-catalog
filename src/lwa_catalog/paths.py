"""On-disk path layout for Parquet catalog artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CatalogLayout:
    """Directory layout for sources / LST-merged / global metacatalog Parquet files."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def sources(self, lst_hour: str, band: str) -> Path:
        """Per-image sources catalog: ``sources_{lst}_{band}.parquet``."""
        return self.root / f"sources_{lst_hour}_{band}.parquet"

    def lst_merged(self, band: str) -> Path:
        """LST-merged per-band catalog: ``metacatalog_lst_{band}.parquet``."""
        return self.root / f"metacatalog_lst_{band}.parquet"

    def metacatalog(self) -> Path:
        """Global metacatalog: ``metacatalog.parquet``."""
        return self.root / "metacatalog.parquet"

    def metacatalog_parquet(self) -> Path:
        """Alias for :meth:`metacatalog`."""
        return self.metacatalog()
