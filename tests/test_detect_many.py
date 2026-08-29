"""Tests for image-level detection batching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lwa_catalog.create.detect import DEFAULT_BDSF_KW, detect_sources_many
from lwa_catalog.create.discover import FitsMetadata


def _meta(name: str) -> FitsMetadata:
    return FitsMetadata(path=Path(name), lst_hour="01h", band="Blue")


def test_detect_sources_many_empty() -> None:
    assert detect_sources_many([]) == []


def test_detect_sources_many_preserves_order() -> None:
    metas = [_meta("a.fits"), _meta("b.fits"), _meta("c.fits")]
    frames = [pd.DataFrame({"RA": [i]}) for i in range(3)]

    with patch(
        "lwa_catalog.create.detect.detect_sources",
        side_effect=frames,
    ) as detect:
        out = detect_sources_many(metas, n_jobs=1, bdsf_kw={"quiet": True})

    assert len(out) == 3
    assert [len(df) for df in out] == [1, 1, 1]
    assert detect.call_count == 3
    assert [call.args[0].path.name for call in detect.call_args_list] == [
        "a.fits",
        "b.fits",
        "c.fits",
    ]


def test_default_bdsf_kw_uses_single_core() -> None:
    assert DEFAULT_BDSF_KW["ncores"] == 1
