"""Tests for image-level detection batching."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from lwa_catalog.create.detect import (
    DEFAULT_BDSF_KW,
    _clear_pybdsf_modules,
    _import_pybdsf_safely,
    detect_sources_many,
    iter_detect_sources,
)
from lwa_catalog.create.discover import FitsMetadata


def _meta(name: str) -> FitsMetadata:
    return FitsMetadata(path=Path(name), lst_hour="01h", band="Blue")


class _ImmediateFuture:
    def __init__(self, value: tuple[str, int]) -> None:
        self._value = value

    def result(self) -> tuple[str, int]:
        return self._value


class _FakePool:
    """Run submitted callables immediately; used to test as_completed order."""

    def submit(self, fn, *args):  # type: ignore[no-untyped-def]
        return _ImmediateFuture(fn(*args))

    def __enter__(self) -> _FakePool:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _paths_for(metas: list[FitsMetadata], tmp_path: Path) -> list[Path]:
    return [tmp_path / f"sources_{m.lst_hour}_{m.band}_{m.path.name}.parquet" for m in metas]


def test_detect_sources_many_empty() -> None:
    assert detect_sources_many([], []) == []
    assert list(iter_detect_sources([], [])) == []


def test_detect_sources_many_preserves_order(tmp_path: Path) -> None:
    metas = [_meta("a.fits"), _meta("b.fits"), _meta("c.fits")]
    paths = _paths_for(metas, tmp_path)
    frames = [pd.DataFrame({"RA": [i]}) for i in range(3)]

    with (
        patch(
            "lwa_catalog.create.detect.detect_sources",
            side_effect=frames,
        ) as detect,
        patch("lwa_catalog.create.detect.write_table") as write_table,
        patch(
            "lwa_catalog.io.read_table",
            side_effect=frames,
        ),
    ):
        out = detect_sources_many(
            metas,
            paths,
            n_jobs=1,
            bdsf_kw={"quiet": True},
        )

    assert len(out) == 3
    assert [len(df) for df in out] == [1, 1, 1]
    assert detect.call_count == 3
    assert write_table.call_count == 3
    assert [call.args[0].path.name for call in detect.call_args_list] == [
        "a.fits",
        "b.fits",
        "c.fits",
    ]


def test_iter_detect_sources_serial_yields_after_worker_write(tmp_path: Path) -> None:
    metas = [_meta("a.fits"), _meta("b.fits"), _meta("c.fits")]
    paths = _paths_for(metas, tmp_path)
    events: list[tuple[str, str]] = []

    def fake_detect(meta: FitsMetadata, **kwargs: object) -> pd.DataFrame:
        events.append(("detect", meta.path.name))
        return pd.DataFrame({"RA": [1.0]})

    with (
        patch("lwa_catalog.create.detect.detect_sources", side_effect=fake_detect),
        patch("lwa_catalog.create.detect.write_table") as write_table,
    ):
        for meta, out_path, n_sources in iter_detect_sources(metas, paths, n_jobs=1):
            events.append(("yield", meta.path.name))
            assert n_sources == 1
            assert out_path == paths[metas.index(meta)].resolve()

    assert write_table.call_count == 3
    assert events == [
        ("detect", "a.fits"),
        ("yield", "a.fits"),
        ("detect", "b.fits"),
        ("yield", "b.fits"),
        ("detect", "c.fits"),
        ("yield", "c.fits"),
    ]


def test_iter_detect_sources_yields_as_completed(tmp_path: Path) -> None:
    metas = [_meta("a.fits"), _meta("b.fits"), _meta("c.fits")]
    paths = _paths_for(metas, tmp_path)
    frames = {
        "a.fits": pd.DataFrame({"RA": [0.0]}),
        "b.fits": pd.DataFrame({"RA": [1.0]}),
        "c.fits": pd.DataFrame({"RA": [2.0]}),
    }

    def fake_detect(meta: FitsMetadata, **kwargs: object) -> pd.DataFrame:
        return frames[meta.path.name]

    with (
        patch("lwa_catalog.create.detect.detect_sources", side_effect=fake_detect),
        patch("lwa_catalog.create.detect.write_table"),
        patch("lwa_catalog.create.detect._import_pybdsf_safely"),
        patch("lwa_catalog.create.detect._fork_process_pool", return_value=_FakePool()),
        patch(
            "lwa_catalog.create.detect.as_completed",
            side_effect=lambda fs: reversed(list(fs)),
        ),
    ):
        yielded = list(iter_detect_sources(metas, paths, n_jobs=3))

    assert [meta.path.name for meta, _, _ in yielded] == ["c.fits", "b.fits", "a.fits"]
    assert [n_sources for _, _, n_sources in yielded] == [1, 1, 1]


def test_detect_sources_many_reorders_when_completion_is_reversed(tmp_path: Path) -> None:
    metas = [_meta("a.fits"), _meta("b.fits"), _meta("c.fits")]
    paths = _paths_for(metas, tmp_path)
    frames = {
        "a.fits": pd.DataFrame({"RA": [0.0]}),
        "b.fits": pd.DataFrame({"RA": [1.0]}),
        "c.fits": pd.DataFrame({"RA": [2.0]}),
    }

    def fake_detect(meta: FitsMetadata, **kwargs: object) -> pd.DataFrame:
        return frames[meta.path.name]

    with (
        patch("lwa_catalog.create.detect.detect_sources", side_effect=fake_detect),
        patch("lwa_catalog.create.detect.write_table"),
        patch("lwa_catalog.create.detect._import_pybdsf_safely"),
        patch("lwa_catalog.create.detect._fork_process_pool", return_value=_FakePool()),
        patch(
            "lwa_catalog.create.detect.as_completed",
            side_effect=lambda fs: reversed(list(fs)),
        ),
        patch(
            "lwa_catalog.io.read_table",
            side_effect=lambda path, **kwargs: frames[
                next(m.path.name for m, p in zip(metas, paths, strict=True) if p.resolve() == Path(path).resolve())
            ],
        ),
    ):
        out = detect_sources_many(metas, paths, n_jobs=3)

    assert [float(df["RA"].iloc[0]) for df in out] == [0.0, 1.0, 2.0]


def test_iter_detect_sources_requires_matching_paths() -> None:
    metas = [_meta("a.fits")]
    with pytest.raises(ValueError, match="catalog_paths length"):
        list(iter_detect_sources(metas, []))


def test_default_bdsf_kw_uses_single_core() -> None:
    assert DEFAULT_BDSF_KW["ncores"] == 1


def test_import_pybdsf_safely_after_spawn_context() -> None:
    pytest.importorskip("bdsf")
    _clear_pybdsf_modules()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pytest.skip("multiprocessing context cannot be reset in this process")

    bdsf = _import_pybdsf_safely()
    assert bdsf.__name__ == "bdsf"
    assert _import_pybdsf_safely() is bdsf
