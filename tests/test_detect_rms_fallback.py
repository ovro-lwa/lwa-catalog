"""Tests for PyBDSF constant-rms fallback on sliding-box RMS failures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.create.detect import (
    _is_recoverable_rms_error,
    _with_constant_rms_fallback,
    run_pybdsf_on_hdu,
)


def test_is_recoverable_rms_error_matches_known_failures() -> None:
    assert _is_recoverable_rms_error(
        FloatingPointError("divide by zero encountered in divide")
    )
    assert _is_recoverable_rms_error(
        RuntimeError(
            "A region with an unphysical rms value has been found. "
            "Please check the input image."
        )
    )
    assert _is_recoverable_rms_error(
        RuntimeError("Clipped rms appears to be zero. Check for regions with values of 0")
    )
    assert not _is_recoverable_rms_error(RuntimeError("unrelated failure"))
    assert not _is_recoverable_rms_error(ValueError("divide by zero encountered in divide"))


def test_with_constant_rms_fallback_overrides_adaptive_boxes() -> None:
    out = _with_constant_rms_fallback(
        {
            "thresh": "hard",
            "adaptive_rms_box": True,
            "rms_box": (128, 64),
            "rms_box_bright": (32, 8),
            "beam": (0.1, 0.1, 0.0),
        }
    )
    assert out["adaptive_rms_box"] is False
    assert out["rms_map"] is False
    assert out["mean_map"] == "const"
    assert out["rms_box"] is None
    assert out["rms_box_bright"] is None
    assert out["thresh"] == "hard"
    assert out["beam"] == (0.1, 0.1, 0.0)


def test_run_pybdsf_retries_with_constant_rms_on_divide_by_zero() -> None:
    hdu = fits.PrimaryHDU(
        data=[[1.0, 2.0], [3.0, 4.0]],
        header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1, "BPA": 0.0}),
    )
    table = Table({"RA": [10.0], "DEC": [20.0]})
    img = MagicMock()
    calls: list[dict[str, object]] = []

    def fake_process(hdu_arg: object, **kw: object) -> MagicMock:
        calls.append(kw)
        if len(calls) == 1:
            raise FloatingPointError("divide by zero encountered in divide")
        return img

    def fake_write_catalog(*, outfile: str, **_kwargs: object) -> None:
        with open(outfile, "wb") as fh:
            fh.write(b"gaul")

    img.write_catalog.side_effect = fake_write_catalog
    bdsf = MagicMock()
    bdsf.process_image.side_effect = fake_process

    with (
        patch("lwa_catalog.create.detect._import_pybdsf_safely", return_value=bdsf),
        patch("lwa_catalog.create.detect.Table.read", return_value=table),
        pytest.warns(UserWarning, match="constant rms/mean"),
    ):
        out = run_pybdsf_on_hdu(
            hdu,
            bdsf_kw={
                "adaptive_rms_box": True,
                "rms_box": (128, 64),
                "rms_box_bright": (32, 8),
            },
        )

    assert out is table
    assert len(calls) == 2
    assert calls[0]["adaptive_rms_box"] is True
    assert calls[0]["rms_box"] == (128, 64)
    assert calls[1]["adaptive_rms_box"] is False
    assert calls[1]["rms_map"] is False
    assert calls[1]["mean_map"] == "const"
    assert calls[1]["rms_box"] is None
    img.write_catalog.assert_called_once()


def test_run_pybdsf_does_not_retry_unrelated_runtime_error() -> None:
    hdu = fits.PrimaryHDU(
        data=[[1.0, 2.0], [3.0, 4.0]],
        header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1, "BPA": 0.0}),
    )
    bdsf = MagicMock()
    bdsf.process_image.side_effect = RuntimeError("some other PyBDSF failure")

    with (
        patch("lwa_catalog.create.detect._import_pybdsf_safely", return_value=bdsf),
        pytest.raises(RuntimeError, match="some other PyBDSF failure"),
    ):
        run_pybdsf_on_hdu(hdu)

    assert bdsf.process_image.call_count == 1
