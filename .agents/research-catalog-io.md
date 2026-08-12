# Research: Catalog Read/Write in Metacatalog Notebooks

---
**Date:** 2026-08-12
**Author:** AI Assistant
**Status:** Active
**Related Documents:**
- [Plan: Catalog I/O Layer](plan-catalog-io.md)

---

## Research Question

What reading and writing of catalog (and related FITS) artifacts exists in the
`lwa-catalog` metacatalog notebooks, and how is persistence organized today?

## Executive Summary

Catalog persistence is implemented entirely inside
`notebooks/ovro_lwa_metacatalog.ipynb`. Artifacts are written as CSV under a
user-configured `OUTPUT_DIR`, with a dual CSV+FITS write only for the final
global metacatalog. Caching is path-existence based (`REUSE_CACHED_CATALOGS`).
`notebooks/metacatalog_sky_view.ipynb` only reads `metacatalog.csv` (plus FITS
for visualization overlays). The package stub `src/lwa_catalog/io.py` already
provides generic CSV/FITS DataFrame round-trip and a minimal metacatalog column
check, but does not encode path conventions, cache checks, or beam backfill.

## Scope

**What This Research Covers:**
- Catalog and FITS I/O call sites in both notebooks
- On-disk naming and artifact layers
- Existing library stubs related to I/O

**What This Research Does NOT Cover:**
- LST/band merge algorithms (except as producers of tables that get written)
- PyBDSF detection internals beyond HDU prepare / catalog export
- SkyWidget / Panel visualization I/O

## Key Findings

### Artifact layers (`ovro_lwa_metacatalog.ipynb`)

| Layer | Filename | Writer | Reader |
| ----- | -------- | ------ | ------ |
| Per-image sources | `sources_{lst_hour}_{band}.csv` | after `detect_sources` | `load_sources_catalog` (+ BMAJ backfill) |
| LST-merged band | `metacatalog_lst_{band}.csv` | after `merge_lst_metacatalog` | `load_lst_merged_from_disk` |
| Global metacatalog | `metacatalog.csv`, `metacatalog.fits` | end of merge cell | sky-view `pd.read_csv` |

Path helpers in the notebook: `sources_csv_path`, `lst_merged_csv_path`,
`all_sources_cached`, `all_lst_merged_cached`.

### Per-image schema

PyBDSF `gaul` columns (`GAUL_COLUMNS`) plus provenance:
`lst_hour`, `band`, `source_file`, `BMAJ`, `BMIN`, `BPA`, optional `time_key`.
Empty detections still write a header-only CSV via `empty_sources_dataframe`.

Legacy CSVs missing beam columns are repaired on read by opening the slot’s
FITS header (`_ensure_bmaj_column`).

### Sky-view read path (`metacatalog_sky_view.ipynb`)

- Config: `METACATALOG_CSV`
- `pd.read_csv` → `validate_metacatalog_csv` (requires `bands_present`,
  `origin_band`, `RA`, `DEC`, `Peak_flux`, and related association columns)
- Separate FITS discovery/load for image overlays (not catalog persistence)

### Library stubs

- `src/lwa_catalog/io.py` — `read_catalog`, `write_catalog`,
  `validate_metacatalog`
- `src/lwa_catalog/create/{detect,merge}.py` — `NotImplementedError` stubs
- `src/lwa_catalog/analyze/summary.py` — in-memory summary helpers

## Architecture Overview

```text
FITS tree (FITS_ROOT)
    → discover / parse FitsMetadata
    → detect_sources → DataFrame
    → write sources_{lst}_{band}.csv
    → merge_lst_metacatalog → write metacatalog_lst_{band}.csv
    → build_global_metacatalog → write metacatalog.{csv,fits}
                                    ↓
                         metacatalog_sky_view read CSV
```

## References

**Files Analyzed:**
- `notebooks/ovro_lwa_metacatalog.ipynb`
- `notebooks/metacatalog_sky_view.ipynb`
- `src/lwa_catalog/io.py`
- `src/lwa_catalog/create/detect.py`
- `src/lwa_catalog/create/merge.py`

## Addendum — 2026-08-12

Planned persistence format for new work is **Apache Parquet via PyArrow**
(see [plan-catalog-io.md](plan-catalog-io.md) v1.1). This research document
still describes the **as-is** notebook CSV/FITS behavior.
