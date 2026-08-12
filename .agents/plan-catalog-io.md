# Implementation Plan: Catalog I/O Layer (Apache Arrow)

---
**Date:** 2026-08-12
**Author:** AI Assistant
**Status:** In Progress (Phases 0–3 complete; Phase 3 manual verify pending)
**Related Documents:**
- [Research: Catalog Read/Write in Metacatalog Notebooks](research-catalog-io.md)

---

## Overview

Extract catalog path conventions and read/write helpers from the metacatalog
notebooks into `lwa_catalog`, and **migrate on-disk catalogs from CSV / FITS
tables to Apache Arrow–backed Parquet**. In-memory interchange uses
`pyarrow.Table`; notebooks may continue to work with pandas via
`Table.to_pandas()` / `Table.from_pandas()`.

Detection and merge algorithms stay in `create/` (stubs or notebook-local until
later plans). This plan covers **paths**, **Arrow/Parquet I/O**, **schemas**,
**beam backfill on sources read**, **legacy CSV/FITS import**, and a **notebook
adapter**.

**Goal:** Persist and reload the three catalog layers through `lwa_catalog` as
Parquet files with stable Arrow schemas, same logical columns as today.

**Motivation:** CSV lacks a durable schema and is slow/lossy for typed science
columns; FITS binary tables are awkward for multi-format provenance strings and
duplicate the global catalog. Arrow/Parquet gives typed columns, compression,
and a single format for all layers
([PyArrow docs](https://arrow.apache.org/docs/python/index.html),
[Parquet](https://arrow.apache.org/docs/python/parquet.html)).

## Current State Analysis

**Existing Implementation:**
- `notebooks/ovro_lwa_metacatalog.ipynb` — CSV path helpers, cache gates,
  `DataFrame.to_csv`, global dual write `metacatalog.csv` + `.fits`
- `notebooks/metacatalog_sky_view.ipynb` — `pd.read_csv(METACATALOG_CSV)` +
  column validation
- `src/lwa_catalog/io.py` — generic CSV/FITS `read_catalog` / `write_catalog` /
  `validate_metacatalog`
- `tests/test_import.py` — CSV round-trip smoke tests

**Current Behavior:**
Three artifact layers under `OUTPUT_DIR` as CSV (plus FITS twin for global
metacatalog only). Cache reuse is file-existence based.

**Current Limitations:**
- Untyped CSV; pandas infers types inconsistently across reloads
- Dual global formats to maintain
- Path/cache logic closed over notebook globals
- No Arrow dependency or schema registry yet

## Desired End State

**New Behavior:**
- `CatalogLayout(root=...)` names **`.parquet`** paths for all three layers
- `read_*` / `write_*` use `pyarrow.parquet` (`pq.read_table` /
  `pq.write_table`) around `pyarrow.Table`
- Public layered APIs accept/return **pandas DataFrame by default** (notebook
  ergonomics) with optional `as_table=True` returning `pa.Table`
- Canonical **Arrow schemas** declared in code for sources / LST-merged /
  metacatalog (column names unchanged from notebooks)
- `write_metacatalog` writes **one** `metacatalog.parquet` (no CSV/FITS pair)
- Optional `import_legacy_catalogs(layout, …)` converts existing CSV/FITS trees
  to Parquet once
- Sky-view loads via `read_metacatalog`

**Success Looks Like:**
- Unit tests round-trip Parquet without PyBDSF; schemas survive empty tables
- Build + sky-view notebooks use library Parquet I/O
- Legacy CSV trees can be imported to Parquet for cache reuse
- `pyarrow` is a core dependency in `pyproject.toml`

## What We're NOT Doing

- [ ] Porting PyBDSF `detect_sources` / merge algorithms (only
      `read_beam_from_fits` for backfill)
- [ ] SkyWidget / FITS **image** overlay loaders (FITS remains for images)
- [ ] CLI entry points
- [ ] Changing logical column **names** / science schema of catalogs
- [ ] Keeping CSV or FITS catalog tables as a supported write path
- [ ] Arrow IPC (`.arrow`) / Feather as the primary on-disk format (see
      decisions)
- [ ] Arrow Dataset multi-file hive partitioning (single file per artifact is
      enough at current scale)

**Rationale:** Focus on a clean Arrow/Parquet persistence contract; detect/merge
and viz can build on it later.

## Implementation Approach

**Technical Strategy:**
Add `CatalogLayout` for `.parquet` paths; implement I/O on `pyarrow.Table` with
pandas adapters at the API boundary; register explicit schemas; provide a
one-shot legacy importer for existing `/fast/claw/metacatalog` CSV trees.

**Key Architectural Decisions:**

1. **Decision:** On-disk format is **Apache Parquet** via `pyarrow.parquet`.
   - **Rationale:** Durable, compressed, schema-aware; first-class in the Arrow
     Python stack
     ([Parquet guide](https://arrow.apache.org/docs/python/parquet.html)).
   - **Trade-offs:** Not human-grepable like CSV; needs `pyarrow`.
   - **Alternatives considered:**
     - Arrow IPC file (`.arrow`) —
       ([IPC docs](https://arrow.apache.org/docs/python/ipc.html)): excellent for
       mmap, less conventional for “catalog product” interchange.
     - Keep CSV + add Arrow only in-memory: rejects the migration request.
     - Feather: effectively IPC; prefer Parquet for compression and tooling.

2. **Decision:** In-memory canonical type is `pyarrow.Table`; layered public
   helpers default to pandas DataFrame in/out.
   - **Rationale:** Notebooks and merge code are pandas-centric today; Arrow is
     the storage/schema layer.
   - **Trade-offs:** Conversion cost on each read/write (acceptable at catalog
     sizes).
   - **Alternatives considered:** Arrow-only API — premature for Phase 3
     notebooks.

3. **Decision:** One Parquet file per artifact; **no** dual CSV/FITS write.
   - **Rationale:** Dual formats were a CSV-era workaround.
   - **Trade-offs:** External tools that only read FITS catalogs need an export
     helper later (out of scope unless requested).

4. **Decision:** Explicit `pa.schema([...])` per layer (sources, lst-merged,
   metacatalog), applied on write; read validates required field names (and
   optionally types).
   - **Rationale:** Avoid CSV-style type drift (`lst_hour` as int vs str, etc.).
   - **Trade-offs:** Schema evolution needs deliberate updates.

5. **Decision:** `CatalogLayout` + cache predicates unchanged in spirit;
   extensions `.parquet`.
   - **Rationale:** Same three-layer layout, new suffixes.
   - **New paths:**
     - `sources_{lst}_{band}.parquet`
     - `metacatalog_lst_{band}.parquet`
     - `metacatalog.parquet`

6. **Decision:** Beam backfill remains opt-in via `fits_path=` on sources read
   (FITS **images** still supply BMAJ/BMIN/BPA).
   - **Rationale:** Unchanged science need; independent of catalog format.

7. **Decision:** Legacy support is **import-only** (`import_legacy_csv_tree` /
   per-file converters), not ongoing dual-read in hot paths.
   - **Rationale:** Avoid permanently supporting two formats in cache logic.
   - **Trade-offs:** Users must run import once (or delete caches and re-detect).

8. **Decision:** Put paths in `paths.py`, Arrow schemas in `schemas.py`,
   Parquet I/O in `io.py`.
   - **Rationale:** Clear module boundaries.

**Patterns to Follow:**
- `pq.write_table(table, path)` / `pq.read_table(path)` from PyArrow Parquet docs
- `pa.Table.from_pandas(df, schema=..., preserve_index=False)` on write
- Empty catalogs: write a 0-row `Table` with full schema (not header-only CSV)
- Image FITS I/O stays on Astropy (`read_beam_from_fits`)

**Resolved defaults:**
- Dependency: `pyarrow>=14` (Parquet bundled in wheels) as a **core** dependency
- Compression: Parquet default (Snappy) unless tests show need to set
  `compression="zstd"`
- `COLOR_BANDS = ("Full", "Blue", "Green", "Red")` in `constants.py`
- Metacatalog required columns (names): `RA`, `DEC`, `Peak_flux`, `origin_band`,
  `bands_present`
- Public exports: `CatalogLayout`, `read_metacatalog`, `write_metacatalog`,
  plus layered sources/LST helpers

## Implementation Phases

### Phase 0: Dependencies and stub cleanup

**Objective:** Project builds with PyArrow; remove CSV/FITS as the intended
catalog storage API.

**Tasks:**
- [x] Add `pyarrow>=14` to `[project] dependencies` in `pyproject.toml`
- [x] Update `.gitignore` if needed (do **not** ignore `*.parquet` under tests)
- [x] Rewrite `src/lwa_catalog/io.py` surface: deprecate/remove CSV/FITS catalog
      `read_catalog`/`write_catalog` **or** repurpose them as Parquet-only
      (`read_table`/`write_table` names preferred)
- [x] Adjust `tests/test_import.py` away from CSV-only assumptions

**Dependencies:** None

**Verification:**
- [x] `pip install -e ".[dev]"` succeeds with pyarrow
- [x] `import pyarrow.parquet` works in the env

### Phase 1: Paths, Arrow schemas, and Parquet I/O API

**Objective:** Library can name, write, and read all three layers as Parquet.

**Tasks:**
- [x] Add `src/lwa_catalog/constants.py` (`COLOR_BANDS`, `ASSOC_BANDS`,
      `GAUL_COLUMNS`, required column frozensets)
- [x] Add `src/lwa_catalog/schemas.py`:
  - [x] `sources_schema() -> pa.Schema`
  - [x] `lst_merged_schema() -> pa.Schema` (minimal + extensible)
  - [x] `metacatalog_schema() -> pa.Schema`
  - [x] Helpers to cast/reorder pandas → `pa.Table` with schema
- [x] Add `src/lwa_catalog/paths.py` with `CatalogLayout`:
      `sources`, `lst_merged`, `metacatalog` → `.parquet` paths
- [x] Implement `src/lwa_catalog/io.py`:
  - [x] `write_table(table|df, path, *, schema=None)` /
        `read_table(path, *, as_pandas=True)`
  - [x] `read_beam_from_fits` / `ensure_beam_columns` (pandas or Table)
  - [x] `write_sources_catalog` / `read_sources_catalog`
  - [x] `write_lst_merged` / `read_lst_merged` / `read_all_lst_merged`
  - [x] `write_metacatalog` / `read_metacatalog` (single Parquet)
  - [x] `sources_cache_complete` / `lst_merged_cache_complete`
  - [x] `validate_metacatalog` (column names on Table or DataFrame)
- [x] Export stable names from `__init__.py`
- [x] Tests: `tests/test_paths.py`, `tests/test_schemas.py`, `tests/test_io.py`

**Dependencies:** Phase 0

**Verification:**
- [x] Filenames match `sources_{lst}_{band}.parquet`, etc.
- [x] Round-trip preserves dtypes for float RA/DEC and string `bands_present`
- [x] 0-row sources table round-trips with full schema
- [x] Beam backfill works when Parquet lacks BMAJ and `fits_path` is set

### Phase 2: Schema helpers and legacy CSV/FITS import

**Objective:** Empty-table constructors + one-way migration from notebook-era
files.

**Tasks:**
- [x] `empty_sources_table(...)` → 0-row `pa.Table` with `sources_schema()`
- [x] `validate_sources_catalog`
- [x] `import_legacy_sources_csv(path) -> pa.Table`
- [x] `import_legacy_metacatalog(csv_path=None, fits_path=None) -> pa.Table`
      (prefer CSV if both exist; FITS via Astropy → pandas → Arrow)
- [x] `migrate_output_dir(layout, *, fits_paths=...)` — convert all
      `sources_*.csv` / `metacatalog_lst_*.csv` / `metacatalog.csv|.fits` found
      under `layout.root` to `.parquet` (optional delete/rename of legacy files
      behind a flag, default keep legacy; ``dry_run=True`` lists plan only)

**Dependencies:** Phase 1

**Verification:**
- [x] Synthetic CSV fixtures import to Parquet and pass validate
- [x] Empty sources Parquet validates
- [x] Migrate dry-run lists expected outputs without requiring PyBDSF

### Phase 3: Notebook adapters

**Objective:** Notebooks read/write Parquet through the library.

**Tasks:**
- [x] `ovro_lwa_metacatalog.ipynb`: replace CSV path/load/write with
      `CatalogLayout` + Parquet helpers; point `OUTPUT_DIR` at a Parquet tree
      (document one-time `migrate_output_dir` for old caches)
- [x] `metacatalog_sky_view.ipynb`: `METACATALOG_PARQUET` (or keep variable name
      but `.parquet` path) + `read_metacatalog`
- [x] Update `notebooks/README.md` for Parquet layout and migration note
- [x] Leave detect/merge function bodies inline

**Dependencies:** Phases 1–2

**Verification:**
- [ ] Fresh detect→merge→write produces only `.parquet` catalog artifacts
- [ ] After migration, `REUSE_CACHED_CATALOGS=True` loads Parquet without
      PyBDSF
- [ ] Sky-view opens `metacatalog.parquet`
*(Manual — requires local FITS/Parquet data under `/fast/claw`.)*

### Phase 4: Polish and docs

**Objective:** README and package metadata match Arrow/Parquet I/O.

**Tasks:**
- [ ] README: layout diagram with `.parquet`, example `CatalogLayout` +
      `write_metacatalog`, link to
      [Arrow Python docs](https://arrow.apache.org/docs/python/index.html)
- [ ] Note that **image** products remain FITS; only **catalog tables** moved
- [ ] `ruff check src tests`; mark plan Complete after validate

**Dependencies:** Phase 3

**Verification:**
- [ ] `pytest` + ruff clean
- [ ] README example runs against `tmp_path` in a smoke test or doctest-style
      snippet in tests

## Success Criteria

### Automated Verification

- [ ] `pytest` passes (`python -m pytest`)
- [ ] `ruff check src tests` passes
- [ ] No catalog writer emits `.csv` or catalog `.fits` in library code
- [ ] Path methods use `.parquet` suffixes matching the layout above
- [ ] Schema round-trip tests for sources + metacatalog
- [ ] Legacy CSV import test fixture
- [ ] I/O tests do not require `bdsf`

### Manual Verification

- [ ] Migrate a real `OUTPUT_DIR` CSV tree → Parquet; spot-check row counts
- [ ] Build notebook cache reuse on Parquet tree
- [ ] Sky-view loads metacatalog Parquet and plots sources
- [ ] Optional: open `metacatalog.parquet` in pandas / DuckDB / Astropy-adjacent
      tooling as a sanity check

## Testing Strategy

**Unit Tests:**
- [ ] `tests/test_paths.py` — parquet naming
- [ ] `tests/test_schemas.py` — schema field sets vs `GAUL_COLUMNS` / metacatalog
      required names
- [ ] `tests/test_io.py` — Parquet round-trip, empty table, beam backfill, cache
      gates, validation errors
- [ ] `tests/test_legacy_import.py` — tiny CSV → Parquet

**Integration Tests:**
- [ ] Multi-file temp layout: sources + lst-merged + metacatalog write/read

**Manual Testing:**
- [ ] Real cache migration + notebook smoke

## Risk Assessment

1. **Risk:** Arrow schema too strict vs messy pandas columns from PyBDSF
   - **Likelihood/Impact:** M/M
   - **Mitigation:** Cast with nulls for missing optional GAUL fields; store
     unknown extras only if needed later; start with notebook `GAUL_COLUMNS` +
     provenance

2. **Risk:** Existing large CSV caches confuse users after format switch
   - **Likelihood/Impact:** M/M
   - **Mitigation:** Document `migrate_output_dir`; cache complete checks only
     look for `.parquet`

3. **Risk:** FITS catalog consumers break (global `metacatalog.fits`)
   - **Likelihood/Impact:** L/M
   - **Mitigation:** Out of scope export helper can be a follow-up; note in
     README

4. **Risk:** Notebook cells still call `.to_csv`
   - **Likelihood/Impact:** M/L
   - **Mitigation:** Phase 3 checklist greps notebooks for `to_csv` /
     `read_csv` / `metacatalog.fits`

## Edge Cases and Error Handling

**Edge Cases:**
1. **Case:** Empty sources catalog
   - **Expected Behavior:** 0-row Parquet with full `sources_schema`
2. **Case:** Legacy CSV without BMAJ and no `fits_path` on import/read
   - **Expected Behavior:** Import succeeds; BMAJ null until backfill
3. **Case:** Partial Parquet cache
   - **Expected Behavior:** `sources_cache_complete` False;
     `read_all_*` raises `FileNotFoundError`
4. **Case:** Both legacy CSV and new Parquet present
   - **Expected Behavior:** Hot path uses Parquet only; migrate skips or
     overwrites behind flag

**Error Scenarios:**
1. **Error:** Path suffix not `.parquet` for catalog I/O
   - **Handling:** `ValueError`
2. **Error:** Schema validation missing required fields
   - **Handling:** `ValueError` listing missing names
3. **Error:** FITS image missing BMAJ/BMIN during backfill
   - **Handling:** `ValueError` from `read_beam_from_fits`

## Open Questions

*(None — Parquet chosen as Arrow on-disk format; IPC deferred.)*

---

## References

**Research Documents:**
- [Research: Catalog Read/Write in Metacatalog Notebooks](research-catalog-io.md)

**Files Analyzed:**
- `notebooks/ovro_lwa_metacatalog.ipynb`
- `notebooks/metacatalog_sky_view.ipynb`
- `src/lwa_catalog/io.py`
- `pyproject.toml`

**External Documentation:**
- [Apache Arrow Python](https://arrow.apache.org/docs/python/index.html)
- [Reading and Writing Parquet](https://arrow.apache.org/docs/python/parquet.html)
- [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html) (considered, not
  primary)
- [CSV in PyArrow](https://arrow.apache.org/docs/python/csv.html) (legacy import
  only)

---

## Review History

### Version 1.0 — 2026-08-12
- Initial plan: CSV/FITS catalog I/O extraction

### Version 1.1 — 2026-08-12
- Migrate storage to Apache Arrow / Parquet; drop catalog CSV/FITS writes;
  add schemas module, Phase 0 dependency work, and legacy import/migration

### Version 1.2 — 2026-08-12
- Phase 0 implemented: `pyarrow` dependency; Parquet-only `read_table`/`write_table`

### Version 1.3 — 2026-08-12
- Phase 1 implemented: CatalogLayout, Arrow schemas, layered Parquet I/O

### Version 1.4 — 2026-08-12
- Phase 2 implemented: empty sources table, validate_sources, legacy CSV/FITS migrate

### Version 1.5 — 2026-08-12
- Phase 3 implemented: notebooks use Parquet I/O via lwa_catalog

