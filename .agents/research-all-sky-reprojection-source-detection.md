# Research: All-Sky Reprojection as Input to Source Detection

---
**Date:** 2026-08-13
**Author:** AI Assistant
**Status:** Active
**Related Documents:**
- [Research: Catalog I/O](research-catalog-io.md)
- [Plan: Catalog I/O Layer](plan-catalog-io.md)

---

## Research Question

A single OVRO-LWA image covers about half the sky. That observation is
repeated each LST hour as the sky moves overhead, so analysis of all hours
can produce a catalog of sources across the entire sky.

Two construction paths are in view:

1. **Catalog-then-merge:** run source detection on each 1-hour image, then
   merge catalogs across hours (and bands).
2. **Image-then-detect:** reproject all 1-hour images onto a single all-sky
   image, then run source detection once on that mosaic.

This research documents how PyBDSF detection and image reprojection work
today, and whether an all-sky reprojected image can be used as input to
the existing source-detection path.

## Executive Summary

The **lwa-catalog** package implements path 1 only. `detect_sources` runs
PyBDSF on one native wide-field FITS file at a time. `merge_lst_metacatalog`
then clusters those detections on the sky across LST hours, and
`build_global_metacatalog` associates color bands. The notebooks that
drive this workflow (`ovro_lwa_metacatalog.ipynb`) treat one FITS file as
one `(lst_hour, band)` slot. There is no image mosaic, regrid, or
all-sky FITS writer in `src/lwa_catalog/`.

Image reprojection lives in sibling repositories. The only multi-hour
coadd that produces a single all-sky map is **lwa-healpix**
(`coadd_fits`). That function can emit either a 1-D HEALPix array
(`nside=...`) or a 2-D array on a caller-supplied WCS
(`target_header=...`). It returns NumPy arrays; it does not write a FITS
file and does not copy `BMAJ`/`BMIN`/`BPA`. Other reprojection code
(astrowidget shader reproject, ovro-lwa-portal ingest regrid,
image-plane-correction `calcflow` resample) operates on a **single**
image or on subbands at one time step, not on a multi-hour all-sky
mosaic.

PyBDSF (`bdsf.process_image`) accepts a filename or an in-memory FITS
HDU, requires a 2-D intensity plane with standard FITS WCS and a
restoring beam, and has an optional `check_outsideuniv` flag whose
docstring names SIN all-sky images as a case where some pixels fall
outside the valid projection. `lwa_catalog.create.detect.prepare_hdu`
further requires the squeezed array to be exactly 2-D, replaces
non-finite pixels with 0, and `beam_from_header` raises if `BMAJ`/`BMIN`
are missing.

Mechanically, a **2-D FITS image** with celestial WCS and beam keywords
can be passed through `prepare_hdu` → `run_pybdsf_on_hdu` regardless of
projection (SIN, CAR, or otherwise). A **1-D HEALPix map** cannot:
`prepare_hdu` rejects `ndim != 2`. The existing `coadd_fits` 2-D path
does not itself produce that FITS+beam product; a caller would have to
write the array and header. Discovery (`parse_fits_metadata`) only
recognizes filenames that encode an LST hour, so an all-sky mosaic would
not enter the notebook discover→detect loop unless a `FitsMetadata` were
constructed by hand.

## Scope

**What This Research Covers:**
- Per-image PyBDSF detection in `lwa-catalog` (`detect.py`)
- LST-hour and band catalog merge (`merge.py`, notebooks)
- FITS discovery / slot model (`discover.py`)
- PyBDSF `process_image` input, WCS, beam, and SIN-all-sky options
- Image reprojection / coadd code in sibling repos (lwa-healpix,
  astrowidget, ovro-lwa-portal, image-plane-correction)
- Interface compatibility: which reprojection outputs satisfy
  `prepare_hdu` / `run_pybdsf_on_hdu`

**What This Research Does NOT Cover:**
- Which of the two construction paths is scientifically preferable
- Flux-scale, primary-beam, or PSF variation across a mosaic
- Performance or memory cost of all-sky PyBDSF
- Implementation of a new all-sky mosaic writer

## Key Findings

### Finding 1 — Catalog creation is per-image detect, then merge

The documented workflow in `README.md` is detect on wide-field FITS,
merge across LST hours per band, then merge across bands:

```82:89:README.md
## Workflow

1. **Detect** sources in wide-field FITS images (PyBDSF; `lwa_catalog.create`).
2. **Merge** per-image catalogs across LST hours (per band), then across bands
   into a global metacatalog (one row per unique sky position).
3. **Persist** catalogs as Parquet (`CatalogLayout` + `write_*` / `read_*`).
4. **Analyze / explore** via `notebooks/metacatalog_sky_view.ipynb` or
   `lwa_catalog.analyze`.
```

`notebooks/ovro_lwa_metacatalog.ipynb` (intro cell) states the same four
steps: Discover → Detect → LST merge → Band merge. Example notebook
output on the configured FITS tree shows 24 LST hours (`00h`–`23h`) and
four color bands (Full, Blue, Green, Red).

There is no image-level mosaic step in this path. Images remain FITS;
only catalog tables are written (`README.md:9-10`, `src/lwa_catalog/io.py:3-5`).

**Relevant Files:**
- `README.md:82-89` — workflow description
- `notebooks/ovro_lwa_metacatalog.ipynb` — discover → detect → merge
- `src/lwa_catalog/create/detect.py:127-170` — per-image detection
- `src/lwa_catalog/create/merge.py:124-144` — LST-hour catalog fusion
- `src/lwa_catalog/create/merge.py:320-362` — global band fusion

**How It Works:**
1. `discover_fits_files` walks `FITS_ROOT` and parses `(lst_hour, band)`
   from filenames (`discover.py:115-128`).
2. `discovered_slots` maps each slot to one FITS path (`discover.py:144-148`).
3. For each slot, `detect_sources` runs PyBDSF and writes
   `sources_{lst}_{band}.parquet`.
4. `merge_lst_metacatalog` concatenates all hours of one band and
   greedily clusters detections whose sky separation is within
   `max(BMAJ_i, BMAJ_cluster)` (`merge.py:69-121, 124-144`).
5. Each cluster becomes one row: median-`Peak_flux` representative,
   `n_lst_contributions`, comma-separated `lst_hours`.
6. `build_global_metacatalog` cross-matches Full → Blue → Green → Red
   with the same beam-radius association (`merge.py:320-362`).

### Finding 2 — One detection unit is one FITS file at one LST hour

`FitsMetadata` carries `path`, `lst_hour` (e.g. `"01h"`), `band`, and
optional `time_key` (`discover.py:40-47`). LST hour is parsed from the
filename or a parent directory matching `NNh` or `NNh_{suffix}`
(`discover.py:10, 37, 55-70`). Valid hours are `00h`–`23h`
(`discover.py:131-141`).

Filename patterns (`discover.py:12-37`):

| Pattern | Example | Band |
|---------|---------|------|
| Deep color | `I_01h_deep_Taper_R0_Full.fits` | Full/Blue/Green/Red |
| Portal lst-color | `Blue_I_..._20250508_LST22h_t0001.fits` | prefix |
| Band prefix + parent hour | `03h/Green_I_deep.fits` | prefix |
| Subband coadd | `18MHz_I_..._LST01h_....fits` | e.g. `18MHz` |

Unrecognized names return `None` and are skipped (`discover.py:112`).
An all-sky mosaic filename that does not encode an LST hour is not
discovered.

The metacatalog notebook glob is
`??h_[RGBF]*//*_I_deep_Taper_Robust+0.0_dewarped*fits` — one dewarped
coadd per hour-and-band directory.

**Relevant Files:**
- `src/lwa_catalog/create/discover.py:10-37` — regexes
- `src/lwa_catalog/create/discover.py:73-112` — `parse_fits_metadata`
- `src/lwa_catalog/create/discover.py:144-148` — slot map
- `tests/test_discover.py` — filename examples

### Finding 3 — `detect_sources` is projection-agnostic and image-shape-strict

Call graph:

```
FitsMetadata
  → prepare_hdu(path)          # 2-D HDU, NaN→0, RESTFREQ
  → beam_from_header(header)   # require BMAJ/BMIN
  → run_pybdsf_on_hdu(hdu)     # bdsf.process_image + gaul FITS
  → DataFrame + provenance
```

`prepare_hdu` (`detect.py:40-55`):
- Opens HDU 0 with memmap
- `np.squeeze` to drop length-1 axes
- Casts to `float32`
- Raises `ValueError` if `data.ndim != 2`
- Replaces non-finite values with `0.0`
- Copies the original header; if any of `RESTFREQ`, `RESTFRQ`,
  `CRVAL3`, `FREQ` is present, writes both `RESTFREQ` and `RESTFRQ`

`beam_from_header` (`detect.py:58-66`) requires `BMAJ` and `BMIN`;
`BPA` defaults to `0.0`.

`run_pybdsf_on_hdu` (`detect.py:84-124`) merges
`DEFAULT_BDSF_KW` → `bdsf_kw` → `**process_kw`, then always sets
`beam` from the header and calls `bdsf.process_image(hdu, **kw)`.
The Gaussian catalog is written to a temp `.gaul.fits` and read back.

`DEFAULT_BDSF_KW` (`detect.py:19-27`):

```python
{"thresh": "hard", "thresh_isl": 7.0, "thresh_pix": 4.0,
 "atrous_do": False, "psf_vary_do": False, "quiet": True, "ncores": 16}
```

`check_outsideuniv` is not set (PyBDSF default `False`).

The detection module does not read `CTYPE`, `CRVAL`, `CDELT`, or
`CRPIX`. WCS is passed through to PyBDSF unchanged. There is no
reprojection, primary-beam correction, or horizon mask in `detect.py`.

Provenance columns attached after detection (`detect.py:162-169`):
`lst_hour`, `band`, `source_file`, optional `time_key`, plus
`BMAJ`/`BMIN`/`BPA` from the header.

`run_pybdsf_on_hdu` can be called with any `PrimaryHDU` that has beam
keywords; `detect_sources` additionally requires a `FitsMetadata` and a
path that `prepare_hdu` can open.

**Relevant Files:**
- `src/lwa_catalog/create/detect.py:19-27` — defaults
- `src/lwa_catalog/create/detect.py:40-66` — HDU prepare + beam
- `src/lwa_catalog/create/detect.py:84-170` — PyBDSF run + DataFrame
- `src/lwa_catalog/constants.py:8-30` — `GAUL_COLUMNS`

**Key Patterns:**
- One FITS → one catalog DataFrame
- Header WCS is trusted, not validated
- Empty or failed PyBDSF runs return a header-only empty DataFrame

### Finding 4 — PyBDSF accepts 2-D FITS/HDU with WCS and beam

Installed package: `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/`.

`bdsf.process_image(input, **kwargs)` (`__init__.py:214-323`) accepts:
1. A FITS or CASA filename
2. A parameter save file / dict
3. An in-memory `PrimaryHDU` or `HDUList` (detected at lines 248–253)

`lwa-catalog` uses path 3 (`detect.py:104`).

WCS initialization (`readimage.py:147-170`) builds an Astropy `WCS`
from the FITS header and uses `wcs_pix2world` / `wcs_world2pix` for
celestial axes. There is no HEALPix-specific reader. Pixel ↔ sky for
island detection and Gaussian positions goes through this WCS.

Beam (`opts.py` `beam` option; `readimage.py` `init_beam`):
`(BMAJ, BMIN, BPA)` in degrees. If not passed, PyBDSF reads the header;
if still missing, it searches HISTORY for an AIPS CLEAN line, else
raises `RuntimeError`. `lwa-catalog` always passes `beam=` from the
header.

`check_outsideuniv` (`opts.py:512-523`, default `False`):

> If True, then the coordinate of each pixel is examined to check if it
> is outside the universe, which may happen when, e.g., an all sky image
> is made with SIN projection (commonly done at LOFAR earlier). When
> found, these pixels are blanked.

`preprocess.py:72-80` runs that check only when the flag is True.
`DEFAULT_BDSF_KW` does not enable it. `trim_box` (`opts.py:524-528`)
can restrict detection to a pixel window.

PyBDSF operates on a 2-D intensity plane (or a cube with Stokes/freq
axes that it reduces to ch0). A 1-D HEALPix vector is not a valid
input. The wavelet path has special handling when `max(shape) > 4096`
(`wavelet_atrous.py`); `atrous_do` is `False` in this repo.

**Relevant Files:**
- `bdsf/__init__.py:214-323` — `process_image` inputs
- `bdsf/readimage.py:147-170` — `init_wcs`
- `bdsf/opts.py:512-528` — `check_outsideuniv`, `trim_box`
- `bdsf/preprocess.py:72-80` — outside-universe blanking

### Finding 5 — No all-sky mosaic writer exists in lwa-catalog

Grep of `src/lwa_catalog/` finds no `reproject`, `healpix`, `mosaic`, or
`regrid` implementation.

The only reprojection in this repo is display-only, in
`notebooks/metacatalog_sky_view.ipynb`: `_reproject_fits_for_shader`
wraps `astrowidget.wcs.build_reproject_maps` and
`scipy.ndimage.map_coordinates` to overlay one Full-band FITS on the
HiPS view after pan/zoom. It produces an in-memory `float32` array and
a naive SIN WCS. It does not write FITS and does not copy beam
keywords.

### Finding 6 — Multi-hour all-sky coadd lives in lwa-healpix

`/home/claw/code/lwa-healpix` is the sibling package that reprojects
and coadds OVRO-LWA FITS images.

`coadd_fits` (`src/lwa_healpix/coadd.py:313-454`):
- Input: list of FITS paths (4-D RA/Dec/Freq/Stokes or 2-D with
  vestigial freq keywords; `_extract_2d` pulls the spatial plane)
- Exactly one of `nside` or `target_header` must be set
- Optional `min_elevation` blanks pixels whose elevation (90° minus
  separation from `CRVAL`, treated as zenith) is below the cut
  (`utils.py:24-38`)
- Optional quality screening on a central patch before reproject
  (`screen_fits_by_quality`, `coadd.py:60-117`); `one_per_lst_hour`
  keeps the lowest-metric file per LST hour
- Per file: `reproject_to_healpix` or `reproject_interp` onto the
  target; `np.nan_to_num(..., nan=0.0)`; footprint-weighted accumulate
- Returns `(combined, total_weight)` arrays — **no FITS write**

Output modes:

| Argument | Output | Shape | FITS? | Beam copied? |
|----------|--------|-------|-------|--------------|
| `nside=...` | HEALPix map | 1-D, `12*nside²` | No | N/A |
| `target_header=...` | Image on that WCS | 2-D `(NAXIS2, NAXIS1)` | No | No |

`combine_fits_to_spectral_cube` (`coadd.py:457-664`) writes a 3-D FITS
cube. It copies `TELESCOP`, `INSTRUME`, `OBSERVER`, `OBJECT`,
`DATE-OBS`, `BUNIT`, `EQUINOX`, `RADESYS` from the reference header
(`coadd.py:649-653`) and **does not copy `BMAJ`/`BMIN`/`BPA`**. If the
frequency axis has length > 1, `prepare_hdu` would reject the cube
(`ndim != 2` after squeeze).

HiPS path (`src/lwa_healpix/hips.py`):
- `DEFAULT_CAR_HEADER` (`hips.py:46-63`): full-sky Plate Carrée,
  3600×1800, 0.1°/pixel, `GLON-CAR`/`GLAT-CAR`
- `_car_header_for_nside` (`hips.py:66-95`) scales CAR to HEALPix NSIDE
- `healpix_to_hips` / `fits_to_hips` write HiPS tiles, not analysis FITS

The package README “Future directions” (`README.md:109-116`) names
all-sky maps by coadding deep images across LST as a planned use of
`coadd_fits`, not as a completed catalog-from-mosaic pipeline.

**Relevant Files:**
- `/home/claw/code/lwa-healpix/src/lwa_healpix/coadd.py:313-454`
- `/home/claw/code/lwa-healpix/src/lwa_healpix/utils.py:24-38`
- `/home/claw/code/lwa-healpix/src/lwa_healpix/hips.py:46-95`
- `/home/claw/code/lwa-healpix/README.md:15-20, 109-116`

### Finding 7 — Other reprojection is single-image or ingest, not all-sky mosaic

| Location | Function | Multi-hour mosaic? | Writes FITS? | Role |
|----------|----------|--------------------|--------------|------|
| `astrowidget` `wcs.py` `reproject_for_shader_display` | Naive SIN onto view center; far-hemisphere mask | No | No | WebGL / HiPS overlay |
| `ovro-lwa-portal` `fits_to_zarr_xradio.py` `_reproject_celestial_plane` / `_resample_lm_reference_to_target_size` | `reproject_interp` onto scaled SIN grid | No (subbands at one time) | Via `export_fits` | Ingest |
| `ovro-lwa-portal` `export_fits.py` | Pixel-faithful 4-D FITS; preserves `BMAJ`/`BMIN`/`BPA` | No | Yes | PyBDSF round-trip of a Zarr slice |
| `image-plane-correction` `flow.py` `calcflow` | Optional `target_size` resample before dewarp | No | `{basename}_dewarp.fits` | Dewarp; header merged from input (beam typically preserved) |

Astrowidget SIN handling (`astrowidget/src/astrowidget/wcs.py`):
`near_hemisphere` (`cos_sep ≤ 0`) blanks the far side of the SIN
two-to-one mapping. That is display logic; it is not used by
`lwa-catalog` detection.

### Finding 8 — Interface compatibility: what can feed PyBDSF today

The existing detection entry points impose these constraints:

| Constraint | Where | Effect on an all-sky product |
|------------|-------|------------------------------|
| Squeezed data must be 2-D | `prepare_hdu` `detect.py:46-48` | HEALPix 1-D maps and multi-channel cubes fail |
| `BMAJ` and `BMIN` required | `beam_from_header` `detect.py:60-61` | `coadd_fits` / `combine_fits_to_spectral_cube` outputs lack these unless a caller adds them |
| Non-finite → 0 | `prepare_hdu` `detect.py:49` | Unobserved / horizon-blanked mosaic pixels become zero flux (not NaN/mask) |
| Standard FITS WCS on axes 1–2 | PyBDSF `init_wcs` | 2-D CAR or SIN FITS is readable; HEALPix is not |
| Filename encodes LST hour | `parse_fits_metadata` `discover.py:73-112` | All-sky mosaic names are skipped by discover unless `FitsMetadata` is built by hand |
| `check_outsideuniv` off | `DEFAULT_BDSF_KW` | SIN all-sky pixels outside the projection volume are not blanked by PyBDSF |
| `psf_vary_do` False | `DEFAULT_BDSF_KW` | Single beam from header is used for the whole image |

Compatibility of existing reprojection outputs:

| Product | Can `prepare_hdu` read it? | Can `run_pybdsf_on_hdu` run? |
|---------|----------------------------|------------------------------|
| Native hourly pipeline FITS (current path) | Yes | Yes — this is the implemented path |
| Portal `export_fits` / `build_fits_hdu` of one slice | Yes (squeeze 4-D singletons) | Yes if beam in header |
| image-plane-correction `_dewarp.fits` | Yes | Yes if beam preserved |
| `coadd_fits(..., nside=...)` 1-D HEALPix | No (`ndim != 2`) | No |
| `coadd_fits(..., target_header=...)` 2-D array | Not a file; no writer | Only after a caller writes 2-D FITS with WCS **and** `BMAJ`/`BMIN` |
| `combine_fits_to_spectral_cube` 3-D FITS | No if `nfreq > 1` | No (and no beam cards) |
| HiPS tiles / CAR intermediate in `hips.py` | No analysis FITS | No |
| astrowidget / sky-view shader reproject | No file | No (no beam, display WCS) |

`run_pybdsf_on_hdu` does not depend on LST-hour metadata. A hand-built
`PrimaryHDU` with 2-D data, celestial WCS, and beam keywords is the
minimum that the PyBDSF wrapper accepts. `detect_sources` additionally
needs a path and `FitsMetadata` for provenance.

## Architecture Overview

```
Current catalog path (lwa-catalog)
─────────────────────────────────
FITS_ROOT / ??h_{Band} / I_*_dewarped*.fits     (one ~half-sky SIN image per hour)
        │
        ▼ discover_fits_files / discovered_slots
   (lst_hour, band) → one FITS
        │
        ▼ detect_sources
   prepare_hdu (2-D, NaN→0) → bdsf.process_image → gaul table
        │
        ▼ sources_{NNh}_{Band}.parquet
        │
        ▼ merge_lst_metacatalog   (sky clustering, median flux)
   metacatalog_lst_{Band}.parquet
        │
        ▼ build_global_metacatalog  (Full→Blue→Green→Red)
   metacatalog.parquet              (one row per unique sky position)


Sibling image-coadd path (lwa-healpix) — not wired to detect
────────────────────────────────────────────────────────────
list of hourly FITS
        │
        ▼ coadd_fits
   ┌────┴────┐
   nside=    target_header=
   1-D HPX   2-D ndarray + weight
   (not      (no FITS write,
    2-D)      no BMAJ/BMIN)
        │
        ▼ hips.py (optional)
   CAR / HiPS tiles  (viewer, not PyBDSF input)


Display reproject (astrowidget / sky-view notebook)
───────────────────────────────────────────────────
one FITS slice → naive SIN at view center → WebGL  (not written, not detected)
```

## Component Interactions

**Flow Description:**

1. **Discovery** (`discover.py:73-128`) — filename/path → `FitsMetadata`.
   LST hour is required. An all-sky mosaic without `LSTnnh` / `NNh` in
   the path is not a discovered slot.

2. **HDU prepare** (`detect.py:40-55`) — any FITS path with a 2-D
   (after squeeze) primary array is accepted. Projection type is not
   inspected. Non-finite pixels become 0.

3. **PyBDSF** (`detect.py:84-104`, `bdsf/__init__.py:214-323`) — reads
   the HDU, initializes WCS from the header, uses the supplied beam,
   writes a Gaussian catalog. RA/DEC in the catalog come from that WCS.

4. **LST merge** (`merge.py:124-144`) — sky-coordinate clustering of
   per-hour catalogs. This is how half-sky hourly detections become an
   all-sky catalog today. Image pixels are not combined.

5. **Band merge** (`merge.py:320-362`) — sequential cross-match of
   LST-merged Full/Blue/Green/Red catalogs.

6. **lwa-healpix coadd** (`coadd.py:313-454`) — independent of
   `lwa_catalog.create`. Same class of hourly FITS can be reprojected
   onto HEALPix or a 2-D target WCS. Output is arrays, not a detection
   input.

## Code Examples

```python
# src/lwa_catalog/create/detect.py:40-55
def prepare_hdu(path: Path) -> fits.PrimaryHDU:
    """Read FITS, squeeze to 2D, sanitize NaNs, and fix header for PyBDSF."""
    ...
    data = np.squeeze(np.asarray(hdu.data, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(...)
    data = np.where(np.isfinite(data), data, 0.0)
```

```python
# src/lwa_catalog/create/merge.py:124-140
def merge_lst_metacatalog(catalogs, *, band):
    """Fuse per-LST detections within one band → one row per source."""
    combined = pd.concat(frames, ignore_index=True)
    for members in _cluster_by_sky_position(combined):
        rep = _pick_median_flux_row(members)
        entry["n_lst_contributions"] = len(members)
        entry["lst_hours"] = ",".join(sorted(members["lst_hour"].unique()))
```

```python
# /home/claw/code/lwa-healpix/src/lwa_healpix/coadd.py:334-338
# Exactly one of *nside* or *target_header* must be given:
# * nside — reproject onto a HEALPix grid.  Returns a 1-D map.
# * target_header — reproject onto a 2-D image grid.  Returns a 2-D array.
```

```python
# bdsf/opts.py:512-522
check_outsideuniv = Bool(False,
    doc="... an all sky image is made with SIN projection "
        "(commonly done at LOFAR earlier). When found, these pixels "
        "are blanked ...")
```

## Technical Decisions

- **Decision:** Catalog-level fusion across LST hours, not image mosaicking
  - **Where:** `merge.py`, `README.md` workflow, metacatalog notebook
  - **What this does:** Each hourly FITS is detected in its native WCS
    (typically SIN, zenith-centered). Overlap is resolved by clustering
    RA/DEC within a beam radius. `n_lst_contributions` records how many
    hours saw a source.

- **Decision:** Detection wrapper is one-image, header-driven
  - **Where:** `detect.py` `prepare_hdu` / `beam_from_header`
  - **What this does:** Any 2-D FITS with `BMAJ`/`BMIN` can be processed.
    Projection is delegated to PyBDSF. NaNs become zeros.

- **Decision:** Reprojection for viewing is separate from detection
  - **Where:** astrowidget; `metacatalog_sky_view.ipynb`
  - **What this does:** Shader/HiPS overlays reproject a single slice
    onto a view-centered SIN grid. Catalogs are plotted as RA/DEC
    points, not re-detected on the overlay.

- **Decision:** lwa-healpix coadd returns arrays
  - **Where:** `coadd_fits` return `(combined, total_weight)`
  - **What this does:** Multi-hour all-sky maps exist as in-memory
    HEALPix or 2-D arrays. FITS+beam packaging for PyBDSF is not part
    of that function.

## Dependencies and Integrations

- **PyBDSF (`bdsf`)** — optional extra `lwa-catalog[detect]`
  (`pyproject.toml`). `process_image` + `write_catalog(catalog_type="gaul")`.
- **Astropy** — FITS I/O, `SkyCoord` clustering, WCS (inside PyBDSF).
- **lwa-healpix** — sibling; `reproject` library (`reproject_interp`,
  `reproject_to_healpix`, `reproject_from_healpix`). Not imported by
  `lwa-catalog`.
- **astrowidget** — display reproject for SkyWidget / sky-view notebook.
- **ovro-lwa-portal** — ingest regrid and `export_fits`; historical
  source of metacatalog prototypes (`README.md:6-7`).
- **image-plane-correction** — dewarp FITS that the metacatalog notebook
  glob already selects (`*_dewarped*fits`).
- **reproject** (Astropy affiliated) — used by lwa-healpix and portal
  ingest, not by `lwa_catalog.create`.

## Edge Cases and Constraints

- **HEALPix maps are 1-D.** `prepare_hdu` requires `ndim == 2`. The
  `nside` output of `coadd_fits` cannot be passed to `detect_sources`.

- **2-D coadd has no FITS or beam.** `coadd_fits(..., target_header=...)`
  returns arrays only. `beam_from_header` would fail on a header that
  only has WCS cards.

- **Spectral cubes.** `combine_fits_to_spectral_cube` writes 3-D FITS
  without beam keywords. Squeeze leaves `ndim == 3` when `nfreq > 1`.

- **NaN → 0.** Horizon blanks and unobserved mosaic pixels become zero
  intensity before PyBDSF. PyBDSF’s own blanking uses NaN/`check_outsideuniv`,
  which this prepare step removes.

- **SIN all-sky invalid pixels.** PyBDSF documents this via
  `check_outsideuniv`. The flag is off in `DEFAULT_BDSF_KW`. A single
  SIN projection cannot represent the full sphere without a region
  outside the projection volume; CAR/HEALPix avoid that geometry.

- **Discovery requires LST hour.** An all-sky mosaic is not a
  `(lst_hour, band)` slot. `detect_sources` can still be called with a
  constructed `FitsMetadata`; the notebook loop will not find the file.

- **Single beam for the whole image.** `psf_vary_do` is False. Hourly
  images each carry their own `BMAJ`; a mosaic would present one beam
  to PyBDSF.

- **Elevation mask assumes CRVAL is zenith.** `lwa_healpix.utils._pixel_elevations`
  uses that assumption. It applies to native hourly SIN images, not to
  a reprojected all-sky CAR/HEALPix grid (where CRVAL is not zenith).

- **CAR default scale is 0.1°/pixel** (`DEFAULT_CAR_HEADER`). That
  header is for HiPS, not for source detection. A detection-oriented
  `target_header` would be caller-defined.

- **Missing hours are skipped**, not filled (`merge.py` / notebook loop
  over discovered slots only). Catalog merge does not require all 24
  hours.

## Open Questions

1. If a 2-D all-sky FITS were written from `coadd_fits(..., target_header=...)`,
   which WCS (CAR, ZEA, a large SIN, other) and pixel scale would be
   used, and how would `BMAJ`/`BMIN` be defined for the mosaic?

2. Would an all-sky detection path call `run_pybdsf_on_hdu` directly
   (bypassing discover/`FitsMetadata`), or would discovery be extended
   for mosaic filenames?

3. Would `check_outsideuniv` be enabled if the mosaic stayed in SIN?

4. How would `n_lst_contributions` / `lst_hours` be represented if
   detection ran once on a coadd instead of per hour?

These are product/design questions; the current code does not answer
them.

## References

- Files analyzed: 20+
  - `src/lwa_catalog/create/detect.py`
  - `src/lwa_catalog/create/discover.py`
  - `src/lwa_catalog/create/merge.py`
  - `src/lwa_catalog/create/__init__.py`
  - `src/lwa_catalog/constants.py`
  - `src/lwa_catalog/paths.py`
  - `src/lwa_catalog/schemas.py`
  - `src/lwa_catalog/io.py`
  - `README.md`
  - `notebooks/README.md`
  - `notebooks/ovro_lwa_metacatalog.ipynb`
  - `notebooks/ovro_lwa_subband_detect.ipynb`
  - `notebooks/metacatalog_sky_view.ipynb`
  - `tests/test_discover.py`
  - `tests/test_merge.py`
  - `/home/claw/code/lwa-healpix/src/lwa_healpix/coadd.py`
  - `/home/claw/code/lwa-healpix/src/lwa_healpix/hips.py`
  - `/home/claw/code/lwa-healpix/src/lwa_healpix/utils.py`
  - `/home/claw/code/lwa-healpix/README.md`
  - `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/__init__.py`
  - `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/opts.py`
  - `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/readimage.py`
  - `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/preprocess.py`

- Related documentation:
  - [Research: Catalog I/O](research-catalog-io.md)
  - lwa-healpix README “Future directions” (all-sky coadd across LST)
  - PyBDSF `check_outsideuniv` option docstring (SIN all-sky pixels)

---

## Follow-up Research [2026-08-13 13:19]

**Question:** How does PyBDSF work on a mosaic made from many hour images? After mosaicking, a 2-D image HDU and header can be created with lwa-healpix, and `BMAJ`/`BMIN` can be calculated if missing. Will PyBDSF interpret the projection properly?

**Findings:**

PyBDSF has no mosaic-specific code path. A multi-hour coadd is treated as one 2-D FITS image. Detection and Gaussian fitting run in **pixel coordinates**. The FITS WCS (including the projection coded in `CTYPE`) is used only when converting fitted pixel positions and sizes to sky. That conversion is delegated to **Astropy WCS**, not to a PyBDSF-owned SIN/CAR implementation.

### How a 2-D mosaic HDU is obtained from lwa-healpix

`coadd_fits(..., target_header=...)` (`/home/claw/code/lwa-healpix/src/lwa_healpix/coadd.py:313-454`) reprojects each hourly FITS onto the caller’s 2-D WCS and returns `(combined, total_weight)` arrays. It does not construct an HDU. A 2-D product is assembled by the caller as `fits.PrimaryHDU(data=combined, header=target_header)` (or an equivalent header built from that WCS).

Existing headers in lwa-healpix that can serve as `target_header`:

| Source | CTYPE | Grid | CDELT present? |
|--------|-------|------|----------------|
| `tests/test_coadd.py:276-291` `_image_target_header` | `RA---SIN` / `DEC--SIN` | caller `nx`×`ny` | Yes |
| `hips.py:46-63` `DEFAULT_CAR_HEADER` | `GLON-CAR` / `GLAT-CAR` | 3600×1800 | Yes (−0.1°, +0.1°) |
| `hips.py:66-95` `_car_header_for_nside` | `GLON-CAR`/`GLAT-CAR` or `RA---CAR`/`DEC--CAR` | full sky, scale from NSIDE | Yes |

`combine_fits_to_spectral_cube` (`coadd.py:570-655`) writes a **3-D** FITS whose spatial WCS is `ref_wcs_2d.to_header()` from the first input file (typically native SIN, not an all-sky CAR). That path is a spectral cube, not the 2-D mosaic HDU described in the question.

`coadd_fits` does not copy `BMAJ`/`BMIN`/`BPA` or frequency cards. Those must be added on the HDU before `beam_from_header` / `init_freq` run.

`_reproject_healpix_to_car` (`hips.py:98-112`) is the other 2-D array+header pair: a HEALPix map resampled onto CAR. Same packaging step (caller builds `PrimaryHDU`).

### What PyBDSF does with that HDU (independent of how many hours were coadded)

Once `bdsf.process_image(hdu)` receives the HDU (`bdsf/__init__.py:248-317`):

1. **Read and axis order** (`functions.py:1252-1320`). `CTYPE` is split on `-`; only the coordinate name is kept (`RA---SIN` → `RA`, `RA---CAR` → `RA`, `GLON-CAR` → `GLON`). The image is accepted if axes include **RA+DEC or GLON+GLAT**. The projection code (`SIN`, `CAR`, `TAN`, …) is not interpreted here. Data are reshaped to `[STOKES, FREQ, x, y]`.

2. **WCS object** (`readimage.py:147-170`). `astropy.wcs.WCS(hdr)` + `wcs.fix()`. This is where the projection in the full `CTYPE` is applied. `p2s` / `s2p` wrap `wcs_pix2world` / `wcs_world2pix` on the first two world coordinates (`readimage.py:176-206`).

3. **Pixel-scale constants** (`readimage.py:171`). `acdelt = [abs(hdr['cdelt1']), abs(hdr['cdelt2'])]`. `CDELT1` and `CDELT2` must exist as header keywords (a CD/PC-only header without `CDELT` raises `KeyError`).

4. **Beam** (`readimage.py:309-400`). `opts.beam` or header `BMAJ`/`BMIN`/`BPA`. Converted to pixels at the **image center** with constant `CDELT`: `s1 = abs(bmaj / cdelt1)` (`readimage.py:319-330`). This conversion does not use local projection scale.

5. **Frequency** (`readimage.py:402-445`). Spectral WCS axis, or `RESTFREQ` / `FREQ`, or `opts.frequency`. A 2-D CAR header from `hips.py` has none of these unless the caller adds them. `lwa_catalog.create.detect.prepare_hdu` copies `RESTFREQ`/`RESTFRQ` from `RESTFREQ`, `RESTFRQ`, `CRVAL3`, or `FREQ` when present (`detect.py:30-37, 58-61`).

6. **Island find + Gaussian fit** (`gausfit.py`). Peak, pixel center, and pixel FWHM/PA. No WCS.

7. **Sky conversion after a successful fit** (`gausfit.py:1054-1064`):
   - `centre_sky = img.pix2sky(pixel_xy)` → Astropy world coordinates
   - `size_sky = img.pix2gaus(..., use_wcs=True)` → local angular size at the source
   - `size_sky_uncorr = img.pix2gaus(..., use_wcs=False)` → `CDELT` at image center only

8. **FITS `gaul` catalog** (`gausfit.py:971-984`). `centre_sky` is written as columns **`RA`, `DEC`**. `size_sky` is written as **`Maj`, `Min`, `PA`**. `lwa-catalog` keeps those columns (`constants.py:8-30`, `detect.py` `write_catalog(..., catalog_type="gaul")`).

There is no branch that knows the image is a mosaic of many LST hours.

### How projection enters positions vs sizes

**Positions.** `pix2sky` is Astropy `wcs_pix2world` (`readimage.py:176-190`). For `RA---SIN`, `RA---CAR`, `DEC--TAN`, etc., the world values are equatorial RA/Dec in degrees. Those values are what appear in the `gaul` `RA`/`DEC` columns. The projection in `CTYPE` is applied by Astropy for every source.

**Sizes (`use_wcs=True`).** `pix2gaus` → `pixdist2angdist` (`readimage.py:233-259, 585-605`):

1. Take two pixel points along the Gaussian PA, centered on the source.
2. Convert each to world via `pix2sky`.
3. Call `func.angsep` (`functions.py:495-518`), a spherical separation on the two world numbers treated as longitude/latitude in degrees.
4. That local degree-per-pixel scale converts FWHM from pixels to degrees. PA is offset by `get_rot` (angle between +y and north at that pixel), also via `pix2sky`/`sky2pix` (`readimage.py:531-553`).

The comment at `readimage.py:208-211` states these transforms are valid **only at the Gaussian’s center** and ignore change across the Gaussian.

**Sizes (`use_wcs=False`) and beam.** `beam2pix` / `pix2beam` (`readimage.py:319-347`) use a single `CDELT` pair. Flux is `peak * size_pix / beam_pix` (`gausfit.py:1050`). Deconvolution uses that same center-scale beam in pixels (`gausfit.py:1060-1061`).

**`correct_proj`** (`opts.py:1299-1310`, default `True`) applies only to **BBS-format** catalogs (`output.py:572-575`). The FITS `gaul` path used by `lwa-catalog` writes `size_sky` (WCS-local) regardless of this flag.

### Will PyBDSF interpret the projection?

What the code does, by header type:

| Mosaic header | Axis acceptance (`functions.py:1271-1277`) | `pix2sky` world system | `gaul` column names | Local size via `pix2gaus` |
|---------------|--------------------------------------------|------------------------|---------------------|---------------------------|
| `RA---SIN` / `DEC--SIN` | RA+DEC | Equatorial (Astropy SIN) | `RA`, `DEC` | Yes, at source pixel |
| `RA---CAR` / `DEC--CAR` (`_car_header_for_nside(..., coord_frame!="galactic")`) | RA+DEC | Equatorial (Astropy CAR) | `RA`, `DEC` | Yes, at source pixel |
| `GLON-CAR` / `GLAT-CAR` (`DEFAULT_CAR_HEADER`, default `_car_header_for_nside`) | GLON+GLAT | Galactic lon/lat | still `RA`, `DEC` | `angsep` on GLON/GLAT as spherical lon/lat |

PyBDSF does not implement SIN or CAR itself. It constructs `WCS(header)` and calls `wcs_pix2world`. Any projection Astropy accepts in a standard 2-D FITS WCS is used for positions. The projection string after `-` in `CTYPE` is ignored by PyBDSF’s axis-ordering logic and is consumed only by Astropy.

Two header requirements are independent of projection:

- **`CDELT1` / `CDELT2`** must be present (`readimage.py:171`). lwa-healpix CAR helpers and the SIN test header include them. `WCS.to_header()` from some Astropy versions can emit a PC matrix without `CDELT`; that form does not satisfy this read.
- **Celestial axis names** must be RA/DEC or GLON/GLAT. Other systems raise `RuntimeError("Image data not found")` (`functions.py:1271-1273`).

### SIN all-sky vs CAR all-sky inside PyBDSF

`check_outsideuniv` (`preprocess.py:154-176`, default **off**): for every pixel, `pix2sky` then `sky2pix`; if the round-trip differs by more than 0.5 pixel, the pixel is set to NaN. The option docstring names SIN all-sky images (`opts.py:512-522`). CAR full-sky grids typically round-trip. `DEFAULT_BDSF_KW` does not set this flag.

`prepare_hdu` as of this follow-up (`detect.py:40-62`) converts non-finite values to **NaN** (not 0) so PyBDSF can blank them. Unobserved mosaic pixels left as NaN stay blanked; zeros would be treated as data.

### Header cards a mosaicked 2-D HDU needs for the existing detect wrapper

For `prepare_hdu` → `run_pybdsf_on_hdu` on a caller-built mosaic HDU:

| Card / property | Required by | Present on lwa-healpix CAR/SIN `target_header`? |
|-----------------|-------------|--------------------------------------------------|
| 2-D data | `prepare_hdu` `detect.py:51-53` | Yes, from `coadd_fits` array |
| `CTYPE1`/`CTYPE2` RA/DEC or GLON/GLAT + projection | PyBDSF `functions.py:1266-1277` + Astropy WCS | Yes on the headers listed above |
| `CDELT1`/`CDELT2` | `readimage.py:171` | Yes on those helpers |
| `CRVAL`, `CRPIX`, `CUNIT` | Astropy WCS | Yes |
| `BMAJ`, `BMIN` (optional `BPA`) | `beam_from_header` `detect.py:67-68`; PyBDSF `init_beam` | No — caller-calculated, as stated in the question |
| Frequency (`RESTFREQ` / `RESTFRQ` / `FREQ` / spectral axis) | PyBDSF `init_freq` `readimage.py:434-445` | No on CAR helpers |
| Equinox / `RADESYS` | `get_equinox` `readimage.py:493-529`; default J2000 if missing | Optional |

`run_pybdsf_on_hdu` can take that in-memory `PrimaryHDU` directly (`detect.py` / `bdsf/__init__.py:248-253`). `detect_sources` additionally needs a path and `FitsMetadata`.

### Pixel-space vs sky-space on a varying-scale mosaic

On a CAR all-sky grid, pixel scale in true angle changes with latitude. PyBDSF:

- Fits Gaussians in **pixels** (constant pixel beam from center `CDELT`).
- Reports **positions** through the full projection.
- Reports **Maj/Min/PA** with a first-order local scale at the source center (`pix2gaus`).
- Converts the restoring beam to pixels with **center `CDELT` only** (`beam2pix`), and uses that for flux and deconvolution.

That is the same mechanism used on a single wide-field SIN snapshot; a mosaic does not change it.

### Note on Finding 3 (NaN handling)

The original Finding 3 text described `prepare_hdu` as replacing non-finite pixels with `0.0`. The current function (`detect.py:40-62`) writes `NaN` instead, so PyBDSF can blank those pixels. That matters for mosaic regions with no contributing hours (`coadd_fits` leaves `total_weight == 0` as 0 in `combined`; a caller who writes NaN where weight is 0 would be blanked).

**Files read for this follow-up:**
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/readimage.py` (full)
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/functions.py:495-518, 1252-1397`
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/gausfit.py:971-1064`
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/opts.py:512-528, 1299-1310`
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/preprocess.py:72-80, 154-176`
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/output.py:572-575`
- `/opt/devel/claw/envs/py312/lib/python3.12/site-packages/bdsf/__init__.py:214-323`
- `/home/claw/code/lwa-healpix/src/lwa_healpix/coadd.py:313-454, 530-664`
- `/home/claw/code/lwa-healpix/src/lwa_healpix/hips.py:46-112`
- `/home/claw/code/lwa-healpix/tests/test_coadd.py:276-291`
- `src/lwa_catalog/create/detect.py:30-62` (current NaN-preserving `prepare_hdu`)
