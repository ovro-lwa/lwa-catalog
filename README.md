# lwa-catalog

Tools for **creating** and **analyzing** OVRO-LWA source catalogs and
metacatalogs.

This repository starts from the metacatalog notebooks developed in
[`ovro-lwa-portal`](https://github.com/uw-ssec/ovro-lwa-portal), with shared
library code to be extracted into an installable Python package.

## Status

**Alpha.** Catalog build and sky-view workflows currently live in notebooks.
Package modules under `src/lwa_catalog/` are stubs for the APIs those notebooks
will call once logic is extracted.

## Repository layout

```text
lwa-catalog/
├── notebooks/
│   ├── ovro_lwa_metacatalog.ipynb   # PyBDSF → LST merge → band metacatalog
│   └── metacatalog_sky_view.ipynb   # Explore metacatalog.csv (SkyWidget + Bokeh)
├── src/lwa_catalog/
│   ├── create/                      # Detection + catalog fusion (stubs)
│   ├── analyze/                     # Catalog QA / analysis (stubs)
│   └── io.py                        # Read/write catalog tables (stubs)
└── tests/
```

## Workflow (target)

1. **Detect** sources in wide-field FITS images (PyBDSF).
2. **Merge** per-image catalogs across LST hours (per band), then across bands
   into a global metacatalog (one row per unique sky position).
3. **Analyze / explore** the metacatalog (cross-matches, band associations,
   interactive sky view).

## Install

```bash
# Editable install (core deps: numpy, pandas, astropy)
pip install -e .

# Optional extras
pip install -e ".[detect]"   # PyBDSF (bdsf)
pip install -e ".[viz]"      # interactive notebooks
pip install -e ".[dev]"      # pytest, ruff
pip install -e ".[all]"
```

Requires Python 3.11+.

### Notebook notes

- `ovro_lwa_metacatalog.ipynb` needs the `detect` extra (`bdsf`) and local FITS
  paths configured in the notebook.
- `metacatalog_sky_view.ipynb` currently also uses helpers from
  `ovro_lwa_portal` (HiPS URL / astrowidget WCS patch). Those dependencies will
  be narrowed or vendored as the viz path stabilizes.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Related projects

- [ovro-lwa-portal](https://github.com/uw-ssec/ovro-lwa-portal) — ingest, Zarr,
  source review UI
- [image-plane-correction](https://github.com/ovro-lwa/image-plane-correction) —
  PyBDSF helpers and image-plane QA conventions
