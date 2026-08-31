# lwa-catalog

Tools for **creating** and **analyzing** OVRO-LWA source catalogs and
metacatalogs.

Shared library code extracted from metacatalog workflows formerly prototyped in
[`ovro-lwa-portal`](https://github.com/uw-ssec/ovro-lwa-portal). Catalog
**tables** are stored as [Apache Parquet](https://arrow.apache.org/docs/python/parquet.html)
via [PyArrow](https://arrow.apache.org/docs/python/index.html). Wide-field
**images** remain FITS.

## Status

**Alpha.** Detection, merge, and Parquet I/O live under `src/lwa_catalog/`.
Build and sky-view workflows live in `notebooks/`.

## Catalog layout (Parquet)

Under an output directory (`CatalogLayout.root`):

```text
OUTPUT_DIR/
├── sources_{lst}_{band}.parquet      # per-image PyBDSF catalogs
├── metacatalog_lst_{band}.parquet    # LST-merged per band
└── metacatalog.parquet               # global metacatalog
```

| Layer | Filename pattern |
| ----- | ---------------- |
| Per-image sources | `sources_{lst}_{band}.parquet` |
| LST-merged band | `metacatalog_lst_{band}.parquet` |
| Global metacatalog | `metacatalog.parquet` |

Bands are typically `Full`, `Blue`, `Green`, `Red`. Image products stay FITS;
only catalog tables use Parquet (no dual CSV/FITS catalog writes).

### Quick example

```python
from pathlib import Path

import pandas as pd

from lwa_catalog import CatalogLayout, read_metacatalog, write_metacatalog

layout = CatalogLayout(Path("catalog_out"))
df = pd.DataFrame(
    {
        "RA": [10.0],
        "DEC": [20.0],
        "Peak_flux": [1.0],
        "origin_band": ["Full"],
        "bands_present": ["Full"],
    }
)
path = write_metacatalog(df, layout)
loaded = read_metacatalog(layout)
assert path.name == "metacatalog.parquet"
assert len(loaded) == 1
```

Legacy CSV/FITS catalog trees can be converted once with
`migrate_output_dir(layout)` (keeps legacy files by default).

## Repository layout

```text
lwa-catalog/
├── notebooks/
│   ├── ovro_lwa_metacatalog.ipynb           # discover → detect → merge → Parquet
│   ├── ovro_lwa_metacatalog_subband.ipynb   # same pipeline on 15 frequency subbands
│   └── metacatalog_sky_view.ipynb           # explore metacatalog.parquet
├── src/lwa_catalog/
│   ├── create/                      # discover, PyBDSF detect, merge
│   ├── analyze/                     # catalog QA / summary helpers
│   ├── paths.py                     # CatalogLayout (.parquet paths)
│   ├── schemas.py                   # Arrow schemas per layer
│   ├── constants.py                 # band / column name constants
│   └── io.py                        # Parquet read/write + legacy migrate
└── tests/
```

## Workflow

1. **Detect** sources in wide-field FITS images (PyBDSF; `lwa_catalog.create`).
2. **Merge** per-image catalogs across LST hours (per band), then across bands
   into a global metacatalog (one row per unique sky position).
3. **Persist** catalogs as Parquet (`CatalogLayout` + `write_*` / `read_*`).
4. **Analyze / explore** via `notebooks/metacatalog_sky_view.ipynb` or
   `lwa_catalog.analyze`.

## Install

```bash
pip install -e .            # core: numpy, pandas, astropy, pyarrow
pip install -e ".[detect]"  # PyBDSF (bdsf)
pip install -e ".[viz]"     # interactive notebooks
pip install -e ".[dev]"     # pytest, ruff
pip install -e ".[all]"
```

Requires Python 3.11+.

### Notebook notes

- `ovro_lwa_metacatalog.ipynb` needs the `detect` extra (`bdsf`) and local FITS
  paths configured in the notebook. Catalog I/O uses Parquet under `OUTPUT_DIR`.
- `ovro_lwa_metacatalog_subband.ipynb` is the same detect → LST-merge → global
  fusion flow on **15 frequency-labeled subbands** (18–82 MHz). The lowest
  subband seeds sequential association; adjacent-channel spectral indices use
  those MHz centers. Detection parameters match `ovro_lwa_metacatalog.ipynb`.
- `metacatalog_sky_view.ipynb` loads `metacatalog.parquet` via
  `read_metacatalog`. It may also use helpers from `ovro-lwa-portal` (HiPS URL /
  astrowidget WCS patch) until those pieces stabilize here.

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
- [Apache Arrow Python](https://arrow.apache.org/docs/python/index.html) —
  in-memory tables and Parquet I/O used by this package
