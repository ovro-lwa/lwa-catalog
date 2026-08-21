# Notebooks

| Notebook | Role |
| -------- | ---- |
| `ovro_lwa_metacatalog.ipynb` | Discover FITS → PyBDSF → LST merge → global metacatalog (Parquet) |
| `ovro_lwa_mosaic_detect.ipynb` | Coadd hourly Full-band FITS to one SIN mosaic → PyBDSF → compare to LST-merged catalog |
| `metacatalog_sky_view.ipynb` | Load `metacatalog.parquet` and explore on the sky |
| `metacatalog_query.ipynb` | Browse all Parquet catalogs, sortable table, nearest-source coordinate query |

## Catalog storage

Catalogs are written and read as **Apache Parquet** through `lwa_catalog`:

| Artifact | Path under `OUTPUT_DIR` |
| -------- | ----------------------- |
| Per-image sources | `sources_{lst}_{band}.parquet` |
| LST-merged band | `metacatalog_lst_{band}.parquet` |
| Global metacatalog | `metacatalog.parquet` |

Image products remain FITS. Detection and merge live in `lwa_catalog.create`;
notebooks keep configuration constants and call library APIs for I/O.

### Migrating legacy CSV/FITS caches

```python
from lwa_catalog import CatalogLayout, migrate_output_dir

layout = CatalogLayout(OUTPUT_DIR)
migrate_output_dir(layout)  # writes Parquet; keeps legacy files by default
```

Or set `MIGRATE_LEGACY_CSV = True` in the build notebook config cell.
