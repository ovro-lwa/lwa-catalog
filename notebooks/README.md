# Notebooks

| Notebook | Role |
| -------- | ---- |
| `ovro_lwa_metacatalog.ipynb` | Discover FITS → PyBDSF → LST merge → global metacatalog (Parquet) |
| `ovro_lwa_metacatalog_subband.ipynb` | Same pipeline on 15 frequency subbands (18–82 MHz) |
| `ovro_lwa_mosaic_detect.ipynb` | Coadd hourly Full-band FITS to one SIN mosaic → PyBDSF → compare to LST-merged catalog |
| `metacatalog_sky_view.ipynb` | Load quality metacatalog and explore on the sky |
| `metacatalog_query.ipynb` | Browse all Parquet catalogs, sortable table, nearest-source coordinate query |
| `metacatalog_reliability.ipynb` | Build cleaned/gold reliability tiers from metacatalog + LST tree |
| `metacatalog_vlssr_qa.ipynb` | VLSSR cross-match QA — Blue completeness, over-split, multiplicity diagnostics |
| `metacatalog_spectral_modeling.ipynb` | Post-hoc Taylor spectral fits — BIC model selection, SED diagnostics |
| `metacatalog_nedlvs_crossmatch.ipynb` | NED-LVS cross-match — galaxy host association, recovery vs distance |

## Catalog storage

Catalogs are written and read as **Apache Parquet** through `lwa_catalog`:

| Artifact | Path under `OUTPUT_DIR` |
| -------- | ----------------------- |
| Per-image sources | `sources_{lst}_{band}.parquet` |
| LST-merged band | `metacatalog_lst_{band}.parquet` |
| Global metacatalog (fusion) | `metacatalog.parquet` |
| QA metacatalog (+ `quality_flag`) | `metacatalog_quality.parquet` |

Analysis notebooks load **`metacatalog_quality.parquet` by default** (when present) via
`read_metacatalog(layout)` and keep rows with `(quality_flag & 33267) == 0`. Set
`quality_mask=None` to skip filtering, or `prefer_quality=False` to read fusion
`metacatalog.parquet` (required when building quality flags).

Image products remain FITS. Detection and merge live in `lwa_catalog.create`;
notebooks keep configuration constants and call library APIs for I/O.

### Migrating legacy CSV/FITS caches

```python
from lwa_catalog import CatalogLayout, migrate_output_dir

layout = CatalogLayout(OUTPUT_DIR)
migrate_output_dir(layout)  # writes Parquet; keeps legacy files by default
```

Or set `MIGRATE_LEGACY_CSV = True` in the build notebook config cell.
