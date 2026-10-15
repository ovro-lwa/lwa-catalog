# Notebooks

| Notebook | Role |
| -------- | ---- |
| `ovro_lwa_metacatalog.ipynb` | Discover FITS → PyBDSF → LST merge → global metacatalog (Parquet) |
| `ovro_lwa_metacatalog_subband.ipynb` | Same pipeline on 15 frequency subbands (18–82 MHz) |
| `ovro_lwa_mosaic_detect.ipynb` | Coadd hourly Full-band FITS to one SIN mosaic → PyBDSF → compare to LST-merged catalog |
| `ovro_lwa_healpix_tile_detect.ipynb` | Coadd hourly FITS to HEALPix (`nside=2048`) → nested TAN tiles (`nside_tile=4`) → PyBDSF; α from `beam_scale.json` or `BEAM_SCALE` dict |
| `beam_alpha_bootstrap.ipynb` | VLSSR-anchored + neighbor Total/Peak α → `beam_alpha_by_band.parquet` for re-detect |
| `metacatalog_sky_view.ipynb` | Load quality metacatalog and explore on the sky |
| `metacatalog_query.ipynb` | Browse all Parquet catalogs, sortable table, nearest-source coordinate query |
| `metacatalog_reliability.ipynb` | Build cleaned/gold reliability tiers from metacatalog + LST tree |
| `metacatalog_vlssr_qa.ipynb` | VLSSR cross-match QA — Blue completeness, over-split, multiplicity diagnostics |
| `metacatalog_spectral_modeling.ipynb` | Post-hoc Taylor spectral fits — BIC model selection, SED diagnostics |
| `metacatalog_nedlvs_crossmatch.ipynb` | NED-LVS cross-match — galaxy host association, recovery vs distance |
| `optical_crossmatch.ipynb` | Optical cross-match — NED-LVS hosts + MilliQUAS → `metacatalog_associations.parquet` |
| `radio_crossmatch.ipynb` | Radio survey attach (LoDeSS/VLSSR/NVSS/VLASS) → `metacatalog_spectral.parquet` + unique associations |
| `xray_crossmatch.ipynb` | eRASS:3 / Freund / Xu / 2RXS → `metacatalog_associations.parquet`; in-memory pie + eRASS/LWA HiPS sky QA |
| `metacatalog_classifications.ipynb` | Harmonize associations, flag conflicts, class pies / stacked bars |
| `target_samples.ipynb` | Class samples (galaxies, clusters, giant radio sources, peaked-spectrum, SNRs, pulsars, PWNe, XRBs, X-ray/optical) for `metacatalog_query.ipynb` |

## Catalog storage

Catalogs are written and read as **Apache Parquet** through `lwa_catalog`:

| Artifact | Path under `OUTPUT_DIR` |
| -------- | ----------------------- |
| Per-image sources | `sources_{lst}_{band}.parquet` |
| LST-merged band | `metacatalog_lst_{band}.parquet` |
| Global metacatalog (fusion + optional `quality_flag`) | `metacatalog.parquet` |
| Analysis subset (`spec_*`, optional survey attach) | `metacatalog_spectral.parquet` |
| Unique cross-match associations (classification bus) | `metacatalog_associations.parquet` |
| Quality bit diagnostics | `metacatalog_quality_flags.parquet` |
| Per-band beam α (re-detect input) | `beam_alpha_by_band.parquet` |
| BEAM_SCALE documentation | `beam_scale.json` |

Pipeline: reliability stamps `quality_flag` on `metacatalog.parquet` → optional
`radio_crossmatch.ipynb` quality-filters and writes `metacatalog_spectral.parquet`
with survey columns → `metacatalog_spectral_modeling.ipynb` reads that file (or
creates it from fusion) and adds `spec_*`. Crossmatch notebooks also append
unique matches into `metacatalog_associations.parquet` (replace-by-catalog);
`metacatalog_classifications.ipynb` harmonizes labels and plots conflicts.
Analysis notebooks load fusion via `read_metacatalog(layout)` (or
`prefer_spectral=True` for the subset product). Set `quality_mask=None` to skip
the default fusion mask.

Image products remain FITS. Detection and merge live in `lwa_catalog.create`;
notebooks keep configuration constants and call library APIs for I/O.

### Migrating legacy CSV/FITS caches

```python
from lwa_catalog import CatalogLayout, migrate_output_dir

layout = CatalogLayout(OUTPUT_DIR)
migrate_output_dir(layout)  # writes Parquet; keeps legacy files by default
```

Or set `MIGRATE_LEGACY_CSV = True` in the build notebook config cell.
