# Agent memory — lwa-catalog

Living design notes for agents working in this repository. Distilled from the
2026-08 through 2026-09 research/plan/implement trail and checked against the
code as of 2026-09-07. Those markdown sources have been removed; **this file is
the memory.** When this file and the code disagree, **trust the code**, then
update this file. Do not resurrect CSV dual-write, HEALPix FITS maps, or
compatibility shims.

---

## Working conventions

- **No backwards-compatibility shims.** On rename/remove, make a clean break.
  Tell the operator to reload the notebook from disk and restart the kernel.
  Mention the new name clearly.
- **Notebooks/kernels lag `src/`.** Tests set `pythonpath = ["src"]`. Jupyter
  and `nbconvert` often import site-packages instead. After API changes:
  `pip install -e .` (or `PYTHONPATH=src`) then restart the kernel.
- **Optional extras:** `detect` (PyBDSF/`bdsf`), `viz` (Panel, ipyaladin),
  `analyze` (healpy, `lwa-healpix`). Lazy-import viz/healpy; do not import
  ipyaladin at module import time.
- **Commits:** conventional commits (`feat`, `fix`, `chore`, `docs`, `refactor`,
  `test`). Reference GitLab issues with `#<n>`.
- **Python 3.11+.** Core deps: numpy, pandas, astropy, pyarrow. Do not add
  sklearn/hdbscan/lmfit unless a later plan explicitly requires them.

---

## What this package does

Pipeline: **detect per hourly FITS** → **LST-merge within a band** → **fuse
bands into one metacatalog row per sky source** → **persist Parquet** →
**post-hoc analyze / QA / sky view**.

It is **catalog-then-merge**, not image-then-detect on an all-sky mosaic.
`src/lwa_catalog/` does not write mosaics or regrid FITS. Mosaic detection lives
in `notebooks/ovro_lwa_mosaic_detect.ipynb` as a comparison experiment.

Shared library was extracted from workflows prototyped in
[ovro-lwa-portal](https://github.com/uw-ssec/ovro-lwa-portal).

---

## Catalog layers (on disk)

Under `CatalogLayout(root)` / `OUTPUT_DIR` / `CATALOG_DIR`:

| Layer | Path | Role |
| ----- | ---- | ---- |
| Per-image sources | `sources_{lst}_{band}.parquet` | One PyBDSF `gaul` catalog per FITS |
| LST-merged band | `metacatalog_lst_{band}.parquet` | Same-source identity within one band |
| Global fusion | `metacatalog.parquet` | One row per unique sky source |
| Quality sidecar | `metacatalog_quality.parquet` | Fusion + `quality_flag` (never overwrite fusion) |
| Spectral sidecar | `metacatalog_spectral.parquet` | Fusion + `spec_model_*` |
| Radio sidecar | `metacatalog_radio.parquet` | Fusion + VLSSR/NVSS/VLASS photometry |
| Reliability HiPS | `hips_*_nside64/` | Peak-flux-weighted maps, not FITS |
| Sky PNGs | `sky_screenshots/` | `ipyaladin.save_view_as_image` |

**Catalog tables are Parquet only.** Image products stay FITS. Maps are HiPS
directories via `lwa-healpix`. Legacy CSV/FITS trees convert once with
`migrate_output_dir(layout)` (keeps legacy files by default). Do not dual-write
CSV/FITS catalogs or HEALPix FITS maps.

`read_metacatalog(layout)` prefers `metacatalog_quality.parquet` when present
(`prefer_quality=True`) and keeps rows with
`(quality_flag & DEFAULT_QUALITY_FLAG_MASK) == 0`.
`DEFAULT_QUALITY_FLAG_MASK = 247` (bits HAS_NAN through RESID_PCTL_MEAN).
Set `quality_mask=None` to skip filtering; `prefer_quality=False` to read
fusion `metacatalog.parquet` (required when *building* quality flags).

RGB color bands: `COLOR_BANDS = ("Full", "Blue", "Green", "Red")`.
Association order: Full+Blue seed, then Green, then Red (`ASSOC_BANDS`).
Subbands: 15 labels `18MHz`…`82MHz` (`SUBBAND_BANDS_MHZ`).

Cache is **path existence**, not content hashing.

---

## Module map

| Package | Owns |
| ------- | ---- |
| `lwa_catalog.paths.CatalogLayout` | All on-disk names |
| `lwa_catalog.schemas` | Arrow schemas per layer; extras via `include_extras=True` |
| `lwa_catalog.io` | Parquet `read_*` / `write_*`; pandas by default, `as_table=True` for Arrow |
| `lwa_catalog.constants` | Bands, beams, frequencies, required columns, overlay colors |
| `lwa_catalog.create` | Discover FITS, PyBDSF detect, LST merge, band fusion |
| `lwa_catalog.analyze` | Post-hoc science: reliability, VLSSR/NVSS/VLASS/NED-LVS, spectral, attach, rematch, Mahalanobis, HiPS maps |
| `lwa_catalog.viz` | Aladin overlays, HiPS URLs, FOV filter, coordinates |

**Post-hoc science belongs in `analyze/` (and `viz/`), not in fusion.** Do not
attach Fit-QA flags, `spec_*`, survey match flags, overlay state, or Mahalanobis
scores onto `metacatalog_schema()`. Sidecars and subset Parquets are the product
pattern. Preserve `meta_id`.

Analyze API pattern (repeat this): frozen `*Config`, `*Result` with `summary` /
tables / `warnings`, a batch function, `summarize_*` text, re-export from
`analyze/__init__.py`, smoke test in `tests/test_import.py`.

---

## Detection

- One detection unit = one 2-D FITS HDU at one LST hour (`FitsMetadata`:
  `path`, `lst_hour` `00h`–`23h`, `band`, optional `time_key`).
- Mosaic filenames without LST are **not** discovered. `detect_sources` can
  still run on a hand-built `FitsMetadata`.
- `prepare_hdu`: squeeze to 2-D `float32`; keep **NaN** for unobserved pixels
  so PyBDSF blanks them. Do not convert mosaic holes to 0.
- Requires `BMAJ`/`BMIN` (degrees). `BPA` defaults 0. Needs `CDELT1`/`CDELT2`
  (PC-only headers fail). Beam→pixels uses **center CDELT only**.
- Default PyBDSF: `thresh="hard"`, `thresh_isl=7`, `thresh_pix=4`,
  `atrous_do=False`, `psf_vary_do=False`. `check_outsideuniv` is **unset
  (False)** — SIN all-sky invalid pixels are not blanked by default.
- **Do not pass 1-D HEALPix, nfreq>1 cubes, or beamless `coadd_fits` arrays
  into `detect_sources`.** Package a 2-D FITS with WCS + beam + frequency
  first.
- Prefer `RA---CAR`/`DEC--CAR` if detecting on a CAR mosaic.
  `GLON-CAR`/`GLAT-CAR` still write columns named `RA`/`DEC` but the numbers
  are Galactic.
- lwa-healpix `min_elevation` assumes `CRVAL` is zenith — valid for native
  hourly SIN, **wrong** on reprojected CAR/HEALPix.
- Missing hours are skipped, not filled. Merge does not require 24 hours.

`GAUL_COLUMNS` kept from PyBDSF: positions, fluxes, shapes, `Resid_Isl_rms`,
`Resid_Isl_mean`. `S_Code` is a string classification (`S`/`C`/`M`), not a
numeric residual. Several former GAUL columns (`E_RA`, `Source_id`, island IDs,
…) are in `DROPPED_GAUL_COLUMNS` and are stripped on rewrite.

---

## Association and merge (the hard invariants)

**“Cluster” almost always means beam-sized sky association**, not ML clustering.

The universal match scale is `sep ≤ max(BMAJ_i, BMAJ_j)` via
`associate_catalogs` (Astropy `search_around_sky`). There is **no project-wide
fixed arcsecond radius**. Matching uses **`BMAJ` only** (`BMIN` unused).
Non-finite RA/DEC are skipped; non-finite `BMAJ` is treated as `0.0`.

Always reuse `associate_catalogs` / `associate_band_into_metacatalog`. Do not
invent a second matcher for VLSSR, rematch, reliability seeding, or survey
attach.

### Two graphs, two failure modes

| Step | Graph | Effect |
| ---- | ----- | ------ |
| LST merge (`merge_lst_metacatalog` / `_cluster_by_sky_position`) | Transitive **union-find** of all hours pooled together | A–B and B–C ⇒ one cluster even if A–C exceeds the beam (**over-merge**). Well-separated detections stay split (**over-split** persists as multiple `meta_id`s). |
| Band fusion (`build_global_metacatalog`) | **Bipartite** attach onto existing base rows | Does **not** merge base rows. `n_assoc_{band} > 1` is confused association (many detections on one row), not a duplicate-row detector. |

- Pool all hours, then cluster. Do not sequential-match hour N onto hour 0.
- `normalize_ra_columns` before clustering — PyBDSF negative longitudes split
  clusters across the RA wrap.
- Intra-band only inside `_cluster_by_sky_position`.
- Output sorted by `Peak_flux` descending.

### What a merged row actually stores

The representative row is **not** a cluster aggregate. Positions, fluxes, and
`Resid_Isl_*` come from the chosen member. Cluster summaries at merge time:

- `n_lst_contributions` — detection **count** (can exceed unique hours)
- `lst_hours` — sorted unique hours
- `representative_lst`
- `Peak_flux_std` (`ddof=1`; NaN if <2 finite)
- `cluster_jitter_rms_deg` — RMS of member–centroid great-circle offsets

Membership is **not persisted**. Recover with `rematch_meta_source` /
`associate_catalogs`. Durable detection key is
`(band, lst_hour, Source_id)` (+ `source_file` when present). `Source_id`
alone is not globally unique, and it may be absent on rewritten trees
(`DROPPED_GAUL_COLUMNS`).

`validate_metacatalog` / `validate_sources_catalog` are **schema** checks, not
scientific validation.

Required fusion columns: `RA`, `DEC`, `Peak_flux`, `origin_band`,
`bands_present`. Subband required set adds `astrometry_band` and has **no**
top-level `Peak_flux`.

---

## Pick rules (do not mix these up)

| Situation | Rule | Notes |
| --------- | ---- | ----- |
| LST representative | **Highest elevation** at cluster-median RA/DEC | Zenith at `RA = LST × 15°`, `Dec = OVRO_LATITUDE_DEG` (37.239777). Merged `Peak_flux` can be the **faintest** member. Tests lock this. |
| Band-fusion multi-match at **build** time | **Highest elevation** (`representative="elevation"`) | `n_assoc_*` still counts **all** hits. |
| LST cluster-center **nudge during clustering** | **Median flux** (`_pick_median_flux_row`) | Only this use. Not a seeder. |
| Survey photometric attach | **Brightest `Peak_flux`** (`representative="peak_flux"`) | NVSS `Peak_intensity` is normalized first. |
| Rematch among beam neighbors | **Seeded `Peak_flux`, then sky separation** | Elevation at rematch recovered the wrong source in confused beams (`meta_id` 17776). |

Elevation is an **LST / LWA** concept. External surveys have no `lst_hour`.

---

## RGB vs subband vs survey columns

Wide layout: `{field}_{band}`, `origin_band`, comma-separated `bands_present`,
`n_assoc_{band}`. Missing measurements are `NaN`. RGB schema is the typed core;
MHz extras persist via `include_extras=True`.

**`bands_present` is not a flux-validity mask.** A band can be listed while
`{flux}_{band}` is `NaN`. Modeling and SED work must require **positive finite**
flux per channel.

| Product | Flux layout | Spectral | Astrometry |
| ------- | ----------- | -------- | ---------- |
| RGB metacatalog | Top-level flux from `origin_band` + `{field}_{Blue,Green,Red}` | Merge-time two-point `alpha_RG` / `alpha_GB` on **Total_flux** | LWA `RA`/`DEC`; `BMAJ_match` grows with attached LWA bands |
| MHz subband | No top-level flux; `Peak_flux_{18MHz…82MHz}` (15×) | Post-hoc Taylor `spec_model_*` at ν₀ = **55 MHz** | `astrometry_band`; no merge-time `alpha_*` |
| Radio attach | `Peak_flux_{VLASS,NVSS,VLSSR}` on **existing LWA rows** | Not in default spectral `bands` | LWA astrometry **frozen**; `BMAJ_match` not updated |

Two-point `alpha = log(S_a/S_b) / log(ν_a/ν_b)` is a 1-term log-power-law
between two frequencies, **not** a multi-point SED. `a₁` from the Taylor fit is
the spectral index **at ν₀**, not the mean across the bandpass.

Blue ≈ 74 MHz (`BAND_FREQ_HZ["Blue"]` ≈ 73.96 MHz) — natural VLSSR comparison
band. Target rows with `"Blue" in bands_present` (`select_blue_associated_rows`),
not `origin_band == "Blue"` only.

---

## Survey attach and cross-match QA

External catalogs live under `REFERENCE_CATALOGS_DIR = Path("/fast/claw/catalogs")`
(machine-specific; tests must not depend on it).

| Survey | Freq | Beam | Default path |
| ------ | ---- | ---- | ------------ |
| VLSSR | 74 MHz | circular 80″ | `vlssr_radecpeak.txt` |
| NVSS | 1.4 GHz | 45″ | `nvss/nvss_vizier.parquet` |
| VLASS | ~3 GHz | 2.5″ | CIRADA QL component CSV |
| NED-LVS | hosts | position+diameter | `NEDLVS_current.fits` |

**LWA-centric astrometry.** Match on LWA `RA`/`DEC` + LWA `BMAJ`
(`resolve_bmaj` / `BMAJ_match`). Surveys do not rewrite positions.

Photometric attach (`attach_radio_surveys_to_metacatalog` in
`analyze/survey_attach.py`):

- Order **VLASS → NVSS → VLSSR** (high → low frequency).
- Flags: `append_unmatched=False`, `update_bmaj_match=False`,
  `representative="peak_flux"`, plus `base_bmaj` so later radii do not loosen.
- Output **row count equals input LWA row count**. Unmatched survey sources
  never become rows.
- `n_assoc_{survey}` counts all hits; stored flux is the **brightest** hit.
- **Never** `append_unmatched=True` for VLASS/NVSS (component tables would
  dominate).
- **Never** `astrometry_from_highest_frequency` for this attach (VLASS
  components are not LWA centroids).

Match-direction diagnostics (VLSSR QA, reusable):

- `associate_catalogs(meta, vlssr)` — completeness / many VLSSR per meta
  (**expected**: LWA beam ≫ 80″).
- `associate_catalogs(vlssr, meta)` — many meta per VLSSR (**over-split**).

QA is report-only: do not write-side filter the parent catalog from VLSSR /
Mahalanobis / Fit-quality notebooks. Reliability tiers and `quality_flag`
sidecars are the first-class subset products, still leaving fusion intact.

---

## Reliability, quality flags, fit QA

Three related but **not interchangeable** layers:

1. **Notebook Fit quality** (`ovro_lwa_metacatalog.ipynb`, helpers local):
   within-band **top 1%** `Resid_Isl_rms` ∪ `|Resid_Isl_mean|`; unphysical
   `sigma = (Total−Peak)/hypot(E) < -3`; 1°×1° `histogram2d` density. Operates
   on **`lst_merged`**, not fusion. Missing errors ⇒ not flagged.
2. **Library reliability** (`analyze/reliability.py`): absolute residual cuts
   (default **1.0 Jy/beam** RMS and |mean|); soft-exclude `cleaned` vs
   hard-include `gold`; contract **`gold ⊆ cleaned`**. Soft: missing evidence
   **keeps** the row. Hard: statistic must be calculable and pass.
   `filter_metacatalog_reliability()` builds flags once and returns both tiers
   (avoids double rematch I/O).
3. **`quality_flag` bitmask** (`SourceQualityFlag`): 0 = check passed, 1 =
   concern. `quality_flag == 0` means every implemented check passed. Bits
   16–31 reserved. Written to `metacatalog_quality.parquet`.

Do not conflate **percentile QA** (Fit quality, Mahalanobis) with **absolute
library cuts** (reliability E3 / `RESID_ABS_FAIL`).

Residual / unphysical / jitter gates evaluate the **origin-band seed LST row**,
not an aggregate over cluster members. Residuals on the global metacatalog are
that representative detection (schema now includes `Resid_Isl_*` /
`cluster_jitter_rms_deg` / `S_Code` — older research saying “residuals stop at
LST-merged” is stale).

Multi-image (`passes_multi_image`): `n_lst_contributions ≥ 2` **or** ≥2 bands
in `bands_present` with `n_assoc_* == 1`. Confused bands do not count.

Unique-assoc default: **on for gold, off for cleaned**.

HiPS maps: `metacatalog_to_healpix(profile="gaussian")` paints elliptical
Gaussians (`Peak_flux`, `Maj`/`Min` FWHM deg, `PA` N→E). `profile="point"` is
single-pixel deposits. **Map sum is not Σ Peak_flux.** Then
`write_healpix_hips` / `metacatalog_to_hips`. There is no `write_healpix_fits`.

---

## Spectral modeling

`analyze/spectral.py`, notebook `metacatalog_spectral_modeling.ipynb`.

- Model: `ln S(ν) = Σ a_j [ln(ν/ν₀)]^j`, ν₀ = `SUBBAND_REF_FREQ_MHZ = 55.0`.
- Default `flux_kind="total"`; `peak` supported. Weights `σ_lnS = E_S/S`;
  missing errors → equal weights.
- Model selection is **reduced-χ² parsimony tie-break, then BIC** — not
  BIC-only (BIC-only over-fit noise-free power laws to 4-term models).
- Columns: `spec_model_n_terms`, `spec_model_bic`, `spec_model_chi2_red`,
  `spec_model_n_flux`, `spec_model_nu0_mhz`, `spec_model_a0`…`a3`.
- 0 valid fluxes → all NaN, `n_flux=0`; 1 valid → 1-term, `a0 = ln(S)`.
- v1 is a Python row loop. Optional sidecar via notebook `WRITE_OUTPUT`.
- Do not put VLASS/NVSS/VLSSR into default `bands`.

---

## Feature-space outliers (Mahalanobis)

Chose **Option B** (classical numpy Mahalanobis), not DBSCAN/HDBSCAN/K-Means/GMM
and not sklearn MCD.

- Module: `analyze/anomaly.py` — `mahalanobis_outlier_scores`,
  `DEFAULT_MAHALANOBIS_COLUMNS`, `resolve_mahalanobis_columns`.
- Score `browser._df` after quality mask and `RADIO_QA_FILTER`. Independent
  MultiChoice (not `display_columns`). Explicit **Run**; **no auto-rerun** on
  filter Apply/Clear.
- Complete-case default (`incomplete="drop"`). Optional `impute_median` still
  fits/thresholds on complete cases. Rows with all selected features missing
  stay unscored.
- d² uses `np.linalg.pinv`. Outlier iff `d2 >=` percentile threshold
  (default 99). Do not write scores into Parquet or fold into `quality_flag`.
- Do not cluster unscaled mixed units or treat raw `RA`/`DEC` as Euclidean.
  `origin_band` factorize codes impose an arbitrary metric.

UI lives at the bottom of `metacatalog_query.ipynb` (and
`notebooks/anomaly_detection.ipynb` exists as a later notebook).

---

## Visualization

- Mix-and-match: **Catalog dropdown and HiPS dropdown are independent.** Do not
  sync catalog to HiPS survey.
- ipyaladin 0.8: one base `survey` + one `overlay_survey`. **Cannot stack
  NVSS+VLASS rasters.** Radio QA toggles **one** of LWA / VLSSR / NVSS / VLASS.
- Vector overlays (ellipses + markers) can be multi-catalog. LWA uses
  `Maj`/`Min`/`PA`; external surveys use circular `VLSSR_BMAJ_DEG` /
  `NVSS_BMAJ_DEG` / `VLASS_BMAJ_DEG` via `catalog_with_survey_beam`.
- LWA band palette (`BAND_OVERLAY_COLORS`) is for LWA bands only. External
  surveys pass `overlay_catalog_by_band(..., color=)`.
- FOV filter then cap (default ~500). Overlay names `catalog_Red`, etc.;
  rematch prefixes `trace_lst`, `trace_src`. `replace=True` removes before
  re-add. Cross fallback when ellipse axes incomplete. PA is N→E.
- Browser fetches HiPS tiles (kernel does not). URLs must be reachable from
  the user’s browser (SSH tunnels). Local `/fast/claw` HiPS are LWA-only;
  VLSSR/NVSS/VLASS use public CDS/NRAO URLs (`survey_hips_url`).
- Sky wiring stays in notebooks (`CatalogBrowser`, `RadioCrossmatchSkyQA`);
  reusable helpers live in `lwa_catalog.viz`. Pan/zoom uses
  `DebouncedAladinViewRefresh`. Explicit **Load sky view** / **Run**.

---

## Notebooks

Thin cells over library APIs. Config cell sets `CATALOG_DIR` / `CatalogLayout`.
Pandas filters in config (`RADIO_QA_FILTER`), not expression-eval UIs. List new
notebooks in `notebooks/README.md` (that file currently lags: it omits
`radio_crossmatch.ipynb`, `anomaly_detection.ipynb`, and
`metacatalog_association_qc_summary.ipynb`).

| Notebook | Role |
| -------- | ---- |
| `ovro_lwa_metacatalog.ipynb` | RGB detect → LST merge → fusion; Fit quality section |
| `ovro_lwa_metacatalog_subband.ipynb` | Same on 15 MHz subbands |
| `ovro_lwa_mosaic_detect.ipynb` | Coadd experiment vs LST-merged catalog |
| `metacatalog_query.ipynb` | Browse, sky overlay, **source trace**, Mahalanobis |
| `metacatalog_reliability.ipynb` | `cleaned` / `gold` / `quality_flag` / HiPS |
| `metacatalog_vlssr_qa.ipynb` | Blue completeness, over-split, multiplicity |
| `metacatalog_spectral_modeling.ipynb` | Taylor SED fits |
| `radio_crossmatch.ipynb` | NVSS/VLASS/VLSSR match, survey attach, Visual QA |
| `metacatalog_nedlvs_crossmatch.ipynb` | Galaxy host association (later than this distillation) |
| `target_samples.ipynb` | Class samples for the query browser |

Trace UI is **query-notebook only**. Do not put rematch cells back into
`ovro_lwa_metacatalog.ipynb`.

---

## Testing

- Synthetic DataFrames and `tmp_path` Parquet only. **No CI on `/fast/claw` or
  PyBDSF.**
- Reuse `_src()`-style row builders from `tests/test_merge.py`.
- Optional deps: `pytest.importorskip("healpy")`; viz tests mock Aladin.
- Notebooks are **manual integration**. Automated checks: pytest, ruff,
  `compile()` of helpers, JSON parse. Almost every implement summary left the
  real-tree Jupyter run unchecked — pytest is not a substitute.
- After import changes, operators must restart kernels.

Locked constructions worth preserving in tests: unphysical 3σ; gold ⊆ cleaned;
confused band does not buy multi-image; survey attach row-count and frozen
astrometry; rematch prefers seeded flux over a bright beam neighbor; spectral
power-law recovers `a1 ≈ α` with parsimony; Mahalanobis threshold equals
`np.percentile` of finite d².

---

## Never reintroduce

- Deprecated aliases, stub re-exports, or “keep old notebook imports working.”
- CSV or FITS **catalog table** writes; Arrow IPC / Feather / hive partitioning
  as primary storage; dual-read of CSV on the hot path.
- `write_healpix_fits` / HEALPix FITS as the map product.
- `resid_thresh_jy` (use dual `resid_rms_thresh_jy` / `resid_mean_thresh_jy`).
- Brightest-flux or median-flux as the **LST representative**.
- `_pick_median_flux_row` as band-association seeder.
- Elevation pick among **rematch** beam neighbors.
- `append_unmatched=True` or `astrometry_from_highest_frequency` for
  VLASS/NVSS/VLSSR attach.
- BIC-only nested Taylor selection; surveys in default spectral `bands`.
- A second sky matcher beside `associate_catalogs`.
- Auto-rerunning Mahalanobis when filters change; writing those flags into
  `quality_flag`.
- Stacking NVSS+VLASS HiPS rasters via `overlay_survey`.
- LWA band colors on external-survey overlays.
- Silently switching healpix default back to point deposits.
- Treating `display_columns` as the Mahalanobis feature selector.
- Passing HEALPix maps or beamless coadds to PyBDSF.
- Using `bands_present` as the SED mask, or two-point `alpha_*` as a Taylor SED.
- Persisting member ID lists into Parquet (rematch instead), unless a new plan
  explicitly replaces rematch.

---

## Implementation lessons

- **When reality diverges from a plan, the code is the source of truth.**
  Recurring overrides: HiPS not FITS; Gaussian healpix; Panel trace in the query
  notebook; rematch flux-then-sep; χ²-then-BIC; `color=` for survey overlays.
  Stale “Key Changes” bullets in implement docs sometimes contradict later
  deviations.
- **Convenience wrappers that avoid double I/O are welcome**
  (`filter_metacatalog_reliability`). Dual public APIs for the same job are not.
- **Interactive QA must be explicit:** Mahalanobis `Run`, VLSSR
  `RUN_LST_MERGED_PASS = False`, spectral `WRITE_OUTPUT`, radio filter-then-load.
- **`EditNotebook` can leave the wrong cell language.** If a markdown cell
  becomes `code`, patch notebook JSON.
- Units that keep biting: `BMAJ` in **degrees**; VLSSR 80″; jitter in degrees;
  residuals **Jy/beam**; fluxes **Jy**; frequencies **Hz**.

---

## Open and deferred

Still true after the workflow trail (manual Jupyter on `/fast/claw` was almost
never signed off):

- Mosaic WCS/beam for detection; whether `n_lst_contributions` means anything
  on a single coadd.
- VLSSR completeness denominator (all sources vs Dec cut vs LWA footprint /
  sensitivity). Seeding union-find from VLSSR was explicitly **out of scope**.
- Peak+Total spectral fits in one pass; RGB as default spectral input; HiPS of
  spectral/VLSSR flags.
- Robust/MCD covariance or clustering Options A/C; folding n-D outliers into
  `quality_flag`.
- HiPS overlay Phase 4: tier-diff overlays, MOC footprints.
- Persist member-level tables instead of rematching.
- `notebooks/README.md` and root `README.md` still under-document radio/spectral
  sidecars and several notebooks.

Later than this distillation (exists in code; treat the modules as authority):
`analyze/nedlvs.py`, `nvss.py`, `vlass.py`, `bootstrap.py`,
`crossmatch_radius.py`, expanded `SourceQualityFlag`,
`notebooks/metacatalog_nedlvs_crossmatch.ipynb`.
