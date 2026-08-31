"""Arrow schemas for catalog Parquet layers."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

from lwa_catalog.constants import ASSOC_BANDS, BAND_FIELDS, GAUL_FLOAT_COLUMNS, SPECTRAL_INDEX_PAIRS


def sources_schema() -> pa.Schema:
    """Schema for per-image ``sources_*.parquet`` catalogs."""
    fields: list[pa.Field] = [
        pa.field(name, pa.float64(), nullable=True) for name in GAUL_FLOAT_COLUMNS
    ]
    fields.extend(
        [
            pa.field("S_Code", pa.string(), nullable=True),
            pa.field("Gaus_id", pa.int64(), nullable=True),
            pa.field("Isl_id", pa.int64(), nullable=True),
            pa.field("Source_id", pa.int64(), nullable=True),
            pa.field("lst_hour", pa.string(), nullable=True),
            pa.field("band", pa.string(), nullable=True),
            pa.field("source_file", pa.string(), nullable=True),
            pa.field("BMAJ", pa.float64(), nullable=True),
            pa.field("BMIN", pa.float64(), nullable=True),
            pa.field("BPA", pa.float64(), nullable=True),
            pa.field("time_key", pa.string(), nullable=True),
        ]
    )
    return pa.schema(fields)


def lst_merged_schema() -> pa.Schema:
    """Schema for ``metacatalog_lst_*.parquet`` (sources columns + LST merge meta)."""
    base = list(sources_schema())
    extra = [
        pa.field("n_lst_contributions", pa.int64(), nullable=True),
        pa.field("lst_hours", pa.string(), nullable=True),
        pa.field("representative_lst", pa.string(), nullable=True),
        pa.field("Peak_flux_std", pa.float64(), nullable=True),
        pa.field("cluster_jitter_rms_deg", pa.float64(), nullable=True),
        pa.field("Resid_Isl_rms", pa.float64(), nullable=True),
        pa.field("Resid_Isl_mean", pa.float64(), nullable=True),
    ]
    existing = {f.name for f in base}
    for field in extra:
        if field.name not in existing:
            base.append(field)
    return pa.schema(base)


def metacatalog_schema() -> pa.Schema:
    """Schema for global ``metacatalog.parquet``."""
    fields: list[pa.Field] = [
        pa.field("origin_band", pa.string(), nullable=True),
        pa.field("bands_present", pa.string(), nullable=True),
        pa.field("RA", pa.float64(), nullable=True),
        pa.field("DEC", pa.float64(), nullable=True),
        pa.field("Peak_flux", pa.float64(), nullable=True),
        pa.field("Total_flux", pa.float64(), nullable=True),
        pa.field("Maj", pa.float64(), nullable=True),
        pa.field("Min", pa.float64(), nullable=True),
        pa.field("PA", pa.float64(), nullable=True),
        pa.field("DC_Maj", pa.float64(), nullable=True),
        pa.field("DC_Min", pa.float64(), nullable=True),
        pa.field("DC_PA", pa.float64(), nullable=True),
        pa.field("BMAJ_match", pa.float64(), nullable=True),
        pa.field("BMAJ_full", pa.float64(), nullable=True),
        pa.field("n_lst_contributions", pa.int64(), nullable=True),
        pa.field("lst_hours", pa.string(), nullable=True),
        pa.field("representative_lst", pa.string(), nullable=True),
        pa.field("Peak_flux_std", pa.float64(), nullable=True),
        pa.field("cluster_jitter_rms_deg", pa.float64(), nullable=True),
        pa.field("Resid_Isl_rms", pa.float64(), nullable=True),
        pa.field("Resid_Isl_mean", pa.float64(), nullable=True),
        pa.field("E_Peak_flux", pa.float64(), nullable=True),
        pa.field("E_Total_flux", pa.float64(), nullable=True),
        pa.field("source_file_Full", pa.string(), nullable=True),
    ]
    for band in ASSOC_BANDS:
        fields.append(pa.field(f"n_assoc_{band}", pa.int64(), nullable=True))
        for name in BAND_FIELDS:
            fields.append(pa.field(f"{name}_{band}", pa.float64(), nullable=True))
        fields.append(pa.field(f"source_file_{band}", pa.string(), nullable=True))
    for label, _, _ in SPECTRAL_INDEX_PAIRS:
        fields.append(pa.field(f"alpha_{label}", pa.float64(), nullable=True))
        fields.append(pa.field(f"E_alpha_{label}", pa.float64(), nullable=True))
    return pa.schema(fields)


def table_from_dataframe(
    df: pd.DataFrame,
    schema: pa.Schema,
    *,
    include_extras: bool = True,
) -> pa.Table:
    """Build a :class:`pyarrow.Table` matching *schema*, filling missing cols with nulls.

    Extra DataFrame columns (not in *schema*) are appended when
    ``include_extras`` is true so frames can carry additional fields without
    failing the write.
    """
    n_rows = len(df)
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name in df.columns:
            arrays.append(pa.array(df[field.name], type=field.type))
        else:
            arrays.append(pa.nulls(n_rows, type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)

    if not include_extras:
        return table

    extra_names = [c for c in df.columns if c not in schema.names]
    if not extra_names:
        return table

    extra = pa.Table.from_pandas(df.loc[:, extra_names], preserve_index=False)
    combined = table
    for name in extra.column_names:
        combined = combined.append_column(name, extra[name])
    return combined
