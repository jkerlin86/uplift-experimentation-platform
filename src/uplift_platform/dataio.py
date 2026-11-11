"""Data loading: chunked CSV-to-parquet conversion, full-sample
streaming statistics, and the seeded analysis sample.

The raw Criteo file is a 3.25 GB uncompressed CSV on a machine with
about 8 GB of RAM, so nothing here ever loads it whole. Conversion
streams 750k-row chunks into one parquet file with float32 features
and int8 indicators. Full-sample statistics are accumulated batch by
batch from the parquet. The analysis sample is drawn by Bernoulli
sampling over parquet batches with a fixed seed, then subsampled down
to the exact configured size, also seeded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DTYPES = {
    **{f"f{i}": "float32" for i in range(12)},
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}

PARQUET_SCHEMA = pa.schema(
    [(f"f{i}", pa.float32()) for i in range(12)]
    + [(c, pa.int8()) for c in ("treatment", "conversion", "visit", "exposure")]
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_csv_to_parquet(raw_csv_gz: Path, out_parquet: Path, chunk_rows: int) -> int:
    """Stream the gzipped CSV into a typed parquet file. Returns rows."""
    if out_parquet.exists():
        return int(pq.ParquetFile(out_parquet).metadata.num_rows)
    writer = pq.ParquetWriter(out_parquet, PARQUET_SCHEMA, compression="snappy")
    total = 0
    try:
        for chunk in pd.read_csv(
            raw_csv_gz, dtype=DTYPES, chunksize=chunk_rows, compression="gzip"
        ):
            table = pa.Table.from_pandas(chunk, schema=PARQUET_SCHEMA, preserve_index=False)
            writer.write_table(table)
            total += len(chunk)
    finally:
        writer.close()
    return total


def iter_batches(parquet_path: Path, batch_size: int = 500_000):
    pf = pq.ParquetFile(parquet_path)
    yield from pf.iter_batches(batch_size=batch_size)


def draw_analysis_sample(
    parquet_path: Path,
    target_n: int,
    probability: float,
    seed: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Seeded Bernoulli sample over parquet batches, cut to target_n."""
    rng = np.random.default_rng(seed)
    columns = feature_columns + ["treatment", "conversion", "visit", "exposure"]
    parts: list[pd.DataFrame] = []
    collected = 0
    for batch in iter_batches(parquet_path):
        frame = batch.to_pandas()
        keep = rng.random(len(frame)) < probability
        if keep.any():
            parts.append(frame.loc[keep, columns])
            collected += int(keep.sum())
        if collected >= int(target_n * 1.25):
            break
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > target_n:
        idx = np.sort(rng.choice(len(sample), size=target_n, replace=False))
        sample = sample.iloc[idx].reset_index(drop=True)
    return sample
