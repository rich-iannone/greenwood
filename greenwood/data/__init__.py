"""Bundled survival datasets for docs, tests, and the R-parity harness.

Datasets are stored as gzipped CSVs next to this module and regenerated from their
authoritative R sources by `scripts/export_datasets.R`. They are loaded lazily.

Available datasets (all from R's `survival` package):

- `lung`: NCCTG lung cancer (228 x 10). Note `status` is coded 1 = censored, 2 = dead, so
  build the response with `Surv.right(time, event=(status == 2))`.
- `veteran`: Veterans' Administration lung cancer trial (137 x 8).
- `ovarian`: ovarian cancer survival (26 x 6).
- `pbc`: Mayo Clinic primary biliary cholangitis (418 x 20).
- `colon`: chemotherapy for colon cancer (1858 x 16).
- `mgus2`: monoclonal gammopathy (1384 x 11), a competing-risks dataset (progression to
  plasma-cell malignancy vs death). Build the endpoint with `ptime`/`pstat` (progression)
  and `futime`/`death`.
"""

from __future__ import annotations

import gzip
from importlib.resources import files
from typing import Any

__all__ = ["load_dataset", "available_datasets"]

_DATASETS = {
    "lung": "lung.csv.gz",
    "veteran": "veteran.csv.gz",
    "ovarian": "ovarian.csv.gz",
    "pbc": "pbc.csv.gz",
    "colon": "colon.csv.gz",
    "mgus2": "mgus2.csv.gz",
}


def available_datasets() -> list[str]:
    """Return the names of the bundled datasets."""
    return sorted(_DATASETS)


def load_dataset(name: str, *, backend: str = "pandas") -> Any:
    """Load a bundled dataset by name.

    Parameters
    ----------
    name
        One of `available_datasets` (e.g. `"lung"`, `"veteran"`).
    backend
        `"pandas"` (default) or `"polars"`.

    Returns
    -------
    A dataframe in the requested backend.
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {available_datasets()}.")
    resource = files("greenwood.data").joinpath(_DATASETS[name])
    text = gzip.decompress(resource.read_bytes()).decode("utf-8")

    if backend == "pandas":
        import io

        import pandas as pd

        return pd.read_csv(io.StringIO(text))
    if backend == "polars":
        import polars as pl

        return pl.read_csv(text.encode("utf-8"))
    raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")
