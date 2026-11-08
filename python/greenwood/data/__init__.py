"""Bundled survival datasets for docs, tests, and the R-parity harness.

Datasets are stored as gzipped CSVs next to this module and regenerated from their
authoritative R sources by `scripts/export_datasets.R`, which also writes `levels.json`, the
levels of each R factor column. They are loaded lazily.

Available datasets (from R's `survival` package, except `e1684`, which is from `smcure`):

- `lung`: NCCTG lung cancer (228 x 10). Note `status` is coded 1 = censored, 2 = dead. As in
  R, this coding passes straight through: `"Surv(time, status)"`.
- `veteran`: Veterans' Administration lung cancer trial (137 x 8).
- `ovarian`: ovarian cancer survival (26 x 6).
- `pbc`: Mayo Clinic primary biliary cholangitis (418 x 20). `status` is coded 0 = censored,
  1 = transplant, 2 = dead, so death is `"Surv(time, status == 2)"`.
- `pbcseq`: Mayo PBC repeated lab measurements (1945 x 19). Companion to `pbc`: one row
  per (subject, visit) with time-varying lab values. Link to `pbc` on `id`; `day` is the
  visit time. Use `split_episodes(baseline=pbc, visits=pbcseq, id="id", time="time",
  event=..., visit_time="day")` to build the counting-process dataset for a TVC Cox model.
- `colon`: chemotherapy for colon cancer (1858 x 16).
- `mgus2`: monoclonal gammopathy (1384 x 11), a competing-risks dataset (progression to
  plasma-cell malignancy vs death). Build the endpoint from `ptime`/`pstat` (progression)
  and `futime`/`death` with `gw.Outcome.first_event(endpoints={"pcm": ("ptime", "pstat"),
  "death": ("futime", "death")})`.
- `e1684`: ECOG E1684 melanoma trial of interferon alpha-2b (284 x 5), a dataset with a cured
  fraction. Use `"Surv(FAILTIME, FAILCENS)"`.

The factor columns (veteran `celltype`, colon `rx`, and `sex` in pbc, pbcseq, and mgus2) load as
categoricals with R's levels, so models take R's reference level and curves come in R's order.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from importlib.resources import files
from typing import Any

__all__ = ["load_dataset", "available_datasets"]

_DATASETS = {
    "lung": "lung.csv.gz",
    "veteran": "veteran.csv.gz",
    "ovarian": "ovarian.csv.gz",
    "pbc": "pbc.csv.gz",
    "pbcseq": "pbcseq.csv.gz",
    "colon": "colon.csv.gz",
    "mgus2": "mgus2.csv.gz",
    "e1684": "e1684.csv.gz",
}


def available_datasets() -> list[str]:
    """Return the names of all bundled datasets.

    Returns
    -------
    list[str]
        Sorted list of dataset names that can be passed to `load_dataset()`.

    Examples
    --------
    ```{python}
    import greenwood as gw

    # List all bundled dataset names
    gw.available_datasets()
    ```
    """
    return sorted(_DATASETS)


def _resolve_backend(backend: str | None) -> str:
    """Pick a data frame backend, matching Greenwood's dataframe-agnostic stance.

    When `backend` is `None`, prefer Polars if it is installed, otherwise fall back to
    pandas. If neither is installed, raise a clear error. An explicit `"pandas"` or
    `"polars"` is honored as given.
    """
    if backend is None:
        if importlib.util.find_spec("polars") is not None:
            return "polars"
        if importlib.util.find_spec("pandas") is not None:
            return "pandas"
        raise ImportError(
            "load_dataset needs either polars or pandas installed. Install one "
            "(for example `pip install greenwood[pl]` or `greenwood[pd]`), or pass "
            "an explicit backend."
        )
    if backend not in ("pandas", "polars"):
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")
    return backend


def load_dataset(name: str, *, backend: str | None = None) -> Any:
    """Load a bundled dataset by name.

    Greenwood ships several classic survival-analysis datasets from R's `survival`{.gd-no-link}
    package, stored as gzipped CSVs. This function decompresses them on the fly and returns a
    DataFrame in your preferred backend.

    Parameters
    ----------
    name
        One of `available_datasets()` (e.g., `"lung"`, `"veteran"`, `"ovarian"`, `"pbc"`,
        `"pbcseq"`, `"colon"`, `"mgus2"`).
    backend
        `"pandas"` or `"polars"`. When left as `None` (the default), Greenwood picks a
        backend for you: it prefers Polars if it is installed, otherwise uses Pandas, and
        raises if neither is available.

    Returns
    -------
    DataFrame
        A Polars or Pandas DataFrame with the dataset contents. Columns that are factors in R
        (veteran `celltype`, colon `rx`, and `sex` in pbc, pbcseq, and mgus2) are categoricals
        with R's levels in R's order: a Polars `Enum` or a Pandas `Categorical`. Models then take
        the first level as the reference, and curves and tests come in level order.

    Examples
    --------
    Load the NCCTG lung cancer dataset as a Polars DataFrame:

    ```{python}
    import greenwood as gw

    # Load the NCCTG lung cancer dataset as a Polars DataFrame
    lung = gw.load_dataset("lung", backend="polars")
    lung.head()
    ```

    The cell type in the veteran dataset keeps R's levels, with squamous first:

    ```{python}
    # The cell type is an Enum with R's levels
    veteran = gw.load_dataset("veteran", backend="polars")
    veteran["celltype"].dtype
    ```

    See all available dataset names:

    ```{python}
    # List all available dataset names
    gw.available_datasets()
    ```
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {available_datasets()}.")
    resource = files("greenwood.data").joinpath(_DATASETS[name])
    content = gzip.decompress(resource.read_bytes())
    factors = _factor_levels(name)

    resolved = _resolve_backend(backend)
    if resolved == "polars":
        import polars as pl

        return pl.read_csv(
            content, schema_overrides={column: pl.Enum(lv) for column, lv in factors.items()}
        )

    import io

    import pandas as pd

    return pd.read_csv(
        io.BytesIO(content),
        dtype={column: pd.CategoricalDtype(lv) for column, lv in factors.items()},
    )


def _factor_levels(name: str) -> dict[str, list[str]]:
    """The levels of each factor column of a dataset, in R's order, from `levels.json`."""
    recorded: dict[str, dict[str, list[str]]] = json.loads(
        files("greenwood.data").joinpath("levels.json").read_text(encoding="utf-8")
    )
    return recorded.get(name, {})
