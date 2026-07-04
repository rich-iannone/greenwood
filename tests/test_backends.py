"""Backend-agnostic input tests.

Greenwood reads data through narwhals, so a `Surv` response built from any supported
data frame backend, and a model fit on any supported covariate frame, must give
identical numerical results. These tests exercise pandas, Polars, and PyArrow against a
pandas reference on the same data. They back the claim made in the user guide's
"Data sources and formats" page.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, KaplanMeier, Surv

TIMES = [180, 365, 730]

Reference = tuple[Any, Any]


@pytest.fixture(scope="module")
def lung_pd() -> Any:
    return gw.data.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def reference(lung_pd: Any) -> Reference:
    """Pandas reference: KM survival at fixed times and Cox coefficients."""
    y = Surv.right(lung_pd["time"], event=(lung_pd["status"] == 2))
    km = KaplanMeier().fit(y).predict(TIMES)
    cox = CoxPH().fit(y, lung_pd[["age", "sex"]]).to_dataframe()["estimate"].to_numpy()
    return km, cox


def test_pandas_columns_and_frame(lung_pd: Any, reference: Reference) -> None:
    ref_km, ref_cox = reference
    y = Surv.right(lung_pd["time"], event=(lung_pd["status"] == 2))
    np.testing.assert_allclose(KaplanMeier().fit(y).predict(TIMES), ref_km)
    cox = CoxPH().fit(y, lung_pd[["age", "sex"]]).to_dataframe()["estimate"].to_numpy()
    np.testing.assert_allclose(cox, ref_cox)


def test_polars_columns_and_frame(lung_pd: Any, reference: Reference) -> None:
    pytest.importorskip("polars")
    ref_km, ref_cox = reference
    lp = gw.data.load_dataset("lung", backend="polars")
    y = Surv.right(lp["time"], event=(lp["status"] == 2))
    np.testing.assert_allclose(KaplanMeier().fit(y).predict(TIMES), ref_km)
    cox = CoxPH().fit(y, lp[["age", "sex"]]).to_dataframe()["estimate"].to_numpy()
    np.testing.assert_allclose(cox, ref_cox)


def test_pyarrow_columns_and_frame(lung_pd: Any, reference: Reference) -> None:
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    ref_km, ref_cox = reference
    tbl = pa.Table.from_pandas(lung_pd)
    y = Surv.right(tbl["time"], event=pc.equal(tbl["status"], 2))
    np.testing.assert_allclose(KaplanMeier().fit(y).predict(TIMES), ref_km)
    cox = CoxPH().fit(y, tbl.select(["age", "sex"])).to_dataframe()["estimate"].to_numpy()
    np.testing.assert_allclose(cox, ref_cox)


def test_numpy_and_list_columns_agree(lung_pd: Any, reference: Reference) -> None:
    # Plain NumPy arrays and Python lists are also valid column inputs.
    ref_km, _ = reference
    time = np.asarray(lung_pd["time"], dtype=float)
    event = np.asarray(lung_pd["status"] == 2)
    np.testing.assert_allclose(
        KaplanMeier().fit(Surv.right(time, event=event)).predict(TIMES), ref_km
    )
    np.testing.assert_allclose(
        KaplanMeier().fit(Surv.right(list(time), event=list(event))).predict(TIMES),
        ref_km,
    )
