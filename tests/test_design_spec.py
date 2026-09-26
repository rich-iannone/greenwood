"""Tests for `DesignSpec`: `predict()` rebuilds the fitted design from any frame by column name."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import Outcome

DEATH = Outcome.right(time="time", event="status", event_value=2)


@pytest.fixture(scope="module")
def lung() -> Any:
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def veteran() -> Any:
    return gw.load_dataset("veteran", backend="pandas")


def _lp(model: Any, newdata: Any) -> Any:
    return np.asarray(model.predict(newdata, type="lp"), dtype=float)


def test_predict_from_full_frame_matches_sliced_frame(lung: Any) -> None:
    cox = gw.CoxPH().fit(DEATH, covariates=["age", "sex"], data=lung)
    np.testing.assert_allclose(_lp(cox, lung), _lp(cox, lung[["age", "sex"]]))


def test_columns_are_matched_by_name_not_position(lung: Any) -> None:
    cox = gw.CoxPH().fit(DEATH, covariates=["age", "sex"], data=lung)
    np.testing.assert_allclose(_lp(cox, lung[["sex", "age"]]), _lp(cox, lung[["age", "sex"]]))


def test_categorical_coding_uses_fitted_levels(veteran: Any) -> None:
    vet = Outcome.right(time="time", event="status")
    cox = gw.CoxPH().fit(vet, covariates=["age", "celltype"], data=veteran)
    full = _lp(cox, veteran)
    # Two rows with a single cell type still get all three dummy columns
    rows = veteran.index[veteran["celltype"] == "large"][:2]
    np.testing.assert_allclose(_lp(cox, veteran.loc[rows]), full[rows])


def test_formula_fit_predicts_from_full_frame(veteran: Any) -> None:
    cox = gw.CoxPH().fit("Surv(time, status) ~ age + C(celltype)", data=veteran)
    full = _lp(cox, veteran)
    np.testing.assert_allclose(_lp(cox, veteran.iloc[[3]]), full[[3]])


def test_missing_column_error(lung: Any) -> None:
    cox = gw.CoxPH().fit(DEATH, covariates=["age", "sex"], data=lung)
    with pytest.raises(ValueError, match="missing the column"):
        cox.predict(lung[["age"]])


def test_unseen_level_error(veteran: Any) -> None:
    cox = gw.CoxPH().fit(
        Outcome.right(time="time", event="status"), covariates=["celltype"], data=veteran
    )
    new = veteran.head(1).assign(celltype="unknown")
    with pytest.raises(ValueError, match="not present when the model was fit"):
        cox.predict(new)


def test_array_designs_still_work(lung: Any) -> None:
    frame = lung.dropna(subset=["age", "sex"])
    y = gw.Surv.right(time="time", event="status", data=frame, event_value=2)
    x = frame[["age", "sex"]].to_numpy()
    cox = gw.CoxPH().fit(y, covariates=x)
    np.testing.assert_allclose(_lp(cox, x[:3]), _lp(cox, x)[:3])
    with pytest.raises(ValueError, match="covariate column"):
        cox.predict(x[:, :1])


@pytest.mark.parametrize("backend", ["polars", "pyarrow", "duckdb"])
def test_predict_newdata_on_any_backend(lung: Any, backend: str) -> None:
    import polars as pl
    import pyarrow as pa

    cox = gw.CoxPH().fit(DEATH, covariates=["age", "sex"], data=lung)
    table = pa.Table.from_pandas(lung, preserve_index=False)
    if backend == "polars":
        newdata: Any = pl.from_pandas(lung)
    elif backend == "pyarrow":
        newdata = table
    else:
        import duckdb

        newdata = duckdb.from_arrow(table)
    np.testing.assert_allclose(_lp(cox, newdata), _lp(cox, lung))


@pytest.mark.parametrize(
    "make",
    [
        lambda: gw.AFT(dist="weibull"),
        lambda: gw.RoystonParmar(df=2),
        lambda: gw.PiecewiseExponential(breaks=[200.0, 400.0]),
        lambda: gw.BuckleyJames(),
    ],
    ids=["AFT", "RoystonParmar", "PiecewiseExponential", "BuckleyJames"],
)
def test_other_models_predict_from_full_frame(lung: Any, make: Any) -> None:
    model = make().fit(DEATH, covariates=["age", "sex"], data=lung)
    full = model.predict(lung.head(4), times=[180.0, 365.0], format="pandas")
    sliced = model.predict(lung[["age", "sex"]].head(4), times=[180.0, 365.0], format="pandas")
    np.testing.assert_allclose(full.to_numpy(dtype=float), sliced.to_numpy(dtype=float))


def test_fine_gray_predicts_from_full_frame() -> None:
    mg = gw.load_dataset("mgus2", backend="pandas")
    cr = Outcome.first_event(endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")})
    fg = gw.FineGray(cause="pcm").fit(cr, covariates=["age", "sex"], data=mg)
    a = fg.predict_cumulative_incidence(mg.head(3), times=[120, 240], format="pandas")
    b = fg.predict_cumulative_incidence(
        mg[["age", "sex"]].head(3), times=[120, 240], format="pandas"
    )
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float))


# -- missing categorical values -------------------------------------------------


@pytest.fixture(scope="module")
def veteran_na(veteran: Any) -> Any:
    frame = veteran.copy()
    frame["celltype"] = frame["celltype"].astype(object)
    frame.loc[[0, 5, 9], "celltype"] = None
    return frame


VET = Outcome.right(time="time", event="status")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_missing_category_drops_the_row(veteran_na: Any, backend: str) -> None:
    import polars as pl

    data = veteran_na if backend == "pandas" else pl.from_pandas(veteran_na)
    ref_frame = veteran_na.dropna(subset=["celltype"])
    y_ref = gw.Surv.right(time="time", event="status", data=ref_frame)
    ref = gw.CoxPH().fit(y_ref, covariates=["age", "celltype"], data=ref_frame)
    # A plain Surv over the full frame: the model's own complete-case step drops the rows
    y = gw.Surv.right(time="time", event="status", data=data)
    cox = gw.CoxPH().fit(y, covariates=["age", "celltype"], data=data)
    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)
    assert cox.n_dropped_ == 3
    assert "3 observations deleted due to missingness" in repr(cox)


def test_missing_category_in_formula_drops_the_row(veteran_na: Any) -> None:
    ref_frame = veteran_na.dropna(subset=["celltype"])
    ref = gw.CoxPH().fit("Surv(time, status) ~ age + C(celltype)", data=ref_frame)
    y = gw.Surv.right(time="time", event="status", data=veteran_na)
    cox = gw.CoxPH().fit(y, covariates="age + C(celltype)", data=veteran_na)
    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)
    assert cox.n_dropped_ == 3


def test_missing_category_gives_nan_prediction(veteran: Any, veteran_na: Any) -> None:
    cox = gw.CoxPH().fit(VET, covariates=["age", "celltype"], data=veteran)
    lp = _lp(cox, veteran_na)
    assert np.isnan(lp[[0, 5, 9]]).all()
    np.testing.assert_allclose(np.delete(lp, [0, 5, 9]), np.delete(_lp(cox, veteran), [0, 5, 9]))
    formula = gw.CoxPH().fit("Surv(time, status) ~ age + C(celltype)", data=veteran)
    assert np.isnan(_lp(formula, veteran_na)[[0, 5, 9]]).all()


def test_formula_predict_rejects_unseen_level(veteran: Any) -> None:
    cox = gw.CoxPH().fit("Surv(time, status) ~ age + C(celltype)", data=veteran)
    new = veteran.head(1).assign(celltype="unknown")
    with pytest.raises(ValueError, match="not present when the model"):
        cox.predict(new)
