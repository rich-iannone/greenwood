"""Tests for elastic-net penalized Cox regression (`CoxNet`).

No R reference (glmnet is not installed), so correctness is pinned three ways: `penalizer=0`
must reproduce the R-validated unpenalized Breslow `CoxPH`; the solution must satisfy the
elastic-net KKT optimality conditions; and the penalty must produce the expected sparsity.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxNet, CoxPH, Surv
from greenwood._cox import _cox_terms


@pytest.fixture(scope="module")
def data():
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    x = lung[cols].fillna(lung[cols].mean())
    return y, x


def _max_kkt_residual(model: CoxNet) -> float:
    """Largest violation of the elastic-net subgradient optimality conditions."""
    x, center = model._x, model._center
    scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    xs = (x - center) / scale
    beta = model.coef_ * scale
    n = x.shape[0]
    groups = [(np.arange(n), np.unique(model._exit[model._event]))]
    _, grad, _ = _cox_terms(
        beta, xs, model._entry, model._exit, model._event, model._weight, groups, "breslow"
    )
    lam, alpha = model.penalizer, model.l1_ratio
    grad_h = -grad / n + lam * (1.0 - alpha) * beta
    residual = 0.0
    for j in range(beta.size):
        if abs(beta[j]) > 1e-8:
            residual = max(residual, abs(grad_h[j] + lam * alpha * np.sign(beta[j])))
        else:
            residual = max(residual, abs(grad_h[j]) - lam * alpha)
    return residual


def test_penalizer_zero_matches_unpenalized_cox(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    ref = CoxPH(ties="breslow").fit(y, x).coef_
    cn = CoxNet(penalizer=0.0).fit(y, x).coef_
    np.testing.assert_allclose(cn, ref, atol=1e-4)


@pytest.mark.parametrize("penalizer,l1_ratio", [(0.05, 1.0), (0.1, 0.5), (0.2, 0.5), (0.05, 0.0)])
def test_kkt_conditions_hold(data, penalizer, l1_ratio) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=penalizer, l1_ratio=l1_ratio).fit(y, x)
    assert _max_kkt_residual(model) < 1e-4


def test_lasso_sparsity_increases_with_penalizer(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    counts = [
        int(np.count_nonzero(CoxNet(penalizer=lam, l1_ratio=1.0).fit(y, x).coef_))
        for lam in (0.01, 0.05, 0.2, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)  # monotonically fewer nonzero
    assert counts[-1] == 0  # a large lasso penalty zeros everything


def test_ridge_shrinks_without_zeroing(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    ridge = CoxNet(penalizer=0.1, l1_ratio=0.0).fit(y, x)
    unpen = np.abs(CoxPH(ties="breslow").fit(y, x).coef_)
    assert np.all(np.count_nonzero(ridge.coef_) == ridge.coef_.size)  # nothing set to zero
    assert np.sum(np.abs(ridge.coef_)) < np.sum(unpen)  # but shrunk toward zero


def test_predict_shapes_and_survival_range(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.05, l1_ratio=0.5).fit(y, x)
    lp = model.predict(x, type="lp")
    assert lp.shape == (model.n_,)
    surv = model.predict(x.iloc[:3], type="survival", times=[180, 365])
    assert list(surv.columns) == ["time", "subject_1", "subject_2", "subject_3"]
    assert ((surv.iloc[:, 1:] >= 0) & (surv.iloc[:, 1:] <= 1)).all().all()


def test_to_dataframe_and_repr(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.1, l1_ratio=1.0).fit(y, x)
    df = model.to_dataframe()
    assert list(df.columns) == ["term", "estimate", "hazard_ratio"]
    text = repr(model)
    assert "elastic-net Cox" in text
    assert "nonzero coefficients" in text
    assert "object at 0x" not in text


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="penalizer"):
        CoxNet(penalizer=-1.0)
    with pytest.raises(ValueError, match="l1_ratio"):
        CoxNet(l1_ratio=2.0)
