"""Optional parity checks for the survival forest against scikit-survival.

These scikit-survival (`sksurv`) tests are optional, so the tests are skipped unless it happens to
be installed in the environment. They do not assert exact numeric equality. The two forests use
independent RNGs, bootstrap draws, and split-search details, so bit-for-bit agreement is neither
expected nor meaningful. Instead they pin *behavioral* parity: Greenwood's `RandomSurvivalForest`
should reach discrimination comparable to sksurv's on the same data and produce risk rankings that
are strongly positively correlated with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import RandomSurvivalForest, Surv

sksurv_ensemble = pytest.importorskip("sksurv.ensemble")
sksurv_util = pytest.importorskip("sksurv.util")


@pytest.fixture(scope="module")
def split():
    lung = gw.load_dataset("lung", backend="pandas").dropna(
        subset=["ph.ecog", "ph.karno", "wt.loss"]
    )
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    x = lung[cols].to_numpy(dtype=float)
    time = lung["time"].to_numpy(dtype=float)
    event = (lung["status"] == 2).to_numpy()

    rng = np.random.default_rng(0)
    order = rng.permutation(len(time))
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]
    return (x[tr], time[tr], event[tr]), (x[te], time[te], event[te]), cols


def _shared_params() -> dict:
    return dict(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=0,
    )


def test_forest_matches_sksurv_discrimination(split) -> None:
    (xtr, ttr, etr), (xte, tte, ete), _ = split

    # Greenwood
    y_tr = Surv.right(ttr, event=etr)
    y_te = Surv.right(tte, event=ete)
    gw_forest = RandomSurvivalForest(**_shared_params()).fit(y_tr, xtr)
    gw_risk = gw_forest.predict(xte)
    gw_c = gw.concordance_index(y_te, gw_risk)

    # scikit-survival
    sk_y_tr = sksurv_util.Surv.from_arrays(event=etr, time=ttr)
    sk_forest = sksurv_ensemble.RandomSurvivalForest(**_shared_params()).fit(xtr, sk_y_tr)
    sk_risk = sk_forest.predict(xte)

    from sksurv.metrics import concordance_index_censored

    sk_c = concordance_index_censored(ete, tte, sk_risk)[0]

    # Comparable discrimination on the held-out set.
    assert abs(gw_c - sk_c) < 0.06

    # Both should beat chance by a clear margin.
    assert gw_c > 0.55 and sk_c > 0.55


def test_forest_risk_rank_correlates_with_sksurv(split) -> None:
    (xtr, ttr, etr), (xte, tte, ete), _ = split
    spearmanr = pytest.importorskip("scipy.stats").spearmanr

    gw_risk = (
        RandomSurvivalForest(**_shared_params()).fit(Surv.right(ttr, event=etr), xtr).predict(xte)
    )
    sk_y_tr = sksurv_util.Surv.from_arrays(event=etr, time=ttr)
    sk_forest = sksurv_ensemble.RandomSurvivalForest(**_shared_params()).fit(xtr, sk_y_tr)
    sk_risk = sk_forest.predict(xte)

    rho = spearmanr(gw_risk, sk_risk).statistic

    assert rho > 0.5
