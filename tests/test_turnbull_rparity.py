"""R-parity test for the Turnbull NPMLE.

`survival::survfit` dispatches `Surv(time1, time2, type = "interval2")` responses to its own,
independently-implemented Turnbull EM (`survival:::survfitTurnbull`). With only exact events and
right-censoring (no genuine interval ambiguity), both that implementation and Greenwood's must
compute the same, uniquely-identified NPMLE, which is exactly Kaplan-Meier. This is the one
regime where a tight tolerance is meaningful: once real interval ambiguity is introduced, R's
internal algorithm uses a different (and only loosely converged, `eps > 5e-5`) construction, so
comparing against it there would mostly be testing R's convergence behavior rather than
Greenwood's numerics (fixture: turnbull_veteran_km_equivalence.json).
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv, Turnbull

from ._r_parity import assert_allclose_to_r, load_fixture


@pytest.mark.rparity
def test_turnbull_veteran_matches_r_km_equivalence() -> None:
    fixture = load_fixture("turnbull_veteran_km_equivalence")
    veteran = gw.load_dataset("veteran", backend="pandas")
    y = Surv.right(veteran["time"], event=(veteran["status"] == 1))
    tb = Turnbull().fit(y)

    r_time = np.asarray(fixture["time"], dtype=float)
    r_surv = np.asarray(fixture["surv"], dtype=float)

    # Every Greenwood atom here should be a resolved point (no genuine interval ambiguity),
    # so `interval_high_` is directly comparable to R's `time`.
    # R's own EM only converges to `eps > 5e-5` on jump sizes (see survival:::survfitTurnbull),
    # so a residual of a few 1e-8 between two independent implementations is expected here.
    predicted = tb.predict(r_time)

    assert_allclose_to_r(predicted, r_surv, atol=1e-6, what="turnbull survival (veteran)")
