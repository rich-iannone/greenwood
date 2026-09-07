"""R-parity test for AFT's three-parameter (threshold) Weibull model.

Validates the location/threshold reparameterization and likelihood construction against an
independently-implemented three-parameter Weibull MLE fit (R's `fitdistrplus::fitdist` with
custom `dweibull3`/`pweibull3` density/CDF functions).

`fitdistrplus` in this environment does not expose a maximum-product-of-spacings method (no
`"mps"` method, no `"MPS"` gof option in `mgedist`), so `method="mps"` is validated only by
internal consistency properties in `tests/test_aft_threshold_mps.py`, not against this fixture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import greenwood as gw
from greenwood import AFT, Surv

from ._r_parity import load_fixture


@pytest.mark.rparity
def test_threshold_weibull_matches_fitdistrplus() -> None:
    fixture = load_fixture("aft_threshold_weibull3")
    time = np.asarray(fixture["data"], dtype=float)
    y = Surv.right(time, event=np.ones(time.shape[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = AFT("weibull", threshold=True).fit(y, np.zeros((time.shape[0], 0)))

    shape = 1.0 / aft.scale_
    scale = float(np.exp(aft.coef_[0]))

    assert shape == pytest.approx(fixture["shape"], rel=1e-2)
    assert scale == pytest.approx(fixture["scale"], rel=1e-2)
    assert aft.threshold_ == pytest.approx(fixture["thres"], rel=1e-2)
    assert aft.loglik_ == pytest.approx(fixture["loglik"], abs=1e-2)


@pytest.mark.rparity
def test_threshold_weibull_mps_close_to_mle_reference() -> None:
    # MPS isn't independently validated here (see module docstring), but it should still land
    # close to the MLE optimum (both estimators target the same underlying parameters) on this
    # well-conditioned (shape > 1) dataset.
    fixture = load_fixture("aft_threshold_weibull3")
    time = np.asarray(fixture["data"], dtype=float)
    y = Surv.right(time, event=np.ones(time.shape[0]))

    aft = gw.AFT("weibull", method="mps", threshold=True).fit(y, np.zeros((time.shape[0], 0)))

    assert (1.0 / aft.scale_) == pytest.approx(fixture["shape"], rel=0.1)
    assert aft.threshold_ == pytest.approx(fixture["thres"], rel=0.1)
