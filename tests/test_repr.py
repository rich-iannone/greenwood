"""Tests for the R-style `__repr__` of the estimators.

Each fitted estimator should print an informative, deterministic summary (no memory
addresses), and each unfitted estimator should say so rather than erroring.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def y(lung):
    return Surv.right(lung["time"], event=(lung["status"] == 2))


@pytest.fixture(scope="module")
def mgus_cr():
    mg = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    return mg, Surv.multistate(etime, event=cause, states=("pcm", "death"))


def _clean(text: str) -> None:
    # No default object repr (memory address) leaked through.
    assert "object at 0x" not in text


def test_coxph_repr(lung, y) -> None:
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])
    text = repr(cox)
    _clean(text)
    assert repr(cox) == repr(cox)  # deterministic
    assert "Cox proportional hazards" in text
    assert "exp(coef)" in text
    assert "age" in text and "sex" in text
    assert "n = 228, events = 165" in text
    assert "Likelihood ratio test" in text


def test_coxph_robust_repr_notes_sandwich(lung, y) -> None:
    text = repr(gw.CoxPH().fit(y, lung[["age", "sex"]], robust=True))
    assert "robust (sandwich)" in text


def test_aft_repr(lung, y) -> None:
    text = repr(gw.AFT("weibull").fit(y, lung[["age", "sex"]]))
    _clean(text)
    assert "accelerated failure time" in text
    assert "dist='weibull'" in text
    assert "Scale =" in text
    assert "Log-likelihood =" in text


def test_kaplan_meier_repr(y) -> None:
    text = repr(gw.KaplanMeier().fit(y))
    _clean(text)
    assert "Kaplan-Meier" in text
    assert "median" in text
    assert "0.95LCL" in text
    assert "228" in text


def test_kaplan_meier_grouped_repr(lung, y) -> None:
    text = repr(gw.KaplanMeier().fit(y, by=lung["sex"]))
    _clean(text)
    # One row per stratum, labeled by the group value.
    assert "\n1 " in text or text.count("\n") >= 3
    assert "138" in text and "90" in text


def test_nelson_aalen_repr(y) -> None:
    text = repr(gw.NelsonAalen().fit(y))
    _clean(text)
    assert "Nelson-Aalen" in text
    assert "max cumhaz" in text


def test_aalen_johansen_repr(mgus_cr) -> None:
    _, ycr = mgus_cr
    text = repr(gw.AalenJohansen().fit(ycr))
    _clean(text)
    assert "cumulative incidence" in text
    assert "states: pcm, death" in text
    assert "final CIF" in text


def test_aalen_johansen_grouped_repr(mgus_cr) -> None:
    mg, ycr = mgus_cr
    text = repr(gw.AalenJohansen().fit(ycr, by=mg["sex"]))
    _clean(text)
    assert "strata: 2" in text


def test_fine_gray_repr(mgus_cr) -> None:
    mg, ycr = mgus_cr
    text = repr(gw.FineGray("pcm").fit(ycr, mg[["age", "sex"]]))
    _clean(text)
    assert "Fine-Gray subdistribution" in text
    assert "cause='pcm'" in text
    assert "robust (clustered)" in text


def test_multistate_repr(mgus_cr) -> None:
    mg, _ = mgus_cr
    start, stop, state, event = [], [], [], []
    for i in range(len(mg)):
        pt, ft = mg["ptime"][i], mg["futime"][i]
        progressed, died = mg["pstat"][i] == 1, mg["death"][i] == 1
        if progressed and pt < ft:
            start += [0, pt]
            stop += [pt, ft]
            state += ["mgus", "pcm"]
            event += ["pcm", "death" if died else None]
        else:
            start += [0]
            stop += [ft]
            state += ["mgus"]
            event += ["death" if died else ("pcm" if progressed else None)]
    rows = [(a, b, s, e) for a, b, s, e in zip(start, stop, state, event, strict=True) if b > a]
    start, stop, state, event = map(list, zip(*rows, strict=True))
    text = repr(gw.MultiState().fit(start, stop, state, event, states=("mgus", "pcm", "death")))
    _clean(text)
    assert "multi-state" in text
    assert "states: mgus, pcm, death" in text
    assert "final occupancy" in text


def test_unfitted_reprs_do_not_error() -> None:
    for text in (repr(gw.CoxPH()), repr(gw.AFT()), repr(gw.KaplanMeier()),
                 repr(gw.NelsonAalen()), repr(gw.AalenJohansen()),
                 repr(gw.FineGray("pcm")), repr(gw.MultiState())):
        assert "<unfitted>" in text
        _clean(text)
