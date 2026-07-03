"""R-parity validation of the risk-set / event-table kernel against `survfit`.

This proves the kernel matches R's `survival::survfit` tabulation (time, n.risk, n.event,
n.censor) on real datasets, including a left-truncation case. The Kaplan-Meier estimator
builds directly on this kernel.
"""

from __future__ import annotations

from typing import Any

import pytest

import greenwood as gw
from greenwood import Surv, event_table

from ._r_parity import assert_allclose_to_r, load_fixture

pytestmark = pytest.mark.rparity


def _check_block(et: gw.EventTable, expected: dict[str, list[float]], label: str) -> None:
    assert_allclose_to_r(et.time, expected["time"], what=f"{label} time")
    assert_allclose_to_r(et.n_risk, expected["n_risk"], what=f"{label} n_risk")
    assert_allclose_to_r(et.n_event, expected["n_event"], what=f"{label} n_event")
    assert_allclose_to_r(et.n_censor, expected["n_censor"], what=f"{label} n_censor")


def test_lung_km_overall_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    # survival::lung codes status 1 = censored, 2 = dead.
    y = Surv.right(df["time"], event=(df["status"] == 2))
    et = event_table(y)
    _check_block(et, load_fixture("lung_km_overall")["overall"], "lung overall")


def test_lung_km_by_sex_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    et = event_table(y, group=df["sex"])
    fixture = load_fixture("lung_km_by_sex")

    assert et.strata is not None
    for level, block in fixture.items():
        mask = et.strata.astype(int).astype(str) == str(level)
        sub = gw.EventTable(
            time=et.time[mask],
            n_risk=et.n_risk[mask],
            n_event=et.n_event[mask],
            n_censor=et.n_censor[mask],
        )
        _check_block(sub, block, f"lung sex={level}")


def test_veteran_km_overall_matches_r() -> None:
    df = gw.data.load_dataset("veteran")
    y = Surv.right(df["time"], event=df["status"])
    et = event_table(y)
    _check_block(et, load_fixture("veteran_km_overall")["overall"], "veteran overall")


def test_counting_left_truncation_matches_r() -> None:
    fixture = load_fixture("counting_truncation")
    data = fixture["data"]
    y = Surv.counting(start=data["start"], stop=data["stop"], event=data["event"])
    et = event_table(y)
    _check_block(et, fixture["overall"], "counting truncation")


# -- Kaplan-Meier / Nelson-Aalen against survfit --------------------------------

_CONF = [("plain", "plain"), ("log", "log"), ("log-log", "loglog")]


def _check_km(km: gw.KaplanMeier, expected: dict[str, Any], label: str) -> None:
    assert_allclose_to_r(km.survival_, expected["surv"], what=f"{label} surv")
    assert_allclose_to_r(km.std_error_, expected["se"], what=f"{label} se(S)")
    assert_allclose_to_r(km.cumhaz_, expected["cumhaz"], what=f"{label} cumhaz")


def test_km_lung_overall_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]

    for conf_type, key in _CONF:
        km = gw.KaplanMeier(conf_type=conf_type).fit(y)
        _check_km(km, expected, f"lung/{conf_type}")
        assert_allclose_to_r(km.conf_low_, expected[f"lower_{key}"], what=f"lung {conf_type} lower")
        assert_allclose_to_r(
            km.conf_high_, expected[f"upper_{key}"], what=f"lung {conf_type} upper"
        )


def test_km_median_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]
    point, lower, upper = gw.KaplanMeier(conf_type="log").fit(y).median(ci=True)
    assert point == expected["median"]
    assert lower == expected["median_lower"]
    assert upper == expected["median_upper"]


def test_km_by_sex_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("km_lung_by_sex")
    km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
    for block in km._blocks:
        expected = fixture[str(block.label)]
        assert_allclose_to_r(block.surv, expected["surv"], what=f"sex={block.label} surv")
        assert_allclose_to_r(block.conf_low, expected["lower_loglog"], what="lower")
        assert_allclose_to_r(block.conf_high, expected["upper_loglog"], what="upper")


def test_km_veteran_overall_matches_r() -> None:
    df = gw.data.load_dataset("veteran")
    y = Surv.right(df["time"], event=df["status"])
    km = gw.KaplanMeier(conf_type="log").fit(y)
    _check_km(km, load_fixture("km_veteran_overall")["overall"], "veteran")


def test_nelson_aalen_matches_r() -> None:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]
    na = gw.NelsonAalen().fit(y)
    assert_allclose_to_r(na.cumhaz_, expected["cumhaz"], what="NA cumhaz")
    assert_allclose_to_r(na.std_error_**2, expected["cumhaz_var"], what="NA cumhaz var")
