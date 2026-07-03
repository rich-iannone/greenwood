"""R-parity validation of the risk-set / event-table kernel against `survfit`.

This proves the kernel matches R's `survival::survfit` tabulation (time, n.risk, n.event,
n.censor) on real datasets, including a left-truncation case. The Kaplan-Meier estimator
builds directly on this kernel.
"""

from __future__ import annotations

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
