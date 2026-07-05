"""Unit tests for competing-risks estimation (Aalen-Johansen CIF)."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AalenJohansen, Surv


def _simple_multistate() -> Surv:
    # Times 1..4, causes: pcm, death, pcm, censor.
    return Surv.multistate([1, 2, 3, 4], event=[1, 2, 1, 0], states=("pcm", "death"))


def test_cif_bounded_and_monotone() -> None:
    aj = AalenJohansen().fit(_simple_multistate())
    table = aj.to_dataframe()
    for cause in ("pcm", "death"):
        cif = table[table["cause"] == cause].sort_values("time")["estimate"].to_numpy()
        assert np.all(np.diff(cif) >= -1e-12)  # non-decreasing
        assert np.all((cif >= 0) & (cif <= 1))


def test_cifs_sum_to_complement_of_survival() -> None:
    # At the last time, sum of CIFs across causes = 1 - overall survival.
    y = _simple_multistate()
    table = AalenJohansen().fit(y).to_dataframe()
    last = table[table["time"] == table["time"].max()]
    total_cif = last["estimate"].sum()
    km = gw.KaplanMeier().fit(Surv.right(y.stop, event=y.event))
    assert total_cif == pytest.approx(1.0 - km.survival_[-1])


def test_requires_multistate() -> None:
    with pytest.raises(ValueError, match="multi-state"):
        AalenJohansen().fit(Surv.right([1, 2, 3], [1, 1, 1]))


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        AalenJohansen(conf_level=2.0)


def test_to_dataframe_columns() -> None:
    table = AalenJohansen().fit(_simple_multistate()).to_dataframe()
    assert list(table.columns) == [
        "cause",
        "time",
        "n_risk",
        "estimate",
        "std_error",
        "conf_low",
        "conf_high",
    ]


def test_grouped_has_strata_column() -> None:
    y = Surv.multistate([1, 2, 3, 4], event=[1, 2, 1, 2], states=("pcm", "death"))
    table = AalenJohansen().fit(y, by=["a", "a", "b", "b"]).to_dataframe()
    assert "strata" in table.columns
    assert set(table["strata"]) == {"a", "b"}


def test_group_length_checked() -> None:
    with pytest.raises(ValueError, match="same length"):
        AalenJohansen().fit(_simple_multistate(), by=["a", "b"])


# -- Fine-Gray -------------------------------------------------------------------


def _mgus2_cr():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    return df, gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))


def test_finegray_requires_multistate() -> None:
    from greenwood import FineGray

    with pytest.raises(ValueError, match="multi-state"):
        FineGray("pcm").fit(gw.Surv.right([1, 2, 3], [1, 1, 1]), np.zeros((3, 1)))


def test_finegray_unknown_cause() -> None:
    from greenwood import FineGray

    df, y = _mgus2_cr()
    with pytest.raises(ValueError, match="not one of the states"):
        FineGray("relapse").fit(y, df[["age"]])


def test_finegray_accepts_cause_by_code() -> None:
    from greenwood import FineGray

    df, y = _mgus2_cr()
    by_label = FineGray("pcm").fit(y, df[["age", "sex"]]).coef_
    by_code = FineGray(1).fit(y, df[["age", "sex"]]).coef_
    np.testing.assert_allclose(by_label, by_code)


def test_finegray_tidy_and_glance() -> None:
    from greenwood import FineGray

    df, y = _mgus2_cr()
    fg = FineGray("pcm").fit(y, df[["age", "sex"]])
    tidy = gw.tidy.tidy(fg, exponentiate=True)
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), fg.hazard_ratio_)
    assert gw.tidy.glance(fg).iloc[0]["nevent"] > 0


def test_finegray_length_mismatch() -> None:
    from greenwood import FineGray

    df, y = _mgus2_cr()
    with pytest.raises(ValueError, match="same number of rows"):
        FineGray("pcm").fit(y, df[["age"]].iloc[:-1])


# -- Multi-state -----------------------------------------------------------------


def test_multistate_illness_death_occupancy() -> None:
    from greenwood import MultiState

    # Two subjects: one mgus->pcm->death, one mgus->death directly.
    ms = MultiState().fit(
        start=[0, 5, 0],
        stop=[5, 8, 6],
        state=["mgus", "pcm", "mgus"],
        event=["pcm", "death", "death"],
        states=("mgus", "pcm", "death"),
    )
    table = ms.to_dataframe()
    # Occupancy probabilities sum to 1 at every time.
    row_sums = table[["mgus", "pcm", "death"]].sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0)
    # Everyone starts in mgus.
    assert table.iloc[0]["mgus"] <= 1.0 and table["death"].iloc[-1] > 0.0


def test_multistate_length_mismatch() -> None:
    from greenwood import MultiState

    with pytest.raises(ValueError, match="same length"):
        MultiState().fit(start=[0, 0], stop=[1, 2], state=["a", "a"], event=["b"])


def test_multistate_predict_step_function() -> None:
    from greenwood import MultiState

    ms = MultiState().fit(
        start=[0, 0], stop=[2, 4], state=["a", "a"], event=["b", "b"], states=("a", "b")
    )
    pred = ms.predict([0.0, 3.0, 5.0])
    assert list(pred.columns) == ["time", "a", "b"]
    np.testing.assert_allclose(pred[["a", "b"]].sum(axis=1).to_numpy(), 1.0)


def test_multistate_infers_states() -> None:
    from greenwood import MultiState

    ms = MultiState().fit(start=[0, 0], stop=[1, 2], state=["a", "a"], event=["b", None])
    assert ms.states_ == ("a", "b")
    assert list(ms.to_dataframe().columns) == ["time", "a", "b"]
