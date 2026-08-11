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
    table = aj.to_frame(format="pandas")
    for cause in ("pcm", "death"):
        cif = table[table["cause"] == cause].sort_values("time")["estimate"].to_numpy()
        assert np.all(np.diff(cif) >= -1e-12)  # non-decreasing
        assert np.all((cif >= 0) & (cif <= 1))


def test_cifs_sum_to_complement_of_survival() -> None:
    # At the last time, sum of CIFs across causes = 1 - overall survival.
    y = _simple_multistate()
    table = AalenJohansen().fit(y).to_frame(format="pandas")
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


def test_invalid_conf_type() -> None:
    with pytest.raises(ValueError, match="conf_type"):
        AalenJohansen(conf_type="bogus")


def test_conf_type_plain_brackets_estimate() -> None:
    aj = AalenJohansen(conf_type="plain").fit(_simple_multistate())
    table = aj.to_frame(format="pandas")
    assert np.all(table["conf_low"].to_numpy() <= table["estimate"].to_numpy() + 1e-12)
    assert np.all(table["estimate"].to_numpy() <= table["conf_high"].to_numpy() + 1e-12)
    assert np.all((table["conf_low"] >= 0) & (table["conf_high"] <= 1))


def test_conf_type_log_brackets_estimate() -> None:
    aj = AalenJohansen(conf_type="log").fit(_simple_multistate())
    table = aj.to_frame(format="pandas")
    finite = ~table["conf_low"].isna()
    assert np.all(
        table.loc[finite, "conf_low"].to_numpy() <= table.loc[finite, "estimate"].to_numpy() + 1e-12
    )
    assert np.all(
        table.loc[finite, "estimate"].to_numpy()
        <= table.loc[finite, "conf_high"].to_numpy() + 1e-12
    )
    assert np.all((table.loc[finite, "conf_low"] >= 0) & (table.loc[finite, "conf_high"] <= 1))


def test_conf_type_loglog_brackets_estimate() -> None:
    aj = AalenJohansen(conf_type="log-log").fit(_simple_multistate())
    table = aj.to_frame(format="pandas")
    finite = ~table["conf_low"].isna()
    assert np.all(
        table.loc[finite, "conf_low"].to_numpy() <= table.loc[finite, "estimate"].to_numpy() + 1e-12
    )
    assert np.all(
        table.loc[finite, "estimate"].to_numpy()
        <= table.loc[finite, "conf_high"].to_numpy() + 1e-12
    )
    assert np.all((table.loc[finite, "conf_low"] >= 0) & (table.loc[finite, "conf_high"] <= 1))


def test_to_pandas_columns() -> None:
    table = AalenJohansen().fit(_simple_multistate()).to_frame(format="pandas")
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
    table = AalenJohansen().fit(y, by=["a", "a", "b", "b"]).to_frame(format="pandas")
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
    tidy = gw.tidy(fg, exponentiate=True)
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), fg.hazard_ratio_)
    assert gw.glance(fg, format="pandas").iloc[0]["nevent"] > 0


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
    table = ms.to_frame(format="pandas")
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
    pred = ms.predict([0.0, 3.0, 5.0], format="pandas")
    assert list(pred.columns) == ["time", "a", "b"]
    np.testing.assert_allclose(pred[["a", "b"]].sum(axis=1).to_numpy(), 1.0)


def test_aalen_johansen_rejects_truncated() -> None:
    y_trunc = Surv.multistate(
        [5, 6, 7, 8],
        event=[1, 2, 1, 0],
        states=("pcm", "death"),
        start=[1, 2, 1, 2],
    )
    with pytest.raises(NotImplementedError, match="Left truncation"):
        AalenJohansen().fit(y_trunc)


def test_multistate_infers_states() -> None:
    from greenwood import MultiState

    ms = MultiState().fit(start=[0, 0], stop=[1, 2], state=["a", "a"], event=["b", None])
    assert ms.states_ == ("a", "b")
    assert list(ms.to_frame(format="pandas").columns) == ["time", "a", "b"]


# -- Gray's test -----------------------------------------------------------------


def test_grays_test_basic() -> None:
    df, y = _mgus2_cr()
    result = gw.grays_test(y, group=df["sex"], cause="pcm")
    assert result.statistic >= 0
    assert 0 <= result.p_value <= 1
    assert result.df == 1
    assert result.method == "Gray's test (cause='pcm')"
    assert len(result.observed) == 2
    assert len(result.expected) == 2


def test_grays_test_cause_by_code() -> None:
    df, y = _mgus2_cr()
    by_label = gw.grays_test(y, group=df["sex"], cause="pcm")
    by_code = gw.grays_test(y, group=df["sex"], cause=1)
    assert by_label.statistic == pytest.approx(by_code.statistic)
    assert by_label.p_value == pytest.approx(by_code.p_value)


def test_grays_test_death_cause() -> None:
    df, y = _mgus2_cr()
    result = gw.grays_test(y, group=df["sex"], cause="death")
    assert result.statistic >= 0
    assert result.method == "Gray's test (cause='death')"


def test_grays_test_requires_multistate() -> None:
    with pytest.raises(ValueError, match="multi-state"):
        gw.grays_test(Surv.right([1, 2, 3], [1, 1, 1]), group=[1, 1, 2])


def test_grays_test_invalid_cause() -> None:
    _, y = _mgus2_cr()
    with pytest.raises(ValueError, match="not a valid"):
        gw.grays_test(y, group=np.ones(y.n, dtype=int), cause=99)


def test_grays_test_unknown_cause_label() -> None:
    _, y = _mgus2_cr()
    with pytest.raises(ValueError, match="not one of the states"):
        gw.grays_test(y, group=np.ones(y.n, dtype=int), cause="relapse")


def test_grays_test_group_length_mismatch() -> None:
    _, y = _mgus2_cr()
    with pytest.raises(ValueError, match="same length"):
        gw.grays_test(y, group=[1, 2])


def test_grays_test_single_group() -> None:
    _, y = _mgus2_cr()
    with pytest.raises(ValueError, match="at least two"):
        gw.grays_test(y, group=np.ones(y.n, dtype=int))


def test_grays_test_no_target_events() -> None:
    y = Surv.multistate([1, 2, 3, 4], event=[0, 2, 0, 2], states=("pcm", "death"))
    with pytest.raises(ValueError, match="No events"):
        gw.grays_test(y, group=[1, 1, 2, 2], cause="pcm")


def test_grays_test_observed_expected_sum() -> None:
    df, y = _mgus2_cr()
    result = gw.grays_test(y, group=df["sex"], cause="pcm")
    total_obs = sum(result.observed.values())
    total_exp = sum(result.expected.values())
    assert total_obs == pytest.approx(total_exp, rel=1e-10)


def test_grays_test_three_groups() -> None:
    df, y = _mgus2_cr()
    age_group = np.where(df["age"] < 60, "young", np.where(df["age"] < 70, "mid", "old"))
    result = gw.grays_test(y, group=age_group, cause="pcm")
    assert result.df == 2
    assert len(result.observed) == 3
