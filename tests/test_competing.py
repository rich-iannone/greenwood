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


# -- Fine-Gray to_frame ----------------------------------------------------------


class TestFineGrayToFrame:
    @pytest.fixture(scope="class")
    def fg(self):  # type: ignore[no-untyped-def]
        from greenwood import FineGray

        df, y = _mgus2_cr()
        return FineGray("pcm").fit(y, df[["age", "sex"]])

    def test_columns(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")
        expected = [
            "term",
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "conf_low",
            "conf_high",
        ]

        assert list(df.columns) == expected

    def test_n_rows_equals_n_terms(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")

        assert df.shape[0] == len(fg.term_names_)

    def test_terms_match(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")

        assert list(df["term"]) == list(fg.term_names_)

    def test_default_estimate_is_log_scale(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")

        np.testing.assert_allclose(df["estimate"].values, fg.coef_)

    def test_exponentiate_returns_hazard_ratios(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(exponentiate=True, format="pandas")

        np.testing.assert_allclose(df["estimate"].values, fg.hazard_ratio_)

    def test_exponentiate_transforms_ci(self, fg) -> None:  # type: ignore[no-untyped-def]
        df_log = fg.to_frame(format="pandas")
        df_exp = fg.to_frame(exponentiate=True, format="pandas")

        np.testing.assert_allclose(df_exp["conf_low"].values, np.exp(df_log["conf_low"].values))
        np.testing.assert_allclose(df_exp["conf_high"].values, np.exp(df_log["conf_high"].values))

    def test_std_error_same_either_scale(self, fg) -> None:  # type: ignore[no-untyped-def]
        df_log = fg.to_frame(format="pandas")
        df_exp = fg.to_frame(exponentiate=True, format="pandas")

        np.testing.assert_allclose(df_log["std_error"].values, df_exp["std_error"].values)

    def test_ci_brackets_estimate(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")

        assert (df["conf_low"] <= df["estimate"]).all()
        assert (df["conf_high"] >= df["estimate"]).all()

    def test_format_polars(self, fg) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        df = fg.to_frame(format="polars")

        assert isinstance(df, pl.DataFrame)
        assert df.columns == [
            "term",
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "conf_low",
            "conf_high",
        ]

    def test_format_default(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame()

        assert df.shape[0] == len(fg.term_names_)

    def test_matches_tidy(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(format="pandas")
        t = gw.tidy(fg, format="pandas")

        assert df.equals(t)

    def test_exponentiate_matches_tidy_exponentiate(self, fg) -> None:  # type: ignore[no-untyped-def]
        df = fg.to_frame(exponentiate=True, format="pandas")
        t = gw.tidy(fg, exponentiate=True, format="pandas")
        np.testing.assert_allclose(df["estimate"].values, t["estimate"].values)


# -- AalenJohansen tidy / glance -------------------------------------------------


class TestAalenJohansenTidy:
    def test_tidy_columns(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        t = gw.tidy(aj, format="pandas")
        expected = ["cause", "time", "n_risk", "estimate", "std_error", "conf_low", "conf_high"]
        assert list(t.columns) == expected

    def test_tidy_matches_to_frame(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        t = gw.tidy(aj, format="pandas")
        f = aj.to_frame(format="pandas")
        assert t.equals(f)

    def test_tidy_stratified_has_strata(self) -> None:
        y = Surv.multistate([1, 2, 3, 4], event=[1, 2, 1, 2], states=("pcm", "death"))
        aj = AalenJohansen().fit(y, by=["a", "a", "b", "b"])
        t = gw.tidy(aj, format="pandas")
        assert "strata" in t.columns
        assert set(t["strata"]) == {"a", "b"}

    def test_tidy_format_polars(self) -> None:
        import polars as pl

        aj = AalenJohansen().fit(_simple_multistate())
        t = gw.tidy(aj, format="polars")
        assert isinstance(t, pl.DataFrame)

    def test_tidy_causes_present(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        t = gw.tidy(aj, format="pandas")
        assert set(t["cause"]) == {"pcm", "death"}


class TestAalenJohansenGlance:
    def test_glance_columns(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        g = gw.glance(aj, format="pandas")
        assert list(g.columns) == ["n_causes", "causes"]
        assert g.shape[0] == 1

    def test_glance_values(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        g = gw.glance(aj, format="pandas")
        assert g["n_causes"].iloc[0] == 2
        assert g["causes"].iloc[0] == "pcm, death"

    def test_glance_stratified(self) -> None:
        y = Surv.multistate([1, 2, 3, 4], event=[1, 2, 1, 2], states=("pcm", "death"))
        aj = AalenJohansen().fit(y, by=["a", "a", "b", "b"])
        g = gw.glance(aj, format="pandas")
        assert "strata" in g.columns
        assert g.shape[0] == 2
        assert list(g["strata"]) == ["a", "b"]

    def test_glance_format_polars(self) -> None:
        import polars as pl

        aj = AalenJohansen().fit(_simple_multistate())
        g = gw.glance(aj, format="polars")
        assert isinstance(g, pl.DataFrame)


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


# -- MultiState tidy / glance ----------------------------------------------------


def _simple_multistate_model():  # type: ignore[no-untyped-def]
    from greenwood import MultiState

    return MultiState().fit(
        start=[0, 5, 0],
        stop=[5, 8, 6],
        state=["mgus", "pcm", "mgus"],
        event=["pcm", "death", "death"],
        states=("mgus", "pcm", "death"),
    )


class TestMultiStateTidy:
    def test_tidy_columns(self) -> None:
        ms = _simple_multistate_model()
        t = gw.tidy(ms, format="pandas")

        assert list(t.columns) == ["time", "mgus", "pcm", "death"]

    def test_tidy_matches_to_frame(self) -> None:
        ms = _simple_multistate_model()
        t = gw.tidy(ms, format="pandas")
        f = ms.to_frame(format="pandas")

        assert t.equals(f)

    def test_tidy_format_polars(self) -> None:
        import polars as pl

        ms = _simple_multistate_model()
        t = gw.tidy(ms, format="polars")

        assert isinstance(t, pl.DataFrame)

    def test_tidy_occupancy_sums_to_one(self) -> None:
        ms = _simple_multistate_model()
        t = gw.tidy(ms, format="pandas")
        row_sums = t[["mgus", "pcm", "death"]].sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, 1.0)


class TestMultiStateGlance:
    def test_glance_columns(self) -> None:
        ms = _simple_multistate_model()
        g = gw.glance(ms, format="pandas")

        assert list(g.columns) == ["n_states", "states", "n_times"]
        assert g.shape[0] == 1

    def test_glance_values(self) -> None:
        ms = _simple_multistate_model()
        g = gw.glance(ms, format="pandas")

        assert g["n_states"].iloc[0] == 3
        assert g["states"].iloc[0] == "mgus, pcm, death"
        assert g["n_times"].iloc[0] == ms.time_.shape[0]

    def test_glance_format_polars(self) -> None:
        import polars as pl

        ms = _simple_multistate_model()
        g = gw.glance(ms, format="polars")

        assert isinstance(g, pl.DataFrame)


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


# ---------------------------------------------------------------------------
# AalenJohansen to_frame expanded tests
# ---------------------------------------------------------------------------


class TestAalenJohansenToFrame:
    def test_format_polars(self) -> None:
        import polars as pl

        aj = AalenJohansen().fit(_simple_multistate())
        df = aj.to_frame(format="polars")
        assert isinstance(df, pl.DataFrame)
        assert "cause" in df.columns
        assert "estimate" in df.columns

    def test_cif_monotone_per_cause(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        df = aj.to_frame(format="pandas")
        for cause in df["cause"].unique():
            cif = df.loc[df["cause"] == cause, "estimate"].to_numpy()
            assert np.all(np.diff(cif) >= -1e-12)

    def test_cif_bounded_zero_one(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        df = aj.to_frame(format="pandas")
        assert (df["estimate"] >= -1e-12).all()
        assert (df["estimate"] <= 1.0 + 1e-12).all()

    def test_n_risk_positive(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        df = aj.to_frame(format="pandas")
        assert (df["n_risk"] > 0).all()

    def test_std_error_nonnegative(self) -> None:
        aj = AalenJohansen().fit(_simple_multistate())
        df = aj.to_frame(format="pandas")
        assert (df["std_error"] >= 0).all()

    def test_real_data_cifs_sum_bounded(self) -> None:
        _, y = _mgus2_cr()
        aj = AalenJohansen().fit(y)
        df = aj.to_frame(format="pandas")
        for t in df["time"].unique():
            total = df.loc[df["time"] == t, "estimate"].sum()
            assert total <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# MultiState predict expanded tests
# ---------------------------------------------------------------------------


class TestMultiStatePredict:
    @pytest.fixture(scope="class")
    def ms(self):  # type: ignore[no-untyped-def]
        from greenwood import MultiState

        return MultiState().fit(
            start=[0, 5, 0],
            stop=[5, 8, 6],
            state=["mgus", "pcm", "mgus"],
            event=["pcm", "death", "death"],
            states=("mgus", "pcm", "death"),
        )

    def test_predict_format_polars(self, ms) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        pred = ms.predict([0.0, 3.0, 5.0], format="polars")
        assert isinstance(pred, pl.DataFrame)
        assert "time" in pred.columns

    def test_predict_occupancy_sums_to_one(self, ms) -> None:  # type: ignore[no-untyped-def]
        pred = ms.predict([0.0, 3.0, 5.0, 8.0], format="pandas")
        state_cols = [c for c in pred.columns if c != "time"]
        row_sums = pred[state_cols].sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, 1.0)

    def test_predict_at_time_zero(self, ms) -> None:  # type: ignore[no-untyped-def]
        pred = ms.predict([0.0], format="pandas")
        assert pred["mgus"].iloc[0] == pytest.approx(1.0)
        assert pred["pcm"].iloc[0] == pytest.approx(0.0)
        assert pred["death"].iloc[0] == pytest.approx(0.0)

    def test_predict_absorbing_state_nondecreasing(self, ms) -> None:  # type: ignore[no-untyped-def]
        pred = ms.predict([0.0, 3.0, 5.0, 6.0, 8.0], format="pandas")
        death_vals = pred["death"].to_numpy()
        assert np.all(np.diff(death_vals) >= -1e-12)

    def test_predict_columns_match_states(self, ms) -> None:  # type: ignore[no-untyped-def]
        pred = ms.predict([1.0], format="pandas")
        assert list(pred.columns) == ["time", "mgus", "pcm", "death"]


# -- FineGray predict_cumulative_incidence --------------------------------------


class TestFineGrayPredict:
    @pytest.fixture(scope="class")
    def fg(self):  # type: ignore[no-untyped-def]
        from greenwood import FineGray

        df, y = _mgus2_cr()
        return FineGray("pcm").fit(y, df[["age", "sex"]])

    @pytest.fixture(scope="class")
    def df_y(self):  # type: ignore[no-untyped-def]
        return _mgus2_cr()

    def test_predict_lp_shape(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        lp = fg.predict(df[["age", "sex"]], type="lp")
        assert lp.shape[0] == len(df)

    def test_predict_risk_positive(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        risk = fg.predict(df[["age", "sex"]], type="risk")
        assert np.all(risk > 0)

    def test_predict_lp_default_uses_training(self, fg) -> None:  # type: ignore[no-untyped-def]
        lp = fg.predict()
        assert lp.shape[0] == fg.n_

    def test_predict_invalid_type(self, fg) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="Unknown predict type"):
            fg.predict(type="bogus")

    def test_predict_cif_columns(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = fg.predict_cumulative_incidence(df[["age", "sex"]][:3], format="pandas")
        assert "time" in cif.columns
        assert "subject_1" in cif.columns
        assert "subject_3" in cif.columns

    def test_predict_cif_bounded(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = fg.predict_cumulative_incidence(df[["age", "sex"]][:5], format="pandas")
        vals = cif.drop(columns="time").to_numpy()
        assert np.all(vals >= -1e-12)
        assert np.all(vals <= 1.0 + 1e-12)

    def test_predict_cif_monotone(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = fg.predict_cumulative_incidence(df[["age", "sex"]][:3], format="pandas")
        for col in ["subject_1", "subject_2", "subject_3"]:
            assert np.all(np.diff(cif[col].to_numpy()) >= -1e-12)

    def test_predict_cif_at_specific_times(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = fg.predict_cumulative_incidence(
            df[["age", "sex"]][:2], times=[100, 200, 300], format="pandas"
        )
        assert list(cif["time"]) == [100.0, 200.0, 300.0]
        assert cif.shape[0] == 3

    def test_predict_cif_time_zero(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = fg.predict_cumulative_incidence(df[["age", "sex"]][:2], times=[0.0], format="pandas")
        vals = cif.drop(columns="time").to_numpy()
        np.testing.assert_allclose(vals, 0.0)

    def test_predict_cif_default_training_data(self, fg) -> None:  # type: ignore[no-untyped-def]
        cif = fg.predict_cumulative_incidence(format="pandas")
        assert cif.shape[1] == fg.n_ + 1  # time + one col per subject

    def test_predict_cif_higher_risk_higher_cif(self, fg) -> None:  # type: ignore[no-untyped-def]
        lp = fg.predict()
        high_idx = int(np.argmax(lp))
        low_idx = int(np.argmin(lp))
        cif = fg.predict_cumulative_incidence(format="pandas")
        cif_high = cif[f"subject_{high_idx + 1}"].iloc[-1]
        cif_low = cif[f"subject_{low_idx + 1}"].iloc[-1]
        assert cif_high > cif_low

    def test_predict_cif_polars(self, fg, df_y) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        df, _ = df_y
        cif = fg.predict_cumulative_incidence(df[["age", "sex"]][:2], times=[100], format="polars")
        assert isinstance(cif, pl.DataFrame)


# -- PenalizedFineGray ---------------------------------------------------------


class TestPenalizedFineGray:
    @pytest.fixture(scope="class")
    def df_y(self):  # type: ignore[no-untyped-def]
        return _mgus2_cr()

    @pytest.fixture(scope="class")
    def pfg(self, df_y):  # type: ignore[no-untyped-def]
        df, y = df_y
        return gw.PenalizedFineGray("pcm", penalizer=0.01, l1_ratio=1.0).fit(y, df[["age", "sex"]])

    def test_coef_shape(self, pfg) -> None:  # type: ignore[no-untyped-def]
        assert pfg.coef_.shape == (2,)

    def test_n_event_positive(self, pfg) -> None:  # type: ignore[no-untyped-def]
        assert pfg.n_event_ > 0

    def test_repr_fitted(self, pfg) -> None:  # type: ignore[no-untyped-def]
        r = repr(pfg)
        assert "PenalizedFineGray" in r
        assert "nonzero" in r

    def test_repr_unfitted(self) -> None:
        r = repr(gw.PenalizedFineGray("pcm"))
        assert "unfitted" in r

    def test_requires_multistate(self) -> None:
        with pytest.raises(ValueError, match="multi-state"):
            gw.PenalizedFineGray("pcm").fit(gw.Surv.right([1, 2, 3], [1, 1, 1]), np.zeros((3, 1)))

    def test_unknown_cause(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        with pytest.raises(ValueError, match="not one of the states"):
            gw.PenalizedFineGray("relapse").fit(y, df[["age"]])

    def test_length_mismatch(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        with pytest.raises(ValueError, match="same number of rows"):
            gw.PenalizedFineGray("pcm").fit(y, df[["age"]].iloc[:-1])

    def test_invalid_penalizer(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gw.PenalizedFineGray("pcm", penalizer=-1.0)

    def test_invalid_l1_ratio(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            gw.PenalizedFineGray("pcm", l1_ratio=2.0)

    def test_to_frame_columns(self, pfg) -> None:  # type: ignore[no-untyped-def]
        df = pfg.to_frame(format="pandas")
        assert list(df.columns) == ["term", "estimate", "hazard_ratio"]

    def test_tidy_and_glance(self, pfg) -> None:  # type: ignore[no-untyped-def]
        t = gw.tidy(pfg, format="pandas")
        assert t.shape[0] == 2
        g = gw.glance(pfg, format="pandas")
        assert "n_nonzero" in g.columns

    def test_predict_lp(self, pfg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        lp = pfg.predict(df[["age", "sex"]], type="lp")
        assert lp.shape[0] == len(df)

    def test_predict_risk(self, pfg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        risk = pfg.predict(df[["age", "sex"]], type="risk")
        assert np.all(risk > 0)

    def test_predict_cif_bounded(self, pfg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = pfg.predict_cumulative_incidence(df[["age", "sex"]][:5], format="pandas")
        vals = cif.drop(columns="time").to_numpy()
        assert np.all(vals >= -1e-12)
        assert np.all(vals <= 1.0 + 1e-12)

    def test_predict_cif_monotone(self, pfg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = pfg.predict_cumulative_incidence(df[["age", "sex"]][:3], format="pandas")
        for col in ["subject_1", "subject_2", "subject_3"]:
            assert np.all(np.diff(cif[col].to_numpy()) >= -1e-12)

    def test_predict_cif_at_specific_times(self, pfg, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        cif = pfg.predict_cumulative_incidence(
            df[["age", "sex"]][:2], times=[100, 200], format="pandas"
        )
        assert list(cif["time"]) == [100.0, 200.0]

    def test_heavy_penalty_shrinks_to_zero(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        pfg_heavy = gw.PenalizedFineGray("pcm", penalizer=10.0, l1_ratio=1.0).fit(
            y, df[["age", "sex"]]
        )
        assert np.allclose(pfg_heavy.coef_, 0.0, atol=1e-3)

    def test_zero_penalty_matches_unpenalized(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        pfg0 = gw.PenalizedFineGray("pcm", penalizer=0.0, l1_ratio=0.5).fit(y, df[["age", "sex"]])
        fg = gw.FineGray("pcm").fit(y, df[["age", "sex"]])
        np.testing.assert_allclose(pfg0.coef_, fg.coef_, atol=1e-4)

    def test_ridge_vs_lasso_sparsity(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        ridge = gw.PenalizedFineGray("pcm", penalizer=0.05, l1_ratio=0.0).fit(y, df[["age", "sex"]])
        lasso = gw.PenalizedFineGray("pcm", penalizer=0.05, l1_ratio=1.0).fit(y, df[["age", "sex"]])
        # Ridge should keep all nonzero; lasso may zero some
        assert np.count_nonzero(ridge.coef_) >= np.count_nonzero(lasso.coef_)

    def test_cause_by_integer(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        by_label = gw.PenalizedFineGray("pcm", penalizer=0.01).fit(y, df[["age", "sex"]]).coef_
        by_code = gw.PenalizedFineGray(1, penalizer=0.01).fit(y, df[["age", "sex"]]).coef_
        np.testing.assert_allclose(by_label, by_code)


# -- CauseSpecificCox ----------------------------------------------------------


class TestCauseSpecificCox:
    @pytest.fixture(scope="class")
    def df_y(self):  # type: ignore[no-untyped-def]
        return _mgus2_cr()

    @pytest.fixture(scope="class")
    def csc(self, df_y):  # type: ignore[no-untyped-def]
        df, y = df_y
        return gw.CauseSpecificCox("pcm").fit(y, df[["age", "sex"]])

    def test_matches_manual_recode(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        csc = gw.CauseSpecificCox("pcm").fit(y, df[["age", "sex"]])
        manual_event = (y.status == 1).astype(int)
        manual = gw.CoxPH().fit(gw.Surv.right(y.stop, event=manual_event), df[["age", "sex"]])
        np.testing.assert_allclose(csc.coef_, manual.coef_)

    def test_repr_fitted(self, csc) -> None:  # type: ignore[no-untyped-def]
        r = repr(csc)
        assert "CauseSpecificCox" in r
        assert "pcm" in r

    def test_repr_unfitted(self) -> None:
        r = repr(gw.CauseSpecificCox("pcm"))
        assert "unfitted" in r

    def test_requires_multistate(self) -> None:
        with pytest.raises(ValueError, match="multi-state"):
            gw.CauseSpecificCox("pcm").fit(gw.Surv.right([1, 2, 3], [1, 1, 1]), np.zeros((3, 1)))

    def test_unknown_cause(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        with pytest.raises(ValueError, match="not one of the states"):
            gw.CauseSpecificCox("relapse").fit(y, df[["age"]])

    def test_cause_by_integer(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        by_label = gw.CauseSpecificCox("pcm").fit(y, df[["age", "sex"]]).coef_
        by_code = gw.CauseSpecificCox(1).fit(y, df[["age", "sex"]]).coef_
        np.testing.assert_allclose(by_label, by_code)

    def test_tidy_and_glance(self, csc) -> None:  # type: ignore[no-untyped-def]
        t = gw.tidy(csc, exponentiate=True, format="pandas")
        np.testing.assert_allclose(t["estimate"].to_numpy(), csc.hazard_ratio_)
        g = gw.glance(csc, format="pandas")
        assert g.iloc[0]["nevent"] > 0

    def test_predict_lp(self, csc, df_y) -> None:  # type: ignore[no-untyped-def]
        df, _ = df_y
        lp = csc.predict(df[["age", "sex"]][:5], type="lp")
        assert lp.shape == (5,)

    def test_predict_survival(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        x = df[["age"]].copy()
        x["male"] = (df["sex"] == "M").astype(int)
        csc = gw.CauseSpecificCox("pcm").fit(y, x)
        surv = csc.predict(x[:2], type="survival", times=[100, 200], format="pandas")
        assert "time" in surv.columns
        assert surv.shape[0] == 2
        vals = surv.drop(columns="time").to_numpy()
        assert np.all((vals >= 0) & (vals <= 1))

    def test_n_event_matches_target_cause(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        csc = gw.CauseSpecificCox("pcm").fit(y, df[["age", "sex"]])
        assert csc.n_event_ == int((y.status == 1).sum())

    def test_death_cause(self, df_y) -> None:  # type: ignore[no-untyped-def]
        df, y = df_y
        csc = gw.CauseSpecificCox("death").fit(y, df[["age", "sex"]])
        assert csc.n_event_ == int((y.status == 2).sum())
        assert "death" in repr(csc)

    def test_delegates_to_frame(self, csc) -> None:  # type: ignore[no-untyped-def]
        df = csc.to_frame(format="pandas")
        assert "term" in df.columns

    def test_attribute_error_unfitted(self) -> None:
        csc = gw.CauseSpecificCox("pcm")
        with pytest.raises(AttributeError):
            _ = csc.coef_

    def test_concordance(self, csc) -> None:  # type: ignore[no-untyped-def]
        assert 0.0 <= csc.concordance() <= 1.0
