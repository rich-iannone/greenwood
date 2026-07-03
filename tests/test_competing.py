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
