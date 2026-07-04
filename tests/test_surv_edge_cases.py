"""Edge-case and validation-path coverage for the Surv response."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import CensoringType, Surv
from greenwood._surv import _to_1d_array


def test_to_1d_array_none_and_shape() -> None:
    with pytest.raises(ValueError, match="array-like"):
        _to_1d_array(None)
    with pytest.raises(ValueError, match="1-D"):
        _to_1d_array(np.zeros((2, 2)))


def test_to_1d_array_non_series_fallback() -> None:
    # A range is not an ndarray/list/tuple and not a narwhals series: the TypeError
    # fallback coerces it with np.asarray.
    out = _to_1d_array(range(3))
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])


def test_stop_time_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        Surv.right([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        Surv.right([1.0, -1.0])


def test_event_indicator_validation() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Surv.right([1, 2], event=[1, np.nan])
    with pytest.raises(ValueError, match="boolean or 0/1"):
        Surv.right([1, 2, 3], event=[1, 2, 1])  # R's 1/2 coding must be converted


def test_negative_status_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        Surv(type=CensoringType.RIGHT, stop=np.array([1.0, 2.0]), status=np.array([-1, 1]))


def test_status_exceeds_state_count() -> None:
    with pytest.raises(ValueError, match="exceeds the number of event states"):
        Surv.multistate([1, 2, 3], event=[1, 2, 3], states=("a", "b"))


def test_counting_start_validation() -> None:
    with pytest.raises(ValueError, match="`start` and `stop`"):
        Surv.counting(start=[0.0], stop=[5.0, 6.0], event=[1, 1])
    with pytest.raises(ValueError, match="start` times must be finite"):
        Surv.counting(start=[np.inf, 0.0], stop=[5.0, 6.0])
    with pytest.raises(ValueError, match="strictly less"):
        Surv.counting(start=[5.0], stop=[5.0])


def test_interval_validation() -> None:
    with pytest.raises(ValueError, match="`lower` and `upper`"):
        Surv.interval(lower=[1.0, 2.0, 3.0], upper=[2.0, 3.0])
    with pytest.raises(ValueError, match="lower` must be <="):
        Surv.interval(lower=[5.0], upper=[2.0])
    # The post-init length check (reached when building Surv directly).
    with pytest.raises(ValueError, match="`lower` and `stop`"):
        Surv(
            type=CensoringType.INTERVAL,
            stop=np.array([1.0, 2.0]),
            status=np.array([2, 2]),
            lower=np.array([1.0]),
        )


def test_weight_validation() -> None:
    with pytest.raises(ValueError, match="`weights` and `stop`"):
        Surv.right([1, 2], weights=[1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        Surv.right([1, 2], weights=[1.0, 0.0])


def test_len_and_repr_variants() -> None:
    assert len(Surv.right([1, 2, 3])) == 3
    trunc = repr(Surv.counting(start=[1, 2], stop=[5, 6], event=[1, 1]))
    assert "truncated" in trunc
    ms = repr(Surv.multistate([1, 2, 3], event=[1, 2, 0], states=("relapse", "death")))
    assert "states=" in ms


def test_as_dataframe_backends() -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0], weights=[1.0, 2.0])
    pandas_df = y.as_dataframe("pandas")
    assert isinstance(pandas_df, pd.DataFrame)
    assert {"start", "stop", "status", "weight"} <= set(pandas_df.columns)
    assert isinstance(y.as_dataframe("polars"), pl.DataFrame)
    with pytest.raises(ValueError, match="Unknown backend"):
        y.as_dataframe("numpy")
