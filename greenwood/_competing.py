"""Competing-risks estimation: the Aalen-Johansen cumulative incidence function.

For competing risks (each subject starts in one state and makes a single transition to one
of several absorbing causes), the cumulative incidence function (CIF) for cause `k` is

    CIF_k(t) = sum_{t_i <= t} S(t_i^-) * d_{ki} / n_i,

where `S` is the all-cause Kaplan-Meier survival, `d_{ki}` the cause-`k` events at `t_i`, and
`n_i` the number at risk. The standard error uses the Aalen (Marubini-Valsecchi)
delta-method estimator. Both are validated to tolerance against R's `survfit`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["AalenJohansen"]

Array = npt.NDArray[Any]


def _cif_block(
    exit_: Array, status: Array, causes: list[int], z: float
) -> dict[int, dict[str, Array]]:
    """Cumulative incidence, delta-method SE, and CI for each cause in one group."""
    times = np.unique(exit_)
    n_risk = np.array([float((exit_ >= t).sum()) for t in times])
    d_any = np.array([float(((exit_ == t) & (status > 0)).sum()) for t in times])

    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_left = np.concatenate(([1.0], surv[:-1]))

    out: dict[int, dict[str, Array]] = {}
    for cause in causes:
        d_k = np.array([float(((exit_ == t) & (status == cause)).sum()) for t in times])
        cif = np.cumsum(surv_left * d_k / n_risk)

        # Aalen (Marubini-Valsecchi) delta-method variance via cumulative sums.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(n_risk > d_any, d_any / (n_risk * (n_risk - d_any)), 0.0)
        b = surv_left**2 * (n_risk - d_k) / n_risk * d_k / n_risk**2
        c = surv_left * d_k / n_risk**2
        c_a = np.cumsum(a)
        c_ac = np.cumsum(a * cif)
        c_ac2 = np.cumsum(a * cif**2)
        c_b = np.cumsum(b)
        c_c = np.cumsum(c)
        c_cc = np.cumsum(c * cif)
        var = cif**2 * c_a - 2 * cif * c_ac + c_ac2 + c_b - 2 * cif * c_c + 2 * c_cc
        se = np.sqrt(np.clip(var, 0.0, None))

        out[cause] = {
            "time": times,
            "n_risk": n_risk,
            "estimate": cif,
            "std_error": se,
            "conf_low": np.clip(cif - z * se, 0.0, 1.0),
            "conf_high": np.clip(cif + z * se, 0.0, 1.0),
        }
    return out


class AalenJohansen:
    """Aalen-Johansen estimator of cumulative incidence functions for competing risks.

    Parameters
    ----------
    conf_level
        Confidence level for the (Wald) confidence intervals (default 0.95).

    Notes
    -----
    Call `fit(surv, by=...)` with a multi-state `Surv` response (built with
    `Surv.multistate`, where `event` codes are 0 for censoring and `1..K` for the competing
    causes). Results are a tidy frame via `to_dataframe` with one row per stratum, cause, and
    time.
    """

    def __init__(self, *, conf_level: float = 0.95) -> None:
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_level = conf_level

    def fit(self, surv: Surv, *, by: Any = None) -> AalenJohansen:
        """Fit cumulative incidence functions to a competing-risks `Surv` response."""
        if not surv.is_multistate:
            raise ValueError(
                "AalenJohansen needs a multi-state response; build it with Surv.multistate "
                "(use KaplanMeier for a single event type)."
            )
        if surv.is_truncated:
            raise NotImplementedError(
                "Left truncation is not yet supported for cumulative incidence."
            )

        assert surv.states is not None
        self.states_ = surv.states
        causes = list(range(1, len(surv.states) + 1))
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        exit_ = surv.stop
        status = surv.status

        if by is None:
            self._grouped = False
            self._blocks = {None: _cif_block(exit_, status, causes, z)}
        else:
            from ._surv import _to_1d_array

            labels = _to_1d_array(by, dtype=object)
            if labels.shape[0] != surv.n:
                raise ValueError("`by` must have the same length as the response.")
            self._grouped = True
            self._blocks = {}
            for level in dict.fromkeys(labels.tolist()):
                mask = labels == level
                self._blocks[level] = _cif_block(exit_[mask], status[mask], causes, z)
        self._causes = causes
        return self

    def to_dataframe(self, backend: str = "pandas") -> Any:
        """Return a tidy frame: [strata,] cause, time, n_risk, estimate, std_error, CI."""
        cols: dict[str, list[Any]] = {
            k: []
            for k in (
                "strata",
                "cause",
                "time",
                "n_risk",
                "estimate",
                "std_error",
                "conf_low",
                "conf_high",
            )
        }
        for label, block in self._blocks.items():
            for cause in self._causes:
                data = block[cause]
                m = data["time"].shape[0]
                cols["strata"].extend([label] * m)
                cols["cause"].extend([self.states_[cause - 1]] * m)
                for key in ("time", "n_risk", "estimate", "std_error", "conf_low", "conf_high"):
                    cols[key].extend(data[key].tolist())
        if not self._grouped:
            cols.pop("strata")

        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")
