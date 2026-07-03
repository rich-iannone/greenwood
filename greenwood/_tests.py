"""Group comparison tests: the log-rank statistic and the G-rho (Fleming-Harrington) family.

`logrank_test` compares survival across two or more groups using the weighted log-rank
statistic. The weight is the Fleming-Harrington `S(t-)^rho * (1 - S(t-))^gamma` evaluated
on the pooled Kaplan-Meier estimate, so `rho=0, gamma=0` is the standard log-rank test and
`rho=1, gamma=0` is the Peto-Peto (Wilcoxon-type) test. This matches R's
`survival::survdiff` (which exposes the `rho` parameter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["logrank_test", "TestResult"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class TestResult:
    """The outcome of a group comparison test.

    Attributes
    ----------
    statistic
        The chi-square test statistic.
    df
        Degrees of freedom (number of groups minus one).
    p_value
        Upper-tail chi-square p-value.
    method
        Human-readable description of the test and its weights.
    observed, expected
        Weighted observed and expected event counts per group, keyed by group label.
    """

    statistic: float
    df: int
    p_value: float
    method: str
    observed: dict[Any, float] = field(default_factory=dict)
    expected: dict[Any, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TestResult(method={self.method!r}, statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.4g})"
        )


