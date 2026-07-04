"""Sample size and power for the log-rank test (Schoenfeld's method).

For a two-group comparison under proportional hazards, Schoenfeld (1981, 1983) showed that
the power of the log-rank test depends on the data only through the total number of events.
The required number of events to detect a hazard ratio `HR` is

    d = (z_{1 - alpha/sides} + z_{power})^2 / (p * (1 - p) * (ln HR)^2),

where `p` is the fraction of subjects allocated to one group. These functions implement that
relationship: the events needed for a target power, the power achieved with a given number of
events, and the sample size needed given the probability that a subject has the event.
"""

from __future__ import annotations

import math

from scipy.stats import norm

__all__ = ["logrank_n_events", "logrank_power", "logrank_sample_size"]


def _check_common(hazard_ratio: float, alpha: float, allocation: float, sides: int) -> None:
    if hazard_ratio <= 0.0 or hazard_ratio == 1.0:
        raise ValueError("hazard_ratio must be positive and not equal to 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0.0 < allocation < 1.0:
        raise ValueError(f"allocation must be in (0, 1), got {allocation}.")
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2, got {sides}.")


def _exact_n_events(
    hazard_ratio: float, power: float, alpha: float, allocation: float, sides: int
) -> float:
    """The unrounded Schoenfeld number of events."""
    z = float(norm.ppf(1.0 - alpha / sides)) + float(norm.ppf(power))
    return z**2 / (allocation * (1.0 - allocation) * math.log(hazard_ratio) ** 2)


def logrank_n_events(
    hazard_ratio: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    allocation: float = 0.5,
    sides: int = 2,
) -> int:
    """Number of events needed for the log-rank test to reach a target power.

    Parameters
    ----------
    hazard_ratio
        The hazard ratio to detect (group 2 versus group 1). The result is symmetric in
        `HR` and `1 / HR`.
    power
        Target power (default 0.8).
    alpha
        Significance level (default 0.05).
    allocation
        Fraction of subjects in one group (default 0.5, a balanced design, which minimizes
        the events required).
    sides
        1 or 2 (default 2, a two-sided test).

    Returns
    -------
    int
        The required number of events, rounded up.

    Examples
    --------
    This is a Schoenfeld calculation: for a two-group comparison under proportional hazards the
    power of the log-rank test depends on the data only through the total number of events. How
    many events are needed to detect a hazard ratio of 0.5 with 90 percent power?

    ```{python}
    import greenwood as gw

    gw.logrank_n_events(hazard_ratio=0.5, power=0.9)
    ```

    The count is the number of events, not subjects. Tune the test with `alpha` (significance
    level), `allocation` (the fraction of subjects in one group; 0.5 is balanced and minimizes
    the events required), and `sides` (1 or 2).
    """
    _check_common(hazard_ratio, alpha, allocation, sides)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    return math.ceil(_exact_n_events(hazard_ratio, power, alpha, allocation, sides))


def logrank_power(
    hazard_ratio: float,
    n_events: float,
    *,
    alpha: float = 0.05,
    allocation: float = 0.5,
    sides: int = 2,
) -> float:
    """Power of the log-rank test given the number of events.

    The inverse of `logrank_n_events`: given `n_events` observed events, return the power to
    detect `hazard_ratio` at level `alpha`.

    Examples
    --------
    Given a study that will observe 60 events, what power does it have to detect a hazard ratio
    of 0.5?

    ```{python}
    import greenwood as gw

    gw.logrank_power(hazard_ratio=0.5, n_events=60)
    ```
    """
    _check_common(hazard_ratio, alpha, allocation, sides)
    if n_events <= 0:
        raise ValueError(f"n_events must be positive, got {n_events}.")
    z_alpha = float(norm.ppf(1.0 - alpha / sides))
    z_power = math.sqrt(n_events * allocation * (1.0 - allocation)) * abs(
        math.log(hazard_ratio)
    ) - z_alpha
    return float(norm.cdf(z_power))


def logrank_sample_size(
    hazard_ratio: float,
    prob_event: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    allocation: float = 0.5,
    sides: int = 2,
) -> int:
    """Total sample size needed for the log-rank test to reach a target power.

    Converts the required number of events (`logrank_n_events`) into subjects by dividing by
    `prob_event`, the overall probability that a subject has the event during the study (which
    you estimate from the expected survival, accrual, and follow-up).

    Examples
    --------
    How many subjects are needed to detect a hazard ratio of 0.5 with 90 percent power, if only
    about 40 percent of subjects are expected to have the event during the study?

    ```{python}
    import greenwood as gw

    gw.logrank_sample_size(hazard_ratio=0.5, prob_event=0.4, power=0.9)
    ```

    The result exceeds the event count from `logrank_n_events` because most subjects will be
    censored before the event is observed.
    """
    _check_common(hazard_ratio, alpha, allocation, sides)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if not 0.0 < prob_event <= 1.0:
        raise ValueError(f"prob_event must be in (0, 1], got {prob_event}.")
    events = _exact_n_events(hazard_ratio, power, alpha, allocation, sides)
    return math.ceil(events / prob_event)
