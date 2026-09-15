"""Synthetic data generators for survival analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._surv import Surv

__all__ = ["simulate_competing_risks", "SimulatedCompetingRisks"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class SimulatedCompetingRisks:
    """Result of :func:`simulate_competing_risks`.

    Attributes
    ----------
    surv
        A multi-state `Surv` response ready for use with `AalenJohansen`, `FineGray`, or any
        competing-risks metric.
    covariates
        Covariate matrix of shape `(n, n_covariates)`, or `None` when `n_covariates=0`.
    latent_times
        Per-cause latent event times of shape `(n, n_causes)` before the competing-risks minimum is
        taken. Useful for constructing oracle predictions in tests and benchmarks.
    censoring_time
        The censoring time for each subject, shape `(n,)`. The observed time is
        `min(min(latent_times, axis=1), censoring_time)`.
    """

    surv: Surv
    covariates: Array | None
    latent_times: Array
    censoring_time: Array


def simulate_competing_risks(
    n: int = 500,
    n_causes: int = 2,
    *,
    shapes: tuple[float, ...] | None = None,
    scales: tuple[float, ...] | None = None,
    censoring_scale: float | None = 1.5,
    n_covariates: int = 0,
    coef: Any | None = None,
    states: tuple[str, ...] | None = None,
    seed: int | None = None,
) -> SimulatedCompetingRisks:
    r"""Generate synthetic competing-risks data from a latent-Weibull model.

    Each cause has its own Weibull distribution with specified shape and scale parameters. For each
    subject, a latent event time is drawn independently for every cause, and the cause with the
    shortest latent time is the one observed. An independent censoring time is drawn from a
    Weibull(1, scale) distribution (i.e., exponential). If the censoring time is shortest, the
    subject is censored.

    When covariates are requested, cause-specific proportional hazards are applied: the Weibull
    scale for cause $k$ is adjusted as
    $b_k \cdot \exp(-\mathbf{x}^\top \boldsymbol{\beta}_k / a_k)$, where $a_k$ and $b_k$ are the
    shape and scale for cause $k$. Covariates are drawn from a standard normal distribution.

    Parameters
    ----------
    n
        Number of subjects (default 500).
    n_causes
        Number of competing event types (default 2).
    shapes
        Weibull shape parameter for each cause. Length must equal `n_causes`. Defaults to
        `(1.0, 1.5, ...)` cycling through `[1.0, 1.5, 0.8]`. A shape of 1.0 gives an exponential
        (constant hazard). Values above 1 give increasing hazard. Values below 1 give decreasing
        hazard.
    scales
        Weibull scale parameter for each cause. Length must equal `n_causes`. Defaults to 10.0 for
        every cause.
    censoring_scale
        Controls censoring intensity. The censoring time is drawn from an exponential distribution
        with scale equal to `censoring_scale` times the mean latent event time. Larger values
        produce less censoring. Set to `None` or `0` to disable censoring entirely. Default is 1.5,
        which gives roughly 20-30% censoring depending on the shape/scale configuration.
    n_covariates
        Number of covariates to generate (default 0). When positive, standard-normal covariates are
        drawn and cause-specific proportional-hazards effects are applied.
    coef
        Coefficient matrix of shape `(n_covariates, n_causes)` for the proportional-hazards effect
        on each cause. When `None` (default), coefficients are drawn from Uniform(-0.5, 0.5).Ignored
        when `n_covariates=0`.
    states
        Names for each cause (e.g., `("relapse", "death")`). Length must equal `n_causes`. Defaults
        to `("cause1", "cause2", ...)`.
    seed
        Random seed for reproducibility.

    Returns
    -------
    SimulatedCompetingRisks
        A dataclass with `surv` (the multi-state `Surv` response), `covariates` (the covariate
        matrix or `None`), `latent_times` (per-cause latent times), and `censoring_time`.

    Examples
    --------
    Generate a simple two-cause competing-risks dataset and fit an Aalen-Johansen
    estimator:

    ```{python}
    import greenwood as gw

    sim = gw.simulate_competing_risks(n=300, n_causes=2, seed=42)
    sim.surv
    ```

    ```{python}
    aj = gw.AalenJohansen().fit(sim.surv)
    aj
    ```

    Generate data with covariates for a Fine-Gray model:

    ```{python}
    sim = gw.simulate_competing_risks(
        n=500,
        n_causes=2,
        n_covariates=3,
        states=("relapse", "death"),
        seed=99,
    )
    fg = gw.FineGray("relapse").fit(sim.surv, sim.covariates)
    gw.tidy(fg, format="polars")
    ```

    Control the event-time distribution and censoring rate:

    ```{python}
    import numpy as np

    # Cause 1: increasing hazard (shape > 1), Cause 2: constant hazard
    sim = gw.simulate_competing_risks(
        n=1000,
        n_causes=2,
        shapes=(2.0, 1.0),
        scales=(15.0, 10.0),
        censoring_scale=0.8,  # heavier censoring
        seed=7,
    )
    # Check the censoring fraction
    frac_censored = (sim.surv.status == 0).mean()
    print(f"Censored: {frac_censored:.1%}")
    ```
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    if n_causes < 1:
        raise ValueError(f"n_causes must be at least 1, got {n_causes}.")

    rng = np.random.default_rng(seed)

    # -- defaults --
    default_shapes_cycle = [1.0, 1.5, 0.8]
    if shapes is None:
        shapes = tuple(default_shapes_cycle[k % len(default_shapes_cycle)] for k in range(n_causes))
    if len(shapes) != n_causes:
        raise ValueError(f"shapes must have length n_causes={n_causes}, got {len(shapes)}.")
    if any(s <= 0 for s in shapes):
        raise ValueError("All shape parameters must be positive.")

    if scales is None:
        scales = tuple(10.0 for _ in range(n_causes))
    if len(scales) != n_causes:
        raise ValueError(f"scales must have length n_causes={n_causes}, got {len(scales)}.")
    if any(s <= 0 for s in scales):
        raise ValueError("All scale parameters must be positive.")

    if states is None:
        states = tuple(f"cause{k + 1}" for k in range(n_causes))
    if len(states) != n_causes:
        raise ValueError(f"states must have length n_causes={n_causes}, got {len(states)}.")

    # -- covariates and coefficients --
    x_mat: Array | None = None
    coef_arr: Array | None = None
    if n_covariates > 0:
        x_mat = rng.standard_normal((n, n_covariates))
        if coef is not None:
            coef_arr = np.asarray(coef, dtype=float)
            if coef_arr.shape != (n_covariates, n_causes):
                raise ValueError(
                    f"coef must have shape (n_covariates, n_causes) = "
                    f"({n_covariates}, {n_causes}), got {coef_arr.shape}."
                )
        else:
            coef_arr = rng.uniform(-0.5, 0.5, size=(n_covariates, n_causes))

    # -- latent event times --
    latent = np.empty((n, n_causes))
    for k in range(n_causes):
        a = shapes[k]
        b = scales[k]
        if x_mat is not None and coef_arr is not None:
            lp: Array = x_mat @ coef_arr[:, k]
            b_subj = b * np.exp(-lp / a)
        else:
            b_subj = np.full(n, b)
        latent[:, k] = b_subj * rng.weibull(a, size=n)

    # -- observed event: cause with shortest latent time --
    event_cause = latent.argmin(axis=1) + 1  # 1-based cause codes
    event_time = latent.min(axis=1)

    # -- censoring --
    if censoring_scale is not None and censoring_scale > 0:
        mean_event_time = float(event_time.mean())
        cens_rate = censoring_scale * mean_event_time
        censoring_time = rng.exponential(cens_rate, size=n)
        censored = censoring_time < event_time
        observed_time = np.where(censored, censoring_time, event_time)
        observed_status = np.where(censored, 0, event_cause)
    else:
        censoring_time = np.full(n, np.inf)
        observed_time = event_time
        observed_status = event_cause

    surv = Surv.multistate(
        observed_time,
        event=observed_status.astype(int),
        states=states,
    )

    return SimulatedCompetingRisks(
        surv=surv,
        covariates=x_mat,
        latent_times=latent,
        censoring_time=censoring_time,
    )
