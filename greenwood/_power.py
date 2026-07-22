r"""Sample size and power for the log-rank test (Schoenfeld's method).

For a two-group comparison under proportional hazards, Schoenfeld (1981, 1983) showed that
the power of the log-rank test depends on the data only through the total number of events.
The required number of events to detect a hazard ratio $\mathrm{HR}$ is

$$
d = \frac{(z_{1 - \alpha/\mathrm{sides}} + z_{\mathrm{power}})^2}
{p \, (1 - p) \, (\ln \mathrm{HR})^2},
$$

where $p$ is the fraction of subjects allocated to one group. These functions implement that
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
    r"""Number of events needed for the log-rank test to reach a target power.

    Computes the minimum number of events required for a log-rank test to achieve a specified
    power when detecting a given hazard ratio in a two-group survival study. This is the
    foundational calculation in trial planning.

    Under the proportional hazards assumption (Schoenfeld, 1981), power depends only on the
    total number of events, not on follow-up duration, censoring distribution, or sample size
    per se. This makes this calculation practical as it tells you "how many events do you need?"
    and you plan your study to observe that many events through appropriate enrollment and
    follow-up.

    Use this when you want to know the required event count; use `logrank_sample_size` to
    convert events to total enrollment given an expected event probability.

    Parameters
    ----------
    hazard_ratio
        The hazard ratio to detect (group 2 versus group 1). Can be < 1 (protective effect,
        better survival) or > 1 (harmful effect, worse survival). The result is symmetric:
        HR=0.5 and HR=2.0 require the same number of events. Typical ranges:

        - 0.5: 50% hazard reduction (strong effect)
        - 0.67: 33% hazard reduction (moderate effect)
        - 0.8: 20% hazard reduction (small effect)

    power
        Target statistical power (default 0.8, i.e., 80%). The probability of detecting the
        effect if it truly exists. Standard choice is 0.8; use 0.9 or 0.95 for higher
        confidence (requires more events).
    alpha
        Significance level or Type-I error rate (default 0.05). The probability of rejecting
        the null hypothesis if it's true. Use 0.05 for the conventional two-sided test with
        p < 0.05 threshold.
    allocation
        Fraction of subjects in one group (default 0.5, a balanced design). For example:

        - 0.5: Equal allocation (balanced, minimizes total events needed)
        - 0.33 or 0.67: 1:2 allocation (unbalanced)
        - 0.25 or 0.75: 1:3 allocation (more unbalanced)

        Unbalanced allocation increases the events needed; balanced (0.5) is optimal for a
        given total sample size.

    sides
        1 (one-sided) or 2 (two-sided, default). A two-sided test checks for differences in
        either direction (one-sided checks only one direction). One-sided tests require fewer
        events for the same power but are appropriate only when direction is known in advance.

    Returns
    -------
    int
        Minimum number of events (rounded up), required across all groups combined.

    Details
    -------
    **Schoenfeld's formula**: The exact number of events is

    $$
    d = \frac{(z_{1 - \alpha/\mathrm{sides}} + z_{\mathrm{power}})^2}
    {p \, (1 - p) \, (\ln \mathrm{HR})^2},
    $$

    where:

    - $z_{1 - \alpha/\mathrm{sides}}$: critical value for significance level and sides
    - $z_{\mathrm{power}}$: critical value for desired power
    - $p$: allocation fraction (default 0.5)
    - $\ln(\mathrm{HR})$: natural log of hazard ratio

    Balanced allocation ($p = 0.5$) minimizes the denominator and thus minimizes events needed.

    **Practical use**: Once you know the required event count, estimate sample size by dividing
    by the expected event rate: `n_subjects = ceil(n_events / prob_event)`. Then design your
    study (enrollment, duration, follow-up) to observe that many events.

    Examples
    --------
    For a trial detecting a hazard ratio of 0.5 (50% hazard reduction) with conventional 80%
    power and two-sided significance level 0.05, how many events are needed?

    ```{python}
    import greenwood as gw

    # Compute events needed to detect a 50% hazard reduction
    gw.logrank_n_events(hazard_ratio=0.5)
    ```

    Repeat for higher power (90%) to see the increase:

    ```{python}
    # Increase power to 90%
    gw.logrank_n_events(hazard_ratio=0.5, power=0.9)
    ```

    Explore sensitivity to effect size. Smaller effects (HR closer to 1) require more events:

    ```{python}
    # Show how required events increase as the effect size shrinks
    for hr in [0.5, 0.6, 0.7, 0.8]:
        events = gw.logrank_n_events(hazard_ratio=hr)
        print(f"HR {hr}: {events} events needed")
    ```

    Use `sides=1` for one-sided testing (higher power, fewer events, but assumes direction):

    ```{python}
    # Use a one-sided test to reduce the required event count
    gw.logrank_n_events(hazard_ratio=0.5, sides=1)
    ```

    Use `allocation=0.33` for unbalanced assignment (e.g., 1 treated per 2 control):

    ```{python}
    # Use 1:2 allocation (one-third of subjects in group 1)
    gw.logrank_n_events(hazard_ratio=0.5, allocation=0.33)
    ```

    After determining event count, convert to sample size using expected event probability.
    For instance, if 40% of subjects are expected to have events, divide by 0.4:

    ```{python}
    # Convert event count to total enrollment given 40% event probability
    events = gw.logrank_n_events(hazard_ratio=0.5)
    subjects = gw.logrank_sample_size(hazard_ratio=0.5, prob_event=0.4)
    print(f"Events needed: {events}")
    print(f"Subjects needed: {subjects}")
    print(f"Ratio: {subjects / events:.1f}")
    ```
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
    r"""Power of the log-rank test given the number of observed events.

    Computes the statistical power to detect a specified hazard ratio using the log-rank test,
    given a fixed number of observed events in a two-group survival study. This is the inverse
    calculation of `logrank_n_events`: instead of finding the events needed for a target power,
    this function finds the power achieved with a given number of events.

    Power depends on three factors:

    1. **Number of events** (`n_events`): More events → higher power
    2. **Effect size** (`hazard_ratio`): Larger effects (HR far from 1.0) → higher power
    3. **Significance level** (`alpha`): More stringent (smaller alpha) → lower power

    Under the proportional hazards assumption, power depends only on the total event count, not
    the follow-up duration, censoring rate, or sample size separately. This makes it a practical
    tool for updating power calculations as events accumulate during a trial.

    Parameters
    ----------
    hazard_ratio
        The hazard ratio to detect (group 2 versus group 1). Can be < 1 (group 2 has lower
        hazard/better survival) or > 1 (group 2 has higher hazard/worse survival). The result
        is symmetric: HR=0.5 and HR=2.0 give the same power.
    n_events
        Total number of observed events. Must be positive. Power increases with more events;
        even small trials can have high power if many events occur.
    alpha
        Significance level (Type-I error rate, default 0.05). The probability of rejecting the
        null hypothesis when it's true. Use `alpha=0.05` for two-sided tests with p < 0.05
        threshold.
    allocation
        Fraction of subjects in one group (default 0.5, a balanced design). For unbalanced
        designs (e.g., 0.3, 0.7), power decreases; balanced allocation minimizes the total
        sample size needed for a target power.
    sides
        1 (one-sided) or 2 (two-sided, default). One-sided tests have higher power but test
        directional hypotheses only. Two-sided tests are standard but require more events for
        the same power.

    Returns
    -------
    float
        Statistical power, a value between 0 and 1. Power of 0.8 (80%) is conventional in many
        fields; higher power (0.9, 0.95) requires more events.

    Details
    -------
    **Schoenfeld's formula**: Under proportional hazards, the power of the log-rank test is

    $$
    \mathrm{Power} = \Phi\!\left(\sqrt{d \, p \, (1-p)} \; |\ln(\mathrm{HR})|
    - z_{1-\alpha/\mathrm{sides}}\right),
    $$

    where $d$ is the number of events, $p$ is the allocation fraction, and $\Phi$ is the
    cumulative normal distribution function. This formula is exact under proportional hazards
    and asymptotically valid for finite samples.

    **Practical use**: During a running trial, as events accumulate, you can use this function
    to assess interim power. If interim power is low despite many events, the effect size may
    be smaller than anticipated.

    Examples
    --------
    A study expects to observe 60 events over its follow-up period. What power does it have
    to detect a hazard ratio of 0.5 (50% hazard reduction)?

    ```{python}
    import greenwood as gw

    # Compute power to detect HR=0.5 with 60 events
    gw.logrank_power(hazard_ratio=0.5, n_events=60)
    ```

    This power (~0.9) is typical for a well-designed trial. Lower power suggests more events
    are needed, or the effect size is smaller than assumed. Compute power for different
    effect sizes to understand study sensitivity:

    ```{python}
    # Show how power decreases as the effect size shrinks
    for hr in [0.5, 0.6, 0.7, 0.8]:
        power = gw.logrank_power(hazard_ratio=hr, n_events=60)
        print(f"HR {hr}: power = {power:.2%}")
    ```

    Use `sides=1` for a one-sided test (higher power, but assumes direction is known):

    ```{python}
    # Compute one-sided power for the same scenario
    gw.logrank_power(hazard_ratio=0.5, n_events=60, sides=1)
    ```

    Use unbalanced allocation if one group is larger. Power decreases with imbalance:

    ```{python}
    # Show the power loss from unbalanced allocation
    gw.logrank_power(hazard_ratio=0.5, n_events=60, allocation=0.3)
    ```
    """
    _check_common(hazard_ratio, alpha, allocation, sides)
    if n_events <= 0:
        raise ValueError(f"n_events must be positive, got {n_events}.")
    z_alpha = float(norm.ppf(1.0 - alpha / sides))
    z_power = (
        math.sqrt(n_events * allocation * (1.0 - allocation)) * abs(math.log(hazard_ratio))
        - z_alpha
    )
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
    r"""Total sample size needed for the log-rank test to reach a target power.

    Computes the number of subjects required to observe enough events for a log-rank test to
    achieve a target power when detecting a specified hazard ratio. This function combines two
    calculations:

    1. First, it computes the required number of events using `logrank_n_events` (Schoenfeld's
       formula under proportional hazards).
    2. Then, it converts events to subjects using the expected probability that a subject
       experiences the event during follow-up.

    **Workflow**: Start here to plan study size. You provide the expected effect size (hazard
    ratio), the fraction of subjects expected to have the event (based on baseline hazard,
    accrual, and follow-up time), and your desired power. The result is the total enrollment
    needed.

    Parameters
    ----------
    hazard_ratio
        The hazard ratio to detect (group 2 versus group 1). Smaller HR (e.g., 0.5 = 50%
        hazard reduction) requires fewer subjects for a given power; larger HR (e.g., 0.8 =
        20% hazard reduction) requires more.
    prob_event
        Probability (or fraction) that a subject experiences the event (death, hospitalization,
        etc.) during the study. Range: (0, 1]. Typical values depend on the condition and
        follow-up duration:

        - Rare disease (annual incidence 1%): 0.01-0.05 per year of follow-up
        - Common condition (annual incidence 20%): 0.2 per year
        - Study design: shorter follow-up → lower prob_event; longer follow-up → higher

        This is usually estimated from historical data, Kaplan-Meier curves, or clinical
        judgment. Use sensitivity analysis (try 0.3, 0.4, 0.5) if uncertain.

    power
        Target statistical power (default 0.8, i.e., 80%). Interpretation: the probability
        that the study detects the effect if it truly exists. Common choices:

        - 0.80 (80%): Conventional, implies 20% Type-II error rate
        - 0.90 (90%): Higher confidence, requires more subjects
        - 0.95 (95%): Stringent, requires many more subjects

    alpha
        Significance level (Type-I error rate, default 0.05). Probability of rejecting the
        null hypothesis if it's true. Use 0.05 for two-sided tests with p < 0.05 threshold;
        use 0.01 for stricter control or 0.10 for more exploratory studies.
    allocation
        Fraction of subjects in one group (default 0.5, balanced design). For unbalanced
        allocation (e.g., control-to-treatment ratio 2:1), use `allocation=0.33`. Unbalanced
        allocation increases total sample size needed; use only if required by design or
        logistics.
    sides
        1 (one-sided) or 2 (two-sided, default). One-sided tests have higher power and require
        fewer subjects but test directional hypotheses only. Two-sided tests are standard but
        require more enrollment.

    Returns
    -------
    int
        Total number of subjects (enrollment) needed, rounded up. This is the total across all
        groups.

    Details
    -------
    **Relationship between events and subjects**:

        n_subjects = ceil(n_events / prob_event)

    More subjects are needed when:

    - `prob_event` is low (most subjects censored before event): e.g., 50% power, 50 events
      needed, but if only 25% get the event, you need 200 subjects.
    - The effect size is smaller: smaller HR requires more events
    - Power is higher: 90% power requires more events than 80%
    - Design is unbalanced: 3:1 allocation needs more subjects than 1:1

    **Estimating prob_event**: Use Kaplan-Meier curves from historical data or prior studies,
    or calculate from baseline rates:
    $\text{prob\_event} \approx 1 - \exp(-\lambda_0 \times t_{\text{follow-up}})$.

    **Sensitivity analysis**: If prob_event is uncertain, compute sample size for a range of
    values (e.g., 0.3 to 0.5) to understand robustness.

    Examples
    --------
    A trial aims to detect a hazard ratio of 0.5 (50% hazard reduction) with 90% power. Based
    on historical data, about 40% of subjects are expected to have the event during follow-up.
    How many subjects must be enrolled?

    ```{python}
    import greenwood as gw

    # Compute enrollment needed for 90% power with 40% event probability
    gw.logrank_sample_size(hazard_ratio=0.5, prob_event=0.4, power=0.9)
    ```

    This sample size (~350) is much larger than the event count from `logrank_n_events`
    (~140) because most subjects will be censored before the event occurs.

    Perform sensitivity analysis for uncertain event probability. How does sample size change
    if only 30% or 50% of subjects have events?

    ```{python}
    # Vary event probability to see how enrollment requirements change
    for prob in [0.3, 0.4, 0.5]:
        n = gw.logrank_sample_size(hazard_ratio=0.5, prob_event=prob, power=0.9)
        print(f"prob_event={prob}: n={n} subjects, {int(prob * n)} events")
    ```

    Compare sample size for different effect sizes (smaller HR → fewer subjects):

    ```{python}
    # Vary effect size to see how enrollment requirements change
    for hr in [0.5, 0.6, 0.7]:
        n = gw.logrank_sample_size(hazard_ratio=hr, prob_event=0.4, power=0.9)
        print(f"HR {hr}: n={n} subjects")
    ```

    Use higher power (0.95) if you want to be very confident the effect is detected:

    ```{python}
    # Increase power to 95% for greater confidence
    gw.logrank_sample_size(hazard_ratio=0.5, prob_event=0.4, power=0.95)
    ```

    Use unbalanced allocation (e.g., 2:1 control:treatment) if logistics require it:

    ```{python}
    # Use 1:2 allocation to see the sample size increase
    gw.logrank_sample_size(hazard_ratio=0.5, prob_event=0.4, power=0.9, allocation=1/3)
    ```
    """
    _check_common(hazard_ratio, alpha, allocation, sides)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if not 0.0 < prob_event <= 1.0:
        raise ValueError(f"prob_event must be in (0, 1], got {prob_event}.")
    events = _exact_n_events(hazard_ratio, power, alpha, allocation, sides)
    return math.ceil(events / prob_event)
