"""Weibull diagnostic plots for assessing distributional fit and proportional hazards.

Plots Kaplan-Meier survival estimates on transformed axes that linearize a given parametric
distribution. If the data follows the specified distribution, the transformed points fall on a
straight line. For stratified fits, parallel lines indicate the proportional hazards assumption
holds between groups.

Supported transforms (controlled by `dist`):

- **Weibull** (default): `log(-log(S(t)))` vs `log(t)` (complementary log-log).
- **Log-normal**: `Phi_inv(1 - S(t))` vs `log(t)` (probit).
- **Log-logistic**: `log((1 - S(t)) / S(t))` vs `log(t)` (logit).

An optional parametric overlay from a fitted `AFT` model draws the theoretical line through the
empirical points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from .._backends import to_dataframe
from ._altair import _PALETTE, _SOLID
from ._shared import _strata_label

if TYPE_CHECKING:
    from .._nonparametric import KaplanMeier
    from .._parametric import AFT

__all__ = ["plot_weibull"]

_VALID_DISTS = frozenset({"weibull", "lognormal", "loglogistic"})

_Y_LABELS: dict[str, str] = {
    "weibull": "log(−log S(t))",
    "lognormal": "Normal quantile",
    "loglogistic": "log((1 − S(t)) / S(t))",
}

_X_LABEL = "log(time)"


def _transform_y(surv: Any, dist: str) -> Any:
    """Apply the linearizing y-axis transform for *dist* to survival values."""
    s = np.asarray(surv, dtype=float)
    if dist == "weibull":
        return np.log(-np.log(s))
    if dist == "lognormal":
        return norm.ppf(1.0 - s)
    # loglogistic
    return np.log((1.0 - s) / s)


def _weibull_columns(km: KaplanMeier, dist: str) -> dict[str, list[Any]]:
    """Build transformed KM scatter data from all blocks."""
    log_time: list[float] = []
    y: list[float] = []
    strata: list[str] = []

    for block in km._blocks:
        mask = (block.surv > 0.0) & (block.surv < 1.0)
        if not np.any(mask):
            continue
        t = block.time[mask]
        s = block.surv[mask]
        lt = np.log(t).tolist()
        yt = _transform_y(s, dist).tolist()
        label = _strata_label(block)
        log_time.extend(lt)
        y.extend(yt)
        strata.extend([label] * len(lt))

    return {"log_time": log_time, "y": y, "strata": strata}


def _parametric_line_columns(
    aft: AFT,
    log_time_range: tuple[float, float],
    dist: str,
    n_points: int = 200,
) -> dict[str, list[float]]:
    """Generate the theoretical parametric line from a fitted AFT."""
    from .._parametric import _log_density_survival

    mu = float(np.mean(aft._x @ aft.coef_))
    sigma = aft.scale_
    Q = getattr(aft, "Q_", None) or 0.0

    lt = np.linspace(log_time_range[0], log_time_range[1], n_points)
    z = (lt - mu) / sigma
    _, log_s = _log_density_survival(aft.dist, z, Q=Q)
    s = np.exp(log_s)

    keep = (s > 0.0) & (s < 1.0)
    lt_kept = lt[keep]
    s_kept = s[keep]
    y_vals = _transform_y(s_kept, dist)

    return {"log_time": lt_kept.tolist(), "y": y_vals.tolist()}


def plot_weibull(
    km: KaplanMeier,
    *,
    aft: Any | None = None,
    dist: str = "weibull",
    title: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    width: int = 500,
    height: int = 300,
) -> Any:
    r"""Plot Kaplan-Meier estimates on distribution-linearizing axes.

    Produces a diagnostic scatter plot of Kaplan-Meier survival estimates on transformed axes where
    data from the specified parametric distribution falls on a straight line. This is the classic
    "Weibull plot" (complementary log-log transform) generalized to other AFT distributions.

    For stratified Kaplan-Meier fits, each stratum is drawn as a separate colored series. Parallel
    lines indicate the proportional hazards assumption holds between groups.

    An optional fitted `AFT` model overlays the theoretical parametric line on the empirical points,
    showing goodness-of-fit visually. The overlay is available only for unstratified Kaplan-Meier
    fits.

    Parameters
    ----------
    km
        A fitted `KaplanMeier` object (stratified or unstratified).
    aft
        An optional fitted `AFT` model for a parametric overlay line. Only allowed with an
        unstratified `km`. The AFT's distribution must match the `dist` parameter (with
        `"exponential"` accepted when `dist="weibull"`).
    dist
        Distribution whose linearizing transform is applied: `"weibull"` (default), `"lognormal"`,
        or `"loglogistic"`.
    title
        Chart title. Defaults to `None` (no title).
    xlab
        X-axis label. Defaults to `"log(time)"`.
    ylab
        Y-axis label. Defaults to a distribution-specific label.
    width, height
        Chart dimensions in pixels (defaults 500 x 300).

    Returns
    -------
    altair.LayerChart
        An interactive Altair chart.

    Examples
    --------
    Check the Weibull assumption for lung cancer survival by sex:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    # Stratified KM where parallel lines suggest PH holds
    km = gw.KaplanMeier().fit(y, by=lung["sex"])
    gw.plot_weibull(km)
    ```

    Overlay a fitted Weibull AFT on an unstratified KM:

    ```{python}
    km_overall = gw.KaplanMeier().fit(y)
    aft = gw.AFT("weibull").fit(y, lung[["age", "sex"]])
    gw.plot_weibull(km_overall, aft=aft)
    ```
    """
    if dist not in _VALID_DISTS:
        raise ValueError(f"dist must be one of {sorted(_VALID_DISTS)}, got {dist!r}.")

    if aft is not None:
        if km._grouped:
            raise ValueError(
                "AFT overlay is only supported for unstratified Kaplan-Meier fits. "
                "Fit an unstratified KM (without by=) for the parametric overlay."
            )
        allowed = {"weibull", "exponential"} if dist == "weibull" else {dist}
        if aft.dist not in allowed:
            raise ValueError(f"AFT distribution {aft.dist!r} does not match plot dist={dist!r}.")

    if xlab is None:
        xlab = _X_LABEL
    if ylab is None:
        ylab = _Y_LABELS[dist]

    return _plot_weibull_altair(
        km,
        aft=aft,
        dist=dist,
        title=title,
        xlab=xlab,
        ylab=ylab,
        width=width,
        height=height,
    )


def _plot_weibull_altair(
    km: KaplanMeier,
    *,
    aft: Any | None,
    dist: str,
    title: str | None,
    xlab: str,
    ylab: str,
    width: int,
    height: int,
) -> Any:
    """Build the Altair chart for the Weibull diagnostic plot."""
    try:
        import altair as alt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Altair visualization requires altair. Install it with `pip install greenwood[altair]`."
        ) from exc

    km_data = _weibull_columns(km, dist)
    km_frame = to_dataframe(km_data)
    strata_names = sorted(set(km_data["strata"]))
    grouped = len(strata_names) > 1

    if grouped:
        color_scale = alt.Scale(
            domain=strata_names,
            range=list(_PALETTE)[: len(strata_names)],
        )
        color_enc = alt.Color("strata:N", title="Stratum", scale=color_scale)
    else:
        color_enc = alt.value(_SOLID)

    points = (
        alt.Chart(km_frame)
        .mark_circle(size=40, opacity=0.8)
        .encode(
            x=alt.X("log_time:Q", title=xlab),
            y=alt.Y("y:Q", title=ylab),
            color=color_enc,
            tooltip=[
                alt.Tooltip("strata:N", title="Stratum"),
                alt.Tooltip("log_time:Q", title=xlab, format=".3f"),
                alt.Tooltip("y:Q", title=ylab, format=".3f"),
            ],
        )
    )

    lines = (
        alt.Chart(km_frame)
        .mark_line(opacity=0.5)
        .encode(
            x=alt.X("log_time:Q"),
            y=alt.Y("y:Q"),
            color=color_enc,
            order="log_time:Q",
        )
    )

    layers: list[Any] = [lines, points]

    if aft is not None:
        log_times = km_data["log_time"]
        if log_times:
            lt_min, lt_max = min(log_times), max(log_times)
            pad = (lt_max - lt_min) * 0.05
            aft_data = _parametric_line_columns(aft, (lt_min - pad, lt_max + pad), dist)
            if aft_data["log_time"]:
                aft_frame = to_dataframe(aft_data)
                aft_line = (
                    alt.Chart(aft_frame)
                    .mark_line(strokeDash=[6, 3], color="#333333", strokeWidth=1.5)
                    .encode(
                        x=alt.X("log_time:Q"),
                        y=alt.Y("y:Q"),
                        order="log_time:Q",
                    )
                )
                layers.append(aft_line)

    chart = alt.layer(*layers).properties(width=width, height=height).interactive()
    if title is not None:
        chart = chart.properties(title=title)
    return chart
