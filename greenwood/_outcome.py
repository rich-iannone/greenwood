"""`Outcome`: a reusable, data-free description of a survival response.

An `Outcome` names the columns of a response and declares its event encoding, without holding any
data. It is bound to a frame later, either explicitly with `Outcome.bind()` or by an estimator's
`fit(outcome, ..., data=df)`. Binding at `fit()` reads every column the model needs from the one
frame and drops incomplete rows once, so the response, covariates, and per-row labels (`by`,
`strata`, `cluster`, `weights`) always stay aligned.

The same description can also be written as an R-style formula response, such as
`"Surv(time, status == 2)"`, and a full formula (`"Surv(time, status == 2) ~ age + sex"`) can be
passed straight to `fit()`. Formulas are parsed into an `Outcome`, so both spellings share one code
path. Parsing uses Python's `ast` module and only accepts column names and literal values. Nothing
in a formula is evaluated.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt

from ._ingest import Duration, as_1d, missing_mask, select_columns
from ._surv import Surv

__all__ = ["Outcome"]

Array = npt.NDArray[Any]

# Argument order of each constructor, used for the repr and for binding.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "right": ("time", "event", "weights"),
    "left": ("time", "event", "weights"),
    "counting": ("start", "stop", "event", "weights"),
    "interval": ("lower", "upper", "weights"),
    "multistate": ("time", "event", "start", "weights"),
}


def _freeze(value: Any) -> Any:
    """Make an encoding value immutable (lists and sets become tuples)."""
    if isinstance(value, (list, set, frozenset)):
        return tuple(value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return value


# Arguments that hold times, where a `duration()` may stand in for a column.
_TIME_ARGUMENTS = re.compile(r"time\d*|start|stop|lower|upper|censor_at")


def _column(arg: str, value: Any, *, optional: bool = False) -> str | Duration | None:
    if value is None and optional:
        return None
    if isinstance(value, Duration) and _TIME_ARGUMENTS.fullmatch(arg):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"`Outcome` describes columns by name, so `{arg}` must be a column name. Got "
            f"{type(value).__name__}. To build a response from values, use `Surv` directly."
        )
    return value


@dataclass(frozen=True)
class Outcome:
    """A reusable description of a survival response: column names plus an event encoding.

    An `Outcome` says which columns hold the response and how to read the event, but holds no data.
    Pass it to any estimator's `fit()` together with `data=`. The estimator reads every column it
    needs from that one frame, drops rows with missing values once (as R's `na.omit` does), and
    fits. Because the description is made of column names and values rather than DataFrame
    expressions, the same `Outcome` works on any backend: pandas, Polars, PyArrow, DuckDB, or a lazy
    frame.

    Build one with the class methods, which mirror the `Surv` constructors (`right`, `left`,
    `counting`, `interval`, `multistate`), or parse an R-style formula response with
    `Outcome.from_formula()`. Call `bind()` to turn it into a `Surv` for a particular frame.

    Examples
    --------
    Describe the `lung` endpoint once, where a `status` of `2` means the patient died:

    ```{python}
    import greenwood as gw

    # Name the columns and the event encoding, with no data attached yet
    death = gw.Outcome.right(time="time", event="status", event_value=2)
    death
    ```

    Pass the description to an estimator along with the frame. Column names work for covariates
    and for `by=` too:

    ```{python}
    lung = gw.load_dataset("lung")

    # Fit a Cox model and a stratified Kaplan-Meier curve from the same description
    cox = gw.CoxPH().fit(death, covariates=["age", "sex"], data=lung)
    km = gw.KaplanMeier().fit(death, by="sex", data=lung)
    cox
    ```

    The same response can be written as a formula, which is parsed into an `Outcome`:

    ```{python}
    gw.Outcome.from_formula(formula="Surv(time, status == 2)") == death
    ```
    """

    kind: str
    columns: tuple[tuple[str, str | Duration], ...]
    event_value: Any = None
    censor_value: Any = None
    states: tuple[str, ...] | None = None
    state_values: tuple[Any, ...] | None = field(default=None, repr=False)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def right(
        cls,
        time: str | Duration,
        event: str | None = None,
        *,
        weights: str | None = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Outcome:
        """Describe a right-censored response.

        Parameters
        ----------
        time
            The column holding exit times, or a `duration()` computed from two date columns.
        event
            The column holding the event indicator. If `None`, every row is an event.
        weights
            The column holding case weights (optional).
        event_value
            The value (or list of values) in `event` that marks an event. Every other row is
            censored. See `Surv.right()`.
        censor_value
            The value (or list of values) in `event` that marks censoring. Every other row is an
            event. Pass this or `event_value`, not both.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.right(time="time", event="status", event_value=2)
        ```
        """
        return cls._build(
            "right",
            {"time": time, "event": event, "weights": weights},
            event_value=event_value,
            censor_value=censor_value,
        )

    @classmethod
    def left(
        cls,
        time: str | Duration,
        event: str | None = None,
        *,
        weights: str | None = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Outcome:
        """Describe a left-censored response.

        Parameters
        ----------
        time
            The column holding observation times, or a `duration()`.
        event
            The column holding the event indicator. If `None`, every row is an event.
        weights
            The column holding case weights (optional).
        event_value
            The value (or list of values) in `event` that marks an event. See `Surv.left()`.
        censor_value
            The value (or list of values) in `event` that marks an event-free row. Pass this or
            `event_value`, not both.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.left(time="time", event="detected", event_value="yes")
        ```
        """
        return cls._build(
            "left",
            {"time": time, "event": event, "weights": weights},
            event_value=event_value,
            censor_value=censor_value,
        )

    @classmethod
    def counting(
        cls,
        start: str | Duration,
        stop: str | Duration,
        event: str | None = None,
        *,
        weights: str | None = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Outcome:
        """Describe a counting-process `(start, stop]` response.

        Parameters
        ----------
        start
            The column holding entry times, or a `duration()`.
        stop
            The column holding exit times, or a `duration()` computed from two date columns.
        event
            The column holding the event indicator. If `None`, every row is an event.
        weights
            The column holding case weights (optional).
        event_value
            The value (or list of values) in `event` that marks an event. See `Surv.counting()`.
        censor_value
            The value (or list of values) in `event` that marks censoring. Pass this or
            `event_value`, not both.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.counting(start="tstart", stop="tstop", event="event")
        ```
        """
        return cls._build(
            "counting",
            {"start": start, "stop": stop, "event": event, "weights": weights},
            event_value=event_value,
            censor_value=censor_value,
        )

    @classmethod
    def interval(
        cls, lower: str | Duration, upper: str | Duration, *, weights: str | None = None
    ) -> Outcome:
        """Describe an interval-censored response.

        Parameters
        ----------
        lower
            The column holding interval lower bounds.
        upper
            The column holding interval upper bounds (`inf` for right-censored rows).
        weights
            The column holding case weights (optional).

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.interval(lower="left", upper="right")
        ```
        """
        return cls._build("interval", {"lower": lower, "upper": upper, "weights": weights})

    @classmethod
    def multistate(
        cls,
        time: str | Duration,
        event: str,
        states: Sequence[str] | Mapping[str, Any],
        *,
        start: str | Duration | None = None,
        weights: str | None = None,
        censor_value: Any = None,
    ) -> Outcome:
        """Describe a multi-state or competing-risks response.

        Parameters
        ----------
        time
            The column holding event or censoring times, or a `duration()`.
        event
            The column holding the outcome.
        states
            A mapping of state label to the `event` value (or values) marking it, such as
            `{"pcm": 1, "death": 2}`, or a sequence of labels when `event` already holds codes
            `0, 1, 2, ...`. See `Surv.multistate()`.
        start
            The column holding entry times, for late entry (optional).
        weights
            The column holding case weights (optional).
        censor_value
            The value (or list of values) in `event` that marks censoring when `states` is a
            mapping. The default is `0`.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.multistate(time="etime", event="cause", states={"pcm": 1, "death": 2})
        ```
        """
        if isinstance(states, Mapping):
            labels = tuple(str(k) for k in states)
            values: tuple[Any, ...] | None = tuple(_freeze(v) for v in states.values())
        else:
            labels = tuple(str(s) for s in states)
            values = None
        if not labels:
            raise ValueError("`states` must name at least one state.")
        outcome = cls._build(
            "multistate",
            {"time": time, "event": event, "start": start, "weights": weights},
            censor_value=censor_value,
        )
        return cls(
            kind=outcome.kind,
            columns=outcome.columns,
            censor_value=outcome.censor_value,
            states=labels,
            state_values=values,
        )

    @classmethod
    def first_event(
        cls,
        endpoints: Mapping[str, Sequence[Any]],
        *,
        censor_at: str | Duration | None = None,
        start: str | Duration | None = None,
        weights: str | None = None,
    ) -> Outcome:
        """Describe a competing-risks response built from several endpoint column pairs.

        Each endpoint has its own time and event column. For each row, the earliest observed event
        wins, ties go to the endpoint listed first, and rows with no observed event are censored
        at the latest endpoint time (or at `censor_at`). See `Surv.first_event()`.

        Parameters
        ----------
        endpoints
            A mapping of state label to `(time, event)` or `(time, event, event_value)` column
            names, in priority order for ties.
        censor_at
            The column holding the censoring time for rows with no observed event (optional).
        start
            The column holding entry times, for late entry (optional).
        weights
            The column holding case weights (optional).

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        # PCM and death are recorded in separate column pairs in mgus2
        cr = gw.Outcome.first_event(
            endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")}
        )
        gw.AalenJohansen().fit(cr, data=gw.load_dataset("mgus2"))
        ```
        """
        if not endpoints:
            raise ValueError("`endpoints` must name at least one endpoint.")
        arguments: dict[str, Any] = {}
        labels: list[str] = []
        values: list[Any] = []
        for i, (label, spec) in enumerate(endpoints.items()):
            if isinstance(spec, str) or len(spec) not in (2, 3):
                raise ValueError(
                    f"Endpoint {label!r} must be `(time, event)` or `(time, event, event_value)`."
                )
            arguments[f"time{i}"] = spec[0]
            arguments[f"event{i}"] = spec[1]
            labels.append(str(label))
            values.append(_freeze(spec[2]) if len(spec) == 3 else None)
        arguments.update({"censor_at": censor_at, "start": start, "weights": weights})
        columns: list[tuple[str, str | Duration]] = []
        for arg, value in arguments.items():
            ref = _column(arg, value, optional=arg in ("censor_at", "start", "weights"))
            if ref is not None:
                columns.append((arg, ref))
        return cls(
            kind="first_event",
            columns=tuple(columns),
            states=tuple(labels),
            state_values=tuple(values),
        )

    @classmethod
    def from_formula(cls, formula: str) -> Outcome:
        """Parse an R-style formula response such as `"Surv(time, status == 2)"`.

        The response is written as a call to `Surv()` with column names and literal values. The
        forms follow R's `survival::Surv()`:

        - `Surv(time)`: every row is an event
        - `Surv(time, event)`: `event` is boolean or `0`/`1`
        - `Surv(time, status == 2)` or `Surv(time, status != 0)`: declare the event value, or the
          censoring value
        - `Surv(time, status in (1, 2))`: several event values (R's `%in% c(1, 2)` is accepted
          too)
        - `Surv(start, stop, event)`: the counting-process form
        - `Surv(time, event, type="left")` and `Surv(lower, upper, type="interval2")`
        - `Surv(time, cause, states={"pcm": 1, "death": 2})`: a multi-state response

        Keyword arguments `event_value=`, `censor_value=`, `weights=`, and `states=` may also be
        given inside the call. Column names that are not valid Python identifiers can be quoted
        with backticks. Dotted R names such as `ph.ecog` work without quoting.

        Parameters
        ----------
        formula
            The response, optionally followed by `~ 1`. To include covariates, pass the full
            formula to an estimator's `fit()` instead.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.from_formula(formula="Surv(time, status == 2)")
        ```
        """
        outcome, rhs = parse_formula(formula)
        if rhs is not None:
            raise ValueError(
                "`Outcome.from_formula()` takes the response only. Pass the full formula "
                f"({formula!r}) to an estimator's `fit()` to include covariates."
            )
        return outcome

    @classmethod
    def _build(
        cls,
        kind: str,
        arguments: dict[str, Any],
        *,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Outcome:
        required = {"time", "start", "stop", "lower", "upper"}
        if kind == "multistate":
            required = {"time", "event"}
        columns: list[tuple[str, str | Duration]] = []
        for arg, value in arguments.items():
            name = _column(arg, value, optional=arg not in required)
            if name is not None:
                columns.append((arg, name))
        if event_value is not None and censor_value is not None:
            raise ValueError("Pass `event_value` or `censor_value`, not both.")
        if (event_value is not None or censor_value is not None) and "event" not in dict(columns):
            raise ValueError("`event_value` and `censor_value` need an `event` column to encode.")
        return cls(
            kind=kind,
            columns=tuple(columns),
            event_value=_freeze(event_value),
            censor_value=_freeze(censor_value),
        )

    # -- views ----------------------------------------------------------------

    @property
    def column_names(self) -> tuple[str, ...]:
        """The distinct column names the response reads, in argument order."""
        names: list[str] = []
        for _, ref in self.columns:
            names += list(ref.columns) if isinstance(ref, Duration) else [ref]
        return tuple(dict.fromkeys(names))

    def __repr__(self) -> str:
        cols = dict(self.columns)
        if self.kind == "first_event":
            parts = [f"endpoints={self._endpoints_argument()!r}"]
            parts += [f"{a}={cols[a]!r}" for a in ("censor_at", "start", "weights") if a in cols]
            return f"Outcome.first_event({', '.join(parts)})"
        order = _ARGUMENTS[self.kind]
        positional = {
            "right": ("time", "event"),
            "left": ("time", "event"),
            "counting": ("start", "stop", "event"),
            "interval": ("lower", "upper"),
            "multistate": ("time", "event"),
        }[self.kind]
        parts = [f"{a}={cols[a]!r}" for a in positional if a in cols]
        if self.kind == "multistate":
            parts.append(f"states={self._states_argument()!r}")
        parts += [f"{a}={cols[a]!r}" for a in order if a in cols and a not in positional]
        if self.event_value is not None:
            parts.append(f"event_value={self.event_value!r}")
        if self.censor_value is not None:
            parts.append(f"censor_value={self.censor_value!r}")
        return f"Outcome.{self.kind}({', '.join(parts)})"

    def _endpoints_argument(self) -> dict[str, tuple[Any, ...]]:
        assert self.states is not None and self.state_values is not None
        cols = dict(self.columns)
        endpoints: dict[str, tuple[Any, ...]] = {}
        for i, (label, value) in enumerate(zip(self.states, self.state_values, strict=True)):
            spec = (cols[f"time{i}"], cols[f"event{i}"])
            endpoints[label] = spec if value is None else (*spec, value)
        return endpoints

    def _states_argument(self) -> tuple[str, ...] | dict[str, Any]:
        assert self.states is not None
        if self.state_values is None:
            return self.states
        return dict(zip(self.states, self.state_values, strict=True))

    # -- binding --------------------------------------------------------------

    def bind(self, data: Any) -> Surv:
        """Build a `Surv` response from the columns of `data`.

        `bind()` is strict: missing values in the response columns raise an error, as they do in
        the `Surv` constructors. To drop incomplete rows automatically, pass the `Outcome` to an
        estimator's `fit()` with `data=` instead, where the response and covariates are filtered
        together.

        Parameters
        ----------
        data
            A data frame (pandas, Polars, PyArrow, DuckDB, a lazy frame, ...) or a mapping of
            column names to values.

        Returns
        -------
        Surv
            The response for this frame.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung")
        gw.Outcome.right(time="time", event="status", event_value=2).bind(lung)
        ```
        """
        cols = dict(self.columns)
        if self.kind == "first_event":
            return Surv.first_event(
                self._endpoints_argument(),
                data=data,
                censor_at=cols.get("censor_at"),
                start=cols.get("start"),
                weights=cols.get("weights"),
            )
        if self.kind == "interval":
            return Surv.interval(
                cols["lower"], cols["upper"], weights=cols.get("weights"), data=data
            )
        if self.kind == "multistate":
            return Surv.multistate(
                cols["time"],
                cols["event"],
                self._states_argument(),
                start=cols.get("start"),
                weights=cols.get("weights"),
                data=data,
                censor_value=self.censor_value,
            )
        encoding: dict[str, Any] = {
            "weights": cols.get("weights"),
            "data": data,
            "event_value": self.event_value,
            "censor_value": self.censor_value,
        }
        if self.kind == "counting":
            return Surv.counting(cols["start"], cols["stop"], cols.get("event"), **encoding)
        if self.kind == "left":
            return Surv.left(cols["time"], cols.get("event"), **encoding)
        return Surv.right(cols["time"], cols.get("event"), **encoding)


# -- formula parsing ----------------------------------------------------------

_BACKTICK = re.compile(r"`([^`]+)`")


def split_formula(formula: str) -> tuple[str, str | None]:
    """Split `"lhs ~ rhs"` into its sides. A missing, empty, or `1` right-hand side is `None`."""
    if formula.count("~") > 1:
        raise ValueError(f"A formula may contain only one `~`. Got {formula!r}.")
    lhs, _, rhs = formula.partition("~")
    rhs_s = rhs.strip()
    return lhs.strip(), (None if rhs_s in ("", "1") else rhs_s)


def parse_formula(formula: str) -> tuple[Outcome, str | None]:
    """Parse `"Surv(...) ~ rhs"` into an `Outcome` and the right-hand side (or `None`)."""
    lhs, rhs = split_formula(formula)
    if not lhs:
        raise ValueError(
            f"The formula {formula!r} has no response. Write it as `Surv(time, status == 2) ~ ...`."
        )
    return _parse_response(lhs), rhs


def _parse_response(text: str) -> Outcome:
    quoted: dict[str, str] = {}

    def _stash(match: re.Match[str]) -> str:
        key = f"__gw_quoted_{len(quoted)}"
        quoted[key] = match.group(1)
        return key

    source = _BACKTICK.sub(_stash, text).replace("%in%", " in ")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Could not parse the formula response {text!r}.") from error
    call = tree.body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        raise ValueError(f"The formula response must be a call to `Surv(...)`. Got {text!r}.")
    if call.func.id != "Surv":
        raise ValueError(
            f"The formula response must be a call to `Surv(...)`, not `{call.func.id}(...)`."
        )

    def name_of(node: ast.expr, what: str) -> str:
        if isinstance(node, ast.Name):
            return quoted.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{name_of(node.value, what)}.{node.attr}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise ValueError(
            f"`Surv()` expects a column name for {what}, got `{ast.unparse(node)}`. Only column "
            "names and literal values are allowed in a formula response."
        )

    def literal(node: ast.expr, what: str) -> Any:
        # R's c(1, 2) becomes a tuple.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
            return tuple(literal(a, what) for a in node.args)
        try:
            return _freeze(ast.literal_eval(node))
        except ValueError as error:
            raise ValueError(
                f"`{ast.unparse(node)}` in {what} must be a literal value (a number, string, or "
                "list of them)."
            ) from error

    args = list(call.args)
    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    unknown = set(keywords) - {"type", "event_value", "censor_value", "weights", "states"}
    if unknown:
        raise ValueError(f"Unknown `Surv()` argument(s): {sorted(unknown)}.")

    ctype = literal(keywords["type"], "`type=`") if "type" in keywords else None
    event_value = (
        literal(keywords["event_value"], "`event_value=`") if "event_value" in keywords else None
    )
    censor_value = (
        literal(keywords["censor_value"], "`censor_value=`") if "censor_value" in keywords else None
    )
    weights = name_of(keywords["weights"], "`weights=`") if "weights" in keywords else None
    states = literal(keywords["states"], "`states=`") if "states" in keywords else None

    def event_of(node: ast.expr) -> tuple[str, Any, Any]:
        """Read an event argument: a bare column, or `col == v` / `!=` / `in` / `not in`."""
        if not isinstance(node, ast.Compare):
            return name_of(node, "the event"), None, None
        if len(node.ops) != 1:
            raise ValueError(f"Chained comparisons are not supported: `{ast.unparse(node)}`.")
        column = name_of(node.left, "the event")
        value = literal(node.comparators[0], "the event encoding")
        op = node.ops[0]
        if isinstance(op, (ast.Eq, ast.In)):
            return column, value, None
        if isinstance(op, (ast.NotEq, ast.NotIn)):
            return column, None, value
        raise ValueError(
            f"`{ast.unparse(node)}` is not supported. Encode the event by value: "
            f"`{column} == v`, `{column} != v`, or `{column} in (v1, v2)`."
        )

    def merged(inline: Any, keyword: Any, what: str) -> Any:
        if inline is not None and keyword is not None:
            raise ValueError(f"The event encoding is given twice (inline and as `{what}=`).")
        return inline if inline is not None else keyword

    n = len(args)
    if states is not None:
        if ctype not in (None, "mstate"):
            raise ValueError("`states=` cannot be combined with `type=`.")
        if n not in (2, 3):
            raise ValueError("A multi-state `Surv()` takes (time, event) or (start, stop, event).")
        event_col, inline_event, inline_censor = event_of(args[-1])
        if inline_event is not None or event_value is not None:
            raise ValueError("Multi-state responses map values with `states=`, not `event_value`.")
        return Outcome.multistate(
            name_of(args[-2], "the time"),
            event_col,
            states,
            start=name_of(args[0], "the start time") if n == 3 else None,
            weights=weights,
            censor_value=merged(inline_censor, censor_value, "censor_value"),
        )

    if ctype in ("interval", "interval2"):
        if n != 2:
            raise ValueError('An interval `Surv()` takes (lower, upper, type="interval2").')
        if event_value is not None or censor_value is not None:
            raise ValueError("An interval-censored response has no event encoding.")
        return Outcome.interval(
            name_of(args[0], "the lower bound"),
            name_of(args[1], "the upper bound"),
            weights=weights,
        )

    if ctype not in (None, "right", "left", "counting"):
        raise ValueError(
            f"Unsupported `type={ctype!r}`. Use 'right', 'left', 'counting', or 'interval2'."
        )
    if ctype == "counting" and n != 3:
        raise ValueError("A counting-process `Surv()` takes (start, stop, event).")
    if ctype in ("right", "left") and n > 2:
        raise ValueError(f'A `type="{ctype}"` response takes (time) or (time, event).')
    if n not in (1, 2, 3):
        raise ValueError("`Surv()` takes 1 to 3 positional arguments.")

    event_col: str | None = None
    inline_event = inline_censor = None
    if n >= 2:
        event_col, inline_event, inline_censor = event_of(args[-1])
    ev = merged(inline_event, event_value, "event_value")
    cv = merged(inline_censor, censor_value, "censor_value")

    if n == 3:
        return Outcome.counting(
            name_of(args[0], "the start time"),
            name_of(args[1], "the stop time"),
            event_col,
            weights=weights,
            event_value=ev,
            censor_value=cv,
        )
    build = Outcome.left if ctype == "left" else Outcome.right
    return build(
        name_of(args[0], "the time"), event_col, weights=weights, event_value=ev, censor_value=cv
    )


def split_terms(rhs: str) -> list[str]:
    """Split a right-hand side on top-level `+`, leaving terms like `C(x, Treatment(1))` intact."""
    terms: list[str] = []
    depth = 0
    current: list[str] = []
    quoted = False
    for ch in rhs:
        if ch == "`":
            quoted = not quoted
        elif not quoted and ch in "([{":
            depth += 1
        elif not quoted and ch in ")]}":
            depth -= 1
        if ch == "+" and depth == 0 and not quoted:
            terms.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    terms.append("".join(current).strip())
    if any(not t for t in terms):
        raise ValueError(f"The right-hand side `{rhs}` has an empty term.")
    return terms


# R's special terms: `strata(inst)` and `cluster(id)` route columns to the matching argument.
_SPECIAL = re.compile(r"(strata|cluster)\((.*)\)", flags=re.S)


def _special_term(term: str) -> tuple[str, list[str]] | None:
    """Recognize `strata(a, b)` / `cluster(id)` and return the argument name and its columns."""
    match = _SPECIAL.fullmatch(term)
    if match is None:
        return None
    names = [_plain_name(n) for n in match.group(2).split(",")]
    return match.group(1), names


def _plain_name(term: str) -> str:
    term = term.strip()
    if term.startswith("`") and term.endswith("`") and len(term) > 2:
        return term[1:-1]
    if not re.fullmatch(r"[A-Za-z_][\w.]*", term):
        raise ValueError(
            f"`{term}` is not a column name. The right-hand side of a formula for this "
            "estimator names grouping columns only (e.g., `~ sex + ph.ecog`)."
        )
    return term


def rhs_terms(rhs: str) -> list[str]:
    """Split a right-hand side such as `"sex + ph.ecog"` into plain column names."""
    return [_plain_name(t) for t in split_terms(rhs)]


# -- binding at fit() ---------------------------------------------------------


@dataclass
class BoundInputs:
    """The resolved inputs of a `fit()` call: a `Surv`, covariate designs, and label arrays."""

    surv: Surv
    designs: dict[str, Any]
    labels: dict[str, Any]
    data: Any
    n_dropped: int = 0
    n_input: int = 0


@dataclass(frozen=True)
class _GroupTerms:
    """Right-hand-side columns to combine into one grouping label (e.g., `by=`)."""

    names: tuple[str, ...]


def bind_fit_inputs(
    surv: Any,
    *,
    data: Any = None,
    designs: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
    rhs_to: str | None = None,
    required: tuple[str, ...] = (),
    estimator: str = "This estimator",
) -> BoundInputs:
    """Resolve a `fit()` call's response, covariates, and labels against `data`.

    `surv` may be a `Surv`, an `Outcome`, or a formula string. `designs` holds covariate
    arguments (a formula string, a list of column names, a frame, an array, or `None`). `labels`
    holds per-row arguments such as `by`, `strata`, `cluster`, and `weights` (a column name, an
    array, or `None`). `rhs_to` names the design or label that receives a formula's right-hand
    side. `required` lists designs or labels that must end up non-`None`.

    With a plain `Surv`, nothing is filtered: column names are looked up in `data` and everything
    else passes through, so existing calls behave exactly as before. With an `Outcome` or a
    formula, every referenced column is read from `data`, rows with a missing value in any of them
    are dropped once, and the response is bound to the remaining rows.
    """
    designs = dict(designs or {})
    labels = dict(labels or {})

    outcome: Outcome | None
    if isinstance(surv, Surv):
        outcome = None
    elif isinstance(surv, Outcome):
        outcome = surv
    elif isinstance(surv, str):
        outcome, rhs = parse_formula(surv)
        if rhs is not None:
            _place_rhs(rhs, rhs_to, designs, labels, estimator)
    else:
        raise TypeError(
            "The response must be a `Surv`, an `Outcome`, or a formula string such as "
            f"'Surv(time, status == 2) ~ age'. Got {type(surv).__name__}."
        )

    for name in required:
        if designs.get(name) is None and labels.get(name) is None:
            example = "age + sex" if name in designs else "sex"
            raise ValueError(
                f"{estimator} needs `{name}`. Pass it as an argument, or give a full formula such "
                f"as 'Surv(time, status == 2) ~ {example}'."
            )

    if outcome is None:
        assert isinstance(surv, Surv)
        return _bind_plain(surv, data, designs, labels)
    if data is None:
        raise ValueError(
            "An `Outcome` or formula names columns, so `fit()` needs the frame that holds them. "
            "Pass `data=`."
        )
    return _bind_outcome(outcome, data, designs, labels)


def _place_rhs(
    rhs: str, rhs_to: str | None, designs: dict[str, Any], labels: dict[str, Any], estimator: str
) -> None:
    if rhs_to is None:
        raise ValueError(
            f"{estimator} takes no covariates, so the formula's right-hand side must be `1`. "
            f"Got `~ {rhs}`."
        )
    grouping = rhs_to not in designs
    terms: list[str] = []
    for term in split_terms(rhs):
        special = _special_term(term)
        if special is None:
            terms.append(term)
            continue
        target, names = special
        if target not in labels:
            if grouping and target == "strata":
                # survfit() treats `strata(x)` as an ordinary grouping term.
                terms += names
                continue
            raise ValueError(f"{estimator} does not support `{target}()` terms in a formula.")
        if labels[target] is not None:
            raise ValueError(
                f"`{target}` was given both in the formula and as `{target}=`. Use one."
            )
        labels[target] = _GroupTerms(tuple(names))

    if not grouping:
        if designs[rhs_to] is not None:
            raise ValueError(
                f"Covariates were given both in the formula and as `{rhs_to}=`. Use one."
            )
        designs[rhs_to] = " + ".join(terms) if terms else None
    elif terms:
        if labels.get(rhs_to) is not None:
            raise ValueError(f"Groups were given both in the formula and as `{rhs_to}=`. Use one.")
        labels[rhs_to] = _GroupTerms(tuple(_plain_name(t) for t in terms))


def _to_frame(data: Any) -> Any:
    if isinstance(data, Mapping):
        raise TypeError(
            "`fit()` needs a DataFrame for `data=` (pandas, Polars, PyArrow, DuckDB, ...). A plain "
            "mapping works with `Surv` constructors and `Outcome.bind()`, but not here."
        )
    try:
        frame: Any = nw.from_native(data)
    except TypeError as error:
        raise TypeError(
            f"`data` must be a DataFrame (pandas, Polars, PyArrow, DuckDB, ...). Got "
            f"{type(data).__name__}."
        ) from error
    if not isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        raise TypeError(f"`data` must be a DataFrame, not a {type(data).__name__}.")
    return frame


def _check_columns(names: Sequence[str], available: Sequence[str]) -> None:
    from ._ingest import _check_names  # pyright: ignore[reportPrivateUsage]

    _check_names(list(names), list(available))


def _formula_columns(formula: str, available: Sequence[str]) -> list[str] | None:
    """The columns a covariate formula reads, or `None` if they cannot all be identified.

    formulaic reports the variables a formula needs, but reads a dotted R name such as `ph.ecog`
    as attribute access on `ph`. A reported variable counts as found when it is a column, or when
    the formula mentions a dotted column that starts with it.
    """
    try:
        from formulaic import Formula  # pyright: ignore[reportMissingImports]

        variables = {str(v) for v in Formula(formula).required_variables}  # pyright: ignore
    except Exception:  # noqa: BLE001  # any parse problem is reported later by the model
        return None
    columns = set(available)
    mentioned = [
        c
        for c in available
        if "." in c and re.search(rf"(?<![\w.`]){re.escape(c)}(?![\w.])", formula)
    ]
    mentioned += [c for c in available if f"`{c}`" in formula]
    found: list[str] = []
    for var in sorted(variables):
        if var in columns:
            found.append(var)
            continue
        dotted = [c for c in mentioned if c.startswith(f"{var}.")]
        if not dotted:
            return None
        found += dotted
    return list(dict.fromkeys([*found, *mentioned]))


def _bind_plain(
    surv: Surv, data: Any, designs: dict[str, Any], labels: dict[str, Any]
) -> BoundInputs:
    """A `Surv` was given: only resolve column names, never filter rows."""
    needs_data = any(isinstance(v, str) for v in labels.values()) or any(
        _is_name_list(v) for v in designs.values()
    )
    if not needs_data:
        return BoundInputs(surv, designs, labels, data, n_input=surv.n)
    if data is None:
        arg = next(
            k for k, v in {**labels, **designs}.items() if isinstance(v, str) or _is_name_list(v)
        )
        raise ValueError(f"`{arg}=` names columns, but no `data=` was given.")
    frame = _to_frame(data)
    wanted = [v for v in labels.values() if isinstance(v, str)]
    for v in designs.values():
        if _is_name_list(v):
            wanted += list(v)
    _check_columns(wanted, frame.collect_schema().names())
    eager = select_columns(frame, list(dict.fromkeys(wanted)))
    out_labels = {
        k: as_1d(eager.get_column(v).to_numpy()) if isinstance(v, str) else v
        for k, v in labels.items()
    }
    out_designs = {
        k: eager.select(list(v)).to_native() if _is_name_list(v) else v for k, v in designs.items()
    }
    return BoundInputs(surv, out_designs, out_labels, data, n_input=surv.n)


def _is_name_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0  # pyright: ignore[reportUnknownArgumentType]
        and all(isinstance(v, str) for v in value)  # pyright: ignore[reportUnknownVariableType]
    )


def _bind_outcome(
    outcome: Outcome, data: Any, designs: dict[str, Any], labels: dict[str, Any]
) -> BoundInputs:
    """An `Outcome` was given: read every column from `data` and drop incomplete rows once."""
    frame = _to_frame(data)
    available = [str(c) for c in frame.collect_schema().names()]

    # Columns that define the analysis rows. A row missing any of them is dropped.
    used: list[str] = list(outcome.column_names)
    for value in labels.values():
        if isinstance(value, str):
            used.append(value)
        elif isinstance(value, _GroupTerms):
            used += list(value.names)
    keep_all_columns = False
    for value in designs.values():
        if _is_name_list(value):
            used += list(value)
        elif isinstance(value, str):
            columns = _formula_columns(value, available)
            if columns is not None:
                used += columns
            else:
                # The formula's columns could not all be identified ahead of time. Keep the whole
                # frame so the formula can still see every column. The model's own complete-case
                # step then covers the covariates.
                keep_all_columns = True
    used = list(dict.fromkeys(used))
    _check_columns(used, available)

    selected = select_columns(frame, None if keep_all_columns else used)
    n_rows = int(selected.shape[0])

    keep = np.ones(n_rows, dtype=bool)
    for name in used:
        keep &= ~missing_mask(as_1d(selected.get_column(name).to_numpy()))
    for value in (*labels.values(), *designs.values()):
        if value is None or isinstance(value, (str, _GroupTerms)) or _is_name_list(value):
            continue
        rows = _row_count(value)
        if rows != n_rows:
            raise ValueError(
                f"An argument has {rows} rows, but `data` has {n_rows}. Values passed alongside an "
                "`Outcome` must line up with the rows of `data`."
            )
    n_dropped = int((~keep).sum())
    filtered = selected.filter(keep.tolist()) if n_dropped else selected
    native = filtered.to_native()

    surv = outcome.bind(native)

    out_labels: dict[str, Any] = {}
    for key, value in labels.items():
        if value is None:
            out_labels[key] = None
        elif isinstance(value, str):
            out_labels[key] = as_1d(filtered.get_column(value).to_numpy())
        elif isinstance(value, _GroupTerms):
            out_labels[key] = _group_labels(filtered, value.names)
        else:
            out_labels[key] = _filter_rows(value, keep)

    out_designs: dict[str, Any] = {}
    for key, value in designs.items():
        if value is None or isinstance(value, str):
            out_designs[key] = value
        elif _is_name_list(value):
            out_designs[key] = filtered.select(list(value)).to_native()
        else:
            out_designs[key] = _filter_rows(value, keep)

    return BoundInputs(surv, out_designs, out_labels, native, n_dropped, n_input=n_rows)


def _filter_rows(value: Any, keep: Array) -> Any:
    """Keep the rows of an array, frame, or series that pass the mask (2-D arrays included)."""
    if isinstance(value, np.ndarray):
        return value[keep]  # pyright: ignore[reportUnknownVariableType]
    try:
        frame: Any = nw.from_native(value, eager_only=True)
    except TypeError:
        arr = np.asarray(value)
        return arr[keep] if arr.ndim > 1 else as_1d(value)[keep]
    return frame.filter(keep.tolist()).to_native()


def _row_count(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.shape[0])  # pyright: ignore[reportUnknownArgumentType]
    try:
        return int(nw.from_native(value, eager_only=True).shape[0])
    except TypeError:
        return int(as_1d(value).shape[0])


def _group_labels(frame: Any, names: tuple[str, ...]) -> Array:
    """One grouping label per row. A single column is used as-is, several are combined R-style."""
    columns = [as_1d(frame.get_column(n).to_numpy()) for n in names]
    if len(columns) == 1:
        return columns[0]
    return np.array(
        [
            ", ".join(f"{n}={v}" for n, v in zip(names, row, strict=True))
            for row in zip(*columns, strict=True)
        ],
        dtype=object,
    )
