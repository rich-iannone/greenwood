"""`Outcome`: a reusable, data-free description of a survival response.

An `Outcome` is the structured form of a formula's left-hand side. It holds the arguments of a
`Surv(...)` or `event_time(...)` call, written exactly as they would be written inside a formula
(column names and simple expressions such as `status == 2`), and no data. It is bound to a frame
later, either explicitly with `Outcome.bind()` or by an estimator's `fit(outcome, ..., data=df)`.
Binding at `fit()` reads every column the model needs from the one frame and drops incomplete rows
once, so the response, covariates, and per-row labels (`by`, `strata`, `cluster`, `weights`)
always stay aligned.

The expressions are evaluated on whatever backend holds the data (pandas, Polars, PyArrow, DuckDB,
lazy frames) and the results go through `Surv()` or `event_time()`, so R's rules apply unchanged.
Parsing uses Python's `ast` module and accepts only column names, literal values, comparisons, and
the functions `duration()`, `factor()`, and `c()`. Nothing in a formula is evaluated as code.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt

from ._event_time import EventTime, as_surv, event_time
from ._ingest import Duration, as_1d, missing_mask, select_columns
from ._ingest import _check_names as check_names  # pyright: ignore[reportPrivateUsage]
from ._surv import Surv, first_event

__all__ = ["Outcome"]

Array = npt.NDArray[Any]

_BACKTICK = re.compile(r"`([^`]+)`")
# R's integer ranges such as `0:2`, which Python can't parse, become `c(0, 1, 2)`.
_RANGE = re.compile(r"(?<![\w.])(-?\d+)\s*:\s*(-?\d+)(?![\w.])")
# The per-endpoint arguments of a first_event outcome: time0, event0, time1, event1, ...
_ENDPOINT_ARGUMENT = re.compile(r"(time|event)\d+")
_SURV_TYPES = ("right", "left", "interval", "counting", "interval2", "mstate")


# -- expressions ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Column:
    """A column, by name."""

    name: str

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name,)

    def source(self) -> str:
        return self.name if _is_identifier(self.name) else f"`{self.name}`"


@dataclass(frozen=True)
class _Compare:
    """`column == v`, `!=`, `in (v1, v2)`, or `not in (...)`, with R's missing-value rules."""

    name: str
    op: str
    values: tuple[Any, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name,)

    def source(self) -> str:
        column = _Column(self.name).source()
        if self.op in ("==", "!="):
            return f"{column} {self.op} {self.values[0]!r}"
        return f"{column} {self.op} {self.values!r}"


@dataclass(frozen=True)
class _Factor:
    """R's `factor(x, levels, labels)`: a categorical whose first level means censored."""

    name: str
    levels: tuple[Any, ...] | None = None
    labels: tuple[str, ...] | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name,)

    def source(self) -> str:
        parts = [_Column(self.name).source()]
        if self.levels is not None:
            parts.append(f"levels={list(self.levels)!r}")
        if self.labels is not None:
            parts.append(f"labels={list(self.labels)!r}")
        return f"factor({', '.join(parts)})"


_Expr = _Column | _Compare | _Factor | Duration


def _expr_columns(expr: _Expr) -> tuple[str, ...]:
    return expr.columns


def _expr_source(expr: _Expr) -> str:
    if isinstance(expr, Duration):
        return f"duration({expr.start!r}, {expr.end!r}, unit={expr.unit!r})"
    return expr.source()


def _is_identifier(name: str) -> bool:
    return all(part.isidentifier() for part in name.split("."))


class _Categorical:
    """A minimal categorical (levels plus integer codes, -1 for missing) that `Surv()` reads as a
    factor, so `factor()` works without importing pandas."""

    def __init__(self, categories: list[str], codes: list[int]) -> None:
        self.categories = categories
        self.codes = codes


def _r_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _r_equal(a: Any, b: Any) -> bool:
    """Equality as R's `==` sees it: a number compared with a string is compared as strings."""
    if isinstance(a, str) != isinstance(b, str):
        return _r_label(a) == _r_label(b)
    return bool(a == b)


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _evaluate(expr: _Expr, get: Any) -> Any:
    """Evaluate an expression against `get(name)`, which returns a Narwhals series or an array."""
    if isinstance(expr, Duration):
        return expr.compute(_as_array(get(expr.start)), _as_array(get(expr.end)))
    column = get(expr.name)
    if isinstance(expr, _Column):
        return column.to_native() if isinstance(column, nw.Series) else column
    values = _as_list(column)
    if isinstance(expr, _Compare):
        out: list[bool | None] = []
        for v in values:
            if expr.op in ("==", "!="):
                # R: a comparison with a missing value is missing.
                if _is_missing_value(v):
                    out.append(None)
                else:
                    out.append(_r_equal(v, expr.values[0]) == (expr.op == "=="))
            else:
                # R: `NA %in% x` is FALSE, so `not in` is TRUE.
                hit = (not _is_missing_value(v)) and any(_r_equal(v, t) for t in expr.values)
                out.append(hit if expr.op == "in" else not hit)
        return out
    present = [v for v in values if not _is_missing_value(v)]
    levels = list(expr.levels) if expr.levels is not None else sorted(set(present))
    for i, level in enumerate(levels):
        if any(_r_equal(level, earlier) for earlier in levels[:i]):
            raise ValueError(f"factor level [{i + 1}] is duplicated")
    labels = list(expr.labels) if expr.labels is not None else [_r_label(lv) for lv in levels]
    if len(labels) != len(levels):
        raise ValueError(
            f"`factor({expr.name}, ...)` has {len(levels)} levels but {len(labels)} labels."
        )
    # As in R, repeated labels merge their levels (for example, two codes that both mean censored).
    categories = list(dict.fromkeys(labels))
    code_of = [categories.index(label) for label in labels]

    def code(v: Any) -> int:
        if _is_missing_value(v):
            return -1
        for level, c in zip(levels, code_of, strict=True):
            if _r_equal(v, level):
                return c
        return -1

    codes = [code(v) for v in values]
    return _Categorical(categories, codes)


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, nw.Series):
        nulls = values.is_null().to_list()
        return [None if null else v for v, null in zip(values.to_list(), nulls, strict=True)]
    return list(as_1d(values).tolist())


def _as_array(values: Any) -> Array:
    if isinstance(values, nw.Series):
        return as_1d(values.to_numpy())
    return as_1d(values)


# -- parsing ----------------------------------------------------------------------------------


class _Parser:
    """Parse the argument expressions of a response, for formulas and `Outcome` constructors."""

    def __init__(self, text: str) -> None:
        self.quoted: dict[str, str] = {}

        def stash(match: re.Match[str]) -> str:
            key = f"__gw_quoted_{len(self.quoted)}"
            self.quoted[key] = match.group(1)
            return key

        def expand(match: re.Match[str]) -> str:
            a, b = int(match.group(1)), int(match.group(2))
            step = 1 if b >= a else -1
            return "c(" + ", ".join(str(i) for i in range(a, b + step, step)) + ")"

        source = _BACKTICK.sub(stash, text).replace("%in%", " in ")
        self.source = _RANGE.sub(expand, source)

    def parse(self) -> ast.expr:
        try:
            return ast.parse(self.source, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"Could not parse {self.restore(self.source)!r}.") from error

    def restore(self, text: str) -> str:
        for key, name in self.quoted.items():
            text = text.replace(key, f"`{name}`")
        return text

    def name(self, node: ast.expr, what: str) -> str:
        if isinstance(node, ast.Name):
            return self.quoted.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.name(node.value, what)}.{node.attr}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise ValueError(
            f"Expected a column name for {what}, got `{self.restore(ast.unparse(node))}`."
        )

    def literal(self, node: ast.expr, what: str) -> Any:
        # R's c(1, 2) becomes a tuple.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
            return tuple(self.literal(a, what) for a in node.args)
        try:
            value = ast.literal_eval(node)
        except ValueError as error:
            raise ValueError(
                f"`{self.restore(ast.unparse(node))}` in {what} must be a literal value (a number, "
                "a string, or a list of them)."
            ) from error
        if isinstance(value, (list, set, frozenset)):
            return tuple(value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        return value

    def values(self, node: ast.expr, what: str) -> tuple[Any, ...]:
        value = self.literal(node, what)
        return value if isinstance(value, tuple) else (value,)  # pyright: ignore[reportUnknownVariableType]

    def expr(self, node: ast.expr, what: str) -> _Expr:
        """A column, a comparison, `factor(...)`, or `duration(...)`."""
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError(
                    f"Chained comparisons are not supported: `{self.restore(ast.unparse(node))}`."
                )
            column = self.name(node.left, what)
            op = node.ops[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                value = self.literal(node.comparators[0], what)
                return _Compare(column, "==" if isinstance(op, ast.Eq) else "!=", (value,))
            if isinstance(op, (ast.In, ast.NotIn)):
                values = self.values(node.comparators[0], what)
                return _Compare(column, "in" if isinstance(op, ast.In) else "not in", values)
            raise ValueError(
                f"`{self.restore(ast.unparse(node))}` is not supported in {what}. Use `==`, `!=`, "
                "`in`, or `not in`."
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "duration":
                return self.duration(node)
            if node.func.id == "factor":
                return self.factor(node, what)
        return _Column(self.name(node, what))

    def duration(self, node: ast.Call) -> Duration:
        usage = (
            "`duration()` takes (start, end) columns and an optional unit, as in "
            '`duration(enroll, exit, unit="years")`.'
        )
        params = ("start", "end", "unit")
        given: dict[str, ast.expr] = dict(zip(params, node.args, strict=False))
        for kw in node.keywords:
            if kw.arg not in params or kw.arg in given:
                raise ValueError(usage)
            given[kw.arg] = kw.value
        if len(node.args) > 3 or "start" not in given or "end" not in given:
            raise ValueError(usage)
        unit = self.literal(given["unit"], "the duration unit") if "unit" in given else "days"
        return Duration(
            self.name(given["start"], "the duration start"),
            self.name(given["end"], "the duration end"),
            unit,
        )

    def factor(self, node: ast.Call, what: str) -> _Factor:
        params = ("x", "levels", "labels")
        given: dict[str, ast.expr] = dict(zip(params, node.args, strict=False))
        for kw in node.keywords:
            if kw.arg not in params or kw.arg in given:
                raise ValueError("`factor()` takes (x, levels, labels), as R's `factor()` does.")
            given[kw.arg] = kw.value
        if "x" not in given or len(node.args) > 3:
            raise ValueError("`factor()` takes (x, levels, labels), as R's `factor()` does.")
        levels = self.values(given["levels"], "the factor levels") if "levels" in given else None
        labels = self.values(given["labels"], "the factor labels") if "labels" in given else None
        return _Factor(
            self.name(given["x"], what),
            levels,
            None if labels is None else tuple(str(label) for label in labels),
        )


def _parse_argument(value: Any, arg: str) -> _Expr:
    """Parse one `Outcome` constructor argument: an expression string or a `Duration`.

    A string that isn't a supported expression (such as a column name with spaces) is taken as a
    column name, so names never need quoting here.
    """
    if isinstance(value, Duration):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"`Outcome` describes columns by name, so `{arg}` must be a column name or an "
            f"expression such as 'status == 2'. Got {type(value).__name__}. To build a response "
            "from values, use `Surv()` or `event_time()` directly."
        )
    parser = _Parser(value)
    try:
        return parser.expr(parser.parse(), f"`{arg}`")
    except ValueError:
        if "`" in value:
            raise
        return _Column(value)


# -- the Outcome ------------------------------------------------------------------------------


class Outcome:
    """A reusable description of a survival response: column names and expressions, no data.

    An `Outcome` is the structured form of a formula's left-hand side. `Outcome.surv()` takes the
    arguments of R's `Surv()` and `Outcome.event_time()` those of `event_time()`, each written as
    it would be inside a formula: a column name, or an expression such as `"status == 2"`. Pass
    the `Outcome` to any estimator's fit method together with `data=`. The estimator reads every
    column it needs from that one frame, drops rows with missing values once (as R's `na.omit`
    does), evaluates the expressions, and builds the response with `Surv()` or `event_time()`.

    Because an `Outcome` is made of names and expressions rather than DataFrame code, the same
    description works on any backend: pandas, Polars, PyArrow, DuckDB, or a lazy frame. That makes
    it, together with the equivalent formula strings, the recommended way to start from a data
    frame. Use `Surv()` or `event_time()` directly when the values are already in hand.

    The expressions understood here and in formulas are:

    - a column name (backticks quote unusual names inside an expression),
    - a comparison: `status == 2`, `status != 0`, `status in (1, 2)` (R's `%in% c(1, 2)` works),
      or `status not in (1, 2)`,
    - `factor(x, levels, labels)`, R's factor, for a multi-state status whose first level means
      censored,
    - `duration(start, end, unit="days")` for a time computed from two date columns.

    Examples
    --------
    We'll use the bundled `lung` dataset, from a North Central Cancer Treatment Group trial in
    advanced lung cancer. `time` is days of follow-up and status is `1` (censored) or `2` (died).
    Here is what the dataset looks like:

    ```{python}
    #| echo: false
    import great_docs as gd
    import greenwood as gw

    gd.tbl_preview(
        gw.load_dataset("lung"),
        n_head=5,
        n_tail=3,
        caption="lung: NCCTG advanced lung cancer, 228 patients",
    )
    ```

    Describe the endpoint once, with the same arguments `Surv()` takes:

    ```{python}
    import greenwood as gw

    death = gw.Outcome.surv(time="time", event="status == 2")
    death
    ```

    Pass it to estimators along with the frame. Column names work for covariates and `by=` too:

    ```{python}
    lung = gw.load_dataset("lung")

    cox = gw.CoxPH().fit(death, covariates=["age", "sex"], data=lung)
    km = gw.KaplanMeier().fit(death, by="sex", data=lung)
    cox
    ```

    The same response written as a formula parses to an equal `Outcome`:

    ```{python}
    gw.Outcome.from_formula(formula="Surv(time, status == 2)") == death
    ```
    """

    __slots__ = ("_kind", "_arguments", "_type", "_origin", "_labels")

    _kind: str
    _arguments: tuple[tuple[str, _Expr], ...]
    _type: str | None
    _origin: float
    _labels: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError(
            "Build an Outcome with Outcome.surv(), Outcome.event_time(), Outcome.first_event(), "
            "or Outcome.from_formula()."
        )

    @classmethod
    def _new(
        cls,
        *,
        kind: str,
        arguments: tuple[tuple[str, _Expr], ...],
        type: str | None = None,
        origin: float = 0.0,
        labels: tuple[str, ...] = (),
    ) -> Outcome:
        obj = object.__new__(cls)
        obj._kind = kind
        obj._arguments = arguments
        obj._type = type
        obj._origin = float(origin)
        obj._labels = labels
        return obj

    def _key(self) -> tuple[Any, ...]:
        return (self._kind, self._arguments, self._type, self._origin, self._labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Outcome):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    # -- constructors -------------------------------------------------------------------------

    @classmethod
    def surv(
        cls,
        time: str | Duration,
        time2: str | Duration | None = None,
        event: str | None = None,
        *,
        type: str | None = None,
        origin: float = 0.0,
    ) -> Outcome:
        """Describe a `Surv()` response by column names and expressions.

        The arguments are those of `Surv()` (and of R's `survival::Surv()`), written as they
        would be inside a formula. When the `Outcome` is bound to data, each one is evaluated and
        the results are passed to `Surv()`, so R's type inference and status rules apply: `1`/`2`
        status passes through, `type=` selects left or interval data, and a factor gives a
        multi-state response.

        Parameters
        ----------
        time
            The time column (or start time for counting data, or lower bound for interval data),
            or `duration(...)` for a time computed from two date columns.
        time2
            The stop time for counting data, or the upper bound for interval data. As in R, a
            second argument without `event=` is the status.
        event
            The status: a column, a comparison such as `"status == 2"`, or
            `"factor(cause, c(0, 1, 2), c('censor', 'pcm', 'death'))"` for a multi-state response.
        type
            As in `Surv()`: `"right"`, `"left"`, `"interval"`, `"counting"`, `"interval2"`, or
            `"mstate"`. The default infers it from the arguments.
        origin
            Subtracted from every time, as in `Surv()`.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        # pbc's status is 0 (censored), 1 (transplant), or 2 (died)
        gw.Outcome.surv(time="time", event="status == 2")
        ```

        A counting-process response and a multi-state one:

        ```{python}
        gw.Outcome.surv(time="tstart", time2="tstop", event="status == 2")
        ```

        ```{python}
        gw.Outcome.surv(
            time="time",
            event="factor(status, c(0, 1, 2), c('censor', 'transplant', 'death'))",
        )
        ```
        """
        if type is not None and type not in _SURV_TYPES:
            raise ValueError(
                f"`type` must be one of {', '.join(map(repr, _SURV_TYPES))}, not {type!r}."
            )
        arguments = [("time", _parse_argument(time, "time"))]
        if time2 is not None:
            arguments.append(("time2", _parse_argument(time2, "time2")))
        if event is not None:
            arguments.append(("event", _parse_argument(event, "event")))
        return cls._new(
            kind="surv", arguments=_surv_arguments(arguments, type), type=type, origin=float(origin)
        )

    @classmethod
    def event_time(
        cls, time: str | Duration, status: str, time_max: str | Duration | None = None
    ) -> Outcome:
        """Describe an `event_time()` response by column names.

        When bound to data, the columns are passed to `event_time()`, so its validation applies:
        status codes are `"e"` (exact), `"r"` (right-censored), `"l"` (left-censored), or `"i"`
        (interval-censored, with the upper bound in `time_max`).

        Parameters
        ----------
        time
            The time column (the lower bound for interval-censored rows), or `duration(...)`.
        status
            The column of status codes.
        time_max
            The column of interval upper bounds, missing for other rows.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.event_time(time="time", status="code", time_max="upper")
        ```
        """
        arguments = [
            ("time", _parse_argument(time, "time")),
            ("status", _parse_argument(status, "status")),
        ]
        if time_max is not None:
            arguments.append(("time_max", _parse_argument(time_max, "time_max")))
        return cls._new(kind="event_time", arguments=tuple(arguments))

    @classmethod
    def first_event(
        cls,
        endpoints: Mapping[str, tuple[str | Duration, str]],
        *,
        censor_at: str | Duration | None = None,
        start: str | Duration | None = None,
    ) -> Outcome:
        """Describe a `first_event()` competing-risks response by column names.

        Each endpoint is a `(time, event)` pair of a time column and an event column or
        comparison (such as `"pstat == 1"`), in priority order for ties. When bound to data, the
        pairs are evaluated and passed to `first_event()`.

        Parameters
        ----------
        endpoints
            A mapping of state name to `(time, event)`.
        censor_at
            The column at which rows with no observed event are censored. The default is the
            latest endpoint time.
        start
            The column of entry times, for late entry.

        Returns
        -------
        Outcome
            A data-free description of the response.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.first_event(endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")})
        ```
        """
        if not endpoints:
            raise ValueError("`endpoints` must name at least one endpoint.")
        arguments: list[tuple[str, _Expr]] = []
        for i, (label, spec) in enumerate(endpoints.items()):
            if isinstance(spec, str) or len(spec) != 2:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"Endpoint {label!r} must be a `(time, event)` pair.")
            arguments.append((f"time{i}", _parse_argument(spec[0], f"endpoint {label!r} time")))
            arguments.append((f"event{i}", _parse_argument(spec[1], f"endpoint {label!r} event")))
        if censor_at is not None:
            arguments.append(("censor_at", _parse_argument(censor_at, "censor_at")))
        if start is not None:
            arguments.append(("start", _parse_argument(start, "start")))
        return cls._new(
            kind="first_event",
            arguments=tuple(arguments),
            labels=tuple(str(label) for label in endpoints),
        )

    @classmethod
    def from_formula(cls, formula: str) -> Outcome:
        """Parse a formula response such as `"Surv(time, status == 2)"`.

        The response is a call to `Surv()` or `event_time()`, with the arguments R's functions
        take: `Surv(time, time2, event, type=, origin=)` and `event_time(time, status, time_max)`.
        Arguments are column names and the expressions listed under `Outcome`. Column names that
        aren't valid Python identifiers are quoted with backticks, and dotted R names such as
        `ph.ecog` work without quoting.

        Parameters
        ----------
        formula
            The response, optionally followed by `~ 1`. To include covariates, pass the full
            formula to an estimator's fit method instead.

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

        R's other forms work too:

        ```{python}
        gw.Outcome.from_formula(formula="Surv(tstart, tstop, status %in% c(1, 2))")
        ```

        ```{python}
        gw.Outcome.from_formula(formula="event_time(time, code, upper)")
        ```
        """
        outcome, rhs = parse_formula(formula)
        if rhs is not None:
            raise ValueError(
                f"`Outcome.from_formula()` takes only a response, but {formula!r} has a right-hand "
                "side. Pass the full formula to an estimator's `fit()` instead."
            )
        return outcome

    # -- views --------------------------------------------------------------------------------

    @property
    def kind(self) -> str:
        """Which function builds the response: `"surv"`, `"event_time"`, or `"first_event"`.

        An `Outcome` from `Outcome.surv()` (or a `Surv(...)` formula) is bound with `Surv()`. One
        from `Outcome.event_time()` (or an `event_time(...)` formula) is bound with `event_time()`
        and then converted with `as_surv()`. One from `Outcome.first_event()` is bound with
        `first_event()`. Every kind produces a `Surv` when bound.

        Returns
        -------
        str
            `"surv"`, `"event_time"`, or `"first_event"`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.from_formula(formula="event_time(time, code)").kind
        ```
        """
        return self._kind

    @property
    def arguments(self) -> dict[str, str]:
        """The arguments of the response, by name, written as they would be in a formula.

        For a `surv` outcome the names are those of `Surv()` that were given (`time=`, `time2=`,
        `event=`), and for an `event_time` outcome those of `event_time()` (`time=`, `status=`,
        `time_max=`). Each value is a column name or an expression such as `"status == 2"`. As in
        R, `Surv(time, x)` without `event=` reads `x` as the status, so it is listed as `event=`.
        For a `first_event` outcome the endpoints are in `endpoints`, and this holds only
        `censor_at=` and `start=` when they were given. The type and origin have their own
        properties, `Outcome.type` and `Outcome.origin`.

        Returns
        -------
        dict of str to str
            Each argument name mapped to its expression text, in argument order.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.from_formula(formula="Surv(tstart, tstop, status == 2)").arguments
        ```
        """
        return {
            arg: _expr_source(expr)
            for arg, expr in self._arguments
            if not (self._kind == "first_event" and _ENDPOINT_ARGUMENT.fullmatch(arg))
        }

    @property
    def type(self) -> str | None:
        """The `type=` passed to `Surv()`, or `None` to let it be inferred when the data is bound.

        One of R's types: `"right"`, `"left"`, `"interval"`, `"counting"`, `"interval2"`, or
        `"mstate"`. `None`, the usual case, means `Surv()` infers right-censored or
        counting-process data from the arguments, or multi-state data from a `factor()` status.
        The type of the bound response is reported by `Surv.type`. Always `None` for
        `event_time` and `first_event` outcomes.

        Returns
        -------
        str or None
            The requested type, or `None`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.surv(time="lower", time2="upper", type="interval2").type
        ```
        """
        return self._type

    @property
    def origin(self) -> float:
        """The `origin=` passed to `Surv()`: a time subtracted from every time when bound.

        `0` (the default) leaves the times unchanged. A non-zero origin shifts every time column,
        as R's `Surv(origin=)` does. Always `0` for `event_time` and `first_event` outcomes.

        Returns
        -------
        float
            The origin.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Outcome.from_formula(formula="Surv(age_at_exit, died, origin=40)").origin
        ```
        """
        return self._origin

    @property
    def endpoints(self) -> dict[str, tuple[str, str]] | None:
        """For a `first_event` outcome, each endpoint's time and event expressions.

        The endpoints are in priority order, which decides ties: when two endpoints happen at the
        same time, the first one listed wins. Each value is a `(time, event)` pair of expression
        texts, as passed to `Outcome.first_event()`. `None` for other kinds of outcome.

        Returns
        -------
        dict of str to tuple of str, or None
            Each state name mapped to its `(time, event)` expressions, or `None`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        cr = gw.Outcome.first_event(
            endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death == 1")}
        )
        cr.endpoints
        ```
        """
        if self._kind != "first_event":
            return None
        args = dict(self._arguments)
        return {
            label: (_expr_source(args[f"time{i}"]), _expr_source(args[f"event{i}"]))
            for i, label in enumerate(self._labels)
        }

    @property
    def column_names(self) -> tuple[str, ...]:
        """The data columns this response reads, in the order they first appear.

        These are the columns named by every argument, including both columns of a `duration()`
        and the column inside a comparison or `factor()`. When a model is fitted from an `Outcome`
        and `data=`, rows with a missing value in any of these columns are dropped (together with
        any covariate columns) before the response is built.

        Returns
        -------
        tuple of str
            The column names, each listed once.

        Examples
        --------
        ```{python}
        import greenwood as gw

        died = gw.Outcome.from_formula(formula="Surv(duration(enroll, exit), outcome == 'died')")
        died.column_names
        ```
        """
        names: list[str] = []
        for _, expr in self._arguments:
            names += _expr_columns(expr)
        return tuple(dict.fromkeys(names))

    def __repr__(self) -> str:
        args = dict(self._arguments)
        if self._kind == "first_event":
            pairs = ", ".join(
                f"{label!r}: "
                f"({_expr_source(args[f'time{i}'])!r}, {_expr_source(args[f'event{i}'])!r})"
                for i, label in enumerate(self._labels)
            )
            parts = [f"endpoints={{{pairs}}}"]
            parts += [f"{a}={_expr_source(args[a])!r}" for a in ("censor_at", "start") if a in args]
            return f"Outcome.first_event({', '.join(parts)})"
        parts = [f"{a}={_expr_source(e)!r}" for a, e in self._arguments]
        if self._type is not None:
            parts.append(f"type={self._type!r}")
        if self._origin:
            parts.append(f"origin={self._origin!r}")
        return f"Outcome.{self._kind}({', '.join(parts)})"

    # -- binding --------------------------------------------------------------------------------

    def bind(self, data: Any) -> Surv:
        """Build the `Surv` response for the columns of `data`.

        Each argument is evaluated against `data` and the results go through `Surv()`,
        `event_time()` (then `as_surv()`), or `first_event()`. Missing values are kept, as they are
        by `Surv()`. To drop incomplete rows together with the covariates, pass the `Outcome` to an
        estimator's fit method with `data=` instead.

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
        gw.Outcome.surv(time="time", event="status").bind(lung)
        ```
        """
        get = _column_getter(data, list(self.column_names))
        values = {arg: _evaluate(expr, get) for arg, expr in self._arguments}
        if self._kind == "surv":
            return Surv(
                values["time"],
                values.get("time2"),
                values.get("event"),
                type=self._type,
                origin=self._origin,
            )
        if self._kind == "event_time":
            return as_surv(event_time(values["time"], values["status"], values.get("time_max")))
        endpoints = {
            label: (values[f"time{i}"], values[f"event{i}"]) for i, label in enumerate(self._labels)
        }
        return first_event(endpoints, censor_at=values.get("censor_at"), start=values.get("start"))


def _column_getter(data: Any, names: list[str]) -> Any:
    """Return `get(name)` for the named columns of a frame (a Narwhals series) or a mapping."""
    if isinstance(data, Mapping):
        mapping: Mapping[Any, Any] = data  # pyright: ignore[reportUnknownVariableType]
        check_names(names, [str(k) for k in mapping])

        def from_mapping(name: str) -> Any:
            return mapping[name]

        return from_mapping
    try:
        frame: Any = nw.from_native(data)
    except TypeError as error:
        raise TypeError(
            "`data` must be a DataFrame (pandas, Polars, PyArrow, DuckDB, ...) or a mapping of "
            f"column names to values. Got {type(data).__name__}."
        ) from error
    if not isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        raise TypeError(f"`data` must be a DataFrame, not a {type(data).__name__}.")
    _check_columns(names, [str(c) for c in frame.collect_schema().names()])
    selected = select_columns(frame, names)

    def from_frame(name: str) -> Any:
        return selected.get_column(name)

    return from_frame


# -- formula parsing ------------------------------------------------------------------------


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
    """Parse `Surv(...)` or `event_time(...)` into an `Outcome`."""
    parser = _Parser(text)
    call = parser.parse()
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        raise ValueError(
            "The formula response must be a call to `Surv(...)` or `event_time(...)`. "
            f"Got {text!r}."
        )
    func = call.func.id
    if func not in ("Surv", "event_time"):
        raise ValueError(
            f"The formula response must be a call to `Surv(...)` or `event_time(...)`, not "
            f"`{func}(...)`."
        )
    names = ("time", "time2", "event") if func == "Surv" else ("time", "status", "time_max")
    allowed = {*names, "type", "origin"} if func == "Surv" else set(names)
    if len(call.args) > 3:
        raise ValueError(f"`{func}()` takes at most 3 positional arguments.")
    given: dict[str, ast.expr] = dict(zip(names, call.args, strict=False))
    for kw in call.keywords:
        if kw.arg is None or kw.arg not in allowed or kw.arg in given:
            raise ValueError(
                f"Unknown or repeated `{func}()` argument `{kw.arg}`. "
                f"`{func}()` takes {', '.join(sorted(allowed))}."
            )
        given[kw.arg] = kw.value

    if func == "event_time":
        if "time" not in given or "status" not in given:
            raise ValueError("`event_time()` needs `time` and `status`.")
        arguments = tuple(
            (arg, parser.expr(given[arg], f"`{arg}`")) for arg in names if arg in given
        )
        return Outcome._new(kind="event_time", arguments=arguments)

    if "time" not in given:
        raise ValueError("`Surv()` needs a `time` argument.")
    ctype = parser.literal(given.pop("type"), "`type=`") if "type" in given else None
    if ctype is not None and ctype not in _SURV_TYPES:
        raise ValueError(
            f"`type` must be one of {', '.join(map(repr, _SURV_TYPES))}, not {ctype!r}."
        )
    origin = float(parser.literal(given.pop("origin"), "`origin=`")) if "origin" in given else 0.0
    # R reads a second argument without `event` as the status, so name it that way in errors.
    status_in_time2 = "event" not in given and ctype in (None, "right", "left", "mstate")

    def what(arg: str) -> str:
        return "the status" if status_in_time2 and arg == "time2" else f"`{arg}`"

    arguments = [(arg, parser.expr(given[arg], what(arg))) for arg in names if arg in given]
    return Outcome._new(
        kind="surv", arguments=_surv_arguments(arguments, ctype), type=ctype, origin=origin
    )


def _surv_arguments(
    arguments: list[tuple[str, _Expr]], ctype: str | None
) -> tuple[tuple[str, _Expr], ...]:
    """Apply R's rule that `Surv(time, x)` without `event` reads `x` as the status.

    R binds the second argument to `time2` and then, for right, left, and multi-state data, uses it
    as `event`. Storing it as `event` gives one spelling, so equal responses compare equal.
    """
    names = [arg for arg, _ in arguments]
    if names == ["time", "time2"] and ctype in (None, "right", "left", "mstate"):
        arguments = [arguments[0], ("event", arguments[1][1])]
    for arg, expr in arguments:
        if arg == "event" and isinstance(expr, Duration):
            raise ValueError("The status can't be a `duration()`. Durations are times.")
    return tuple(arguments)


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
_SPECIAL = re.compile(r"(strata|cluster|frailty(?:\.(?:gamma|gaussian))?)\((.*)\)", flags=re.S)

# The label argument each special term fills.
_SPECIAL_TARGET = {"strata": "strata", "cluster": "cluster", "frailty": "frailty_cluster"}

# R's frailty distributions and their Greenwood names.
_FRAILTY_DIST = {"gamma": "gamma", "gaussian": "lognormal", "lognormal": "lognormal"}


def _special_term(term: str) -> tuple[str, str, list[str], dict[str, Any]] | None:
    """Recognize `strata(a, b)`, `cluster(id)`, and `frailty(id)` terms.

    Returns the term name, the label argument it fills, its columns, and any options (the frailty
    distribution). `frailty(id)` is gamma, as in R. `frailty(id, distribution="gaussian")`,
    `frailty.gaussian(id)`, and `frailty.gamma(id)` choose the distribution explicitly.
    """
    match = _SPECIAL.fullmatch(term)
    if match is None:
        return None
    head = match.group(1)
    name = head.split(".")[0]
    columns: list[str] = []
    options: dict[str, Any] = {}
    for part in (p.strip() for p in match.group(2).split(",")):
        if "=" in part and name == "frailty":
            key, _, value = (x.strip() for x in part.partition("="))
            dist = value.strip("\"'")
            if key != "distribution" or dist not in _FRAILTY_DIST:
                raise ValueError(
                    f"`{term}` is not supported. Use `frailty(id)`, or choose the distribution "
                    'with `distribution="gamma"` or `distribution="gaussian"` (lognormal).'
                )
            options["frailty"] = _FRAILTY_DIST[dist]
        else:
            columns.append(_plain_name(part))
    if name == "frailty":
        dist = head.split(".")[1] if "." in head else None
        if dist is not None:
            if options.get("frailty", _FRAILTY_DIST[dist]) != _FRAILTY_DIST[dist]:
                raise ValueError(f"`{term}` names two different frailty distributions.")
            options["frailty"] = _FRAILTY_DIST[dist]
        options.setdefault("frailty", "gamma")
        if len(columns) != 1:
            raise ValueError(f"`{term}` must name exactly one cluster column.")
    return name, _SPECIAL_TARGET[name], columns, options


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
    options: dict[str, Any] = field(default_factory=lambda: {})


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
    options: dict[str, Any] = {}

    outcome: Outcome | None
    if isinstance(surv, EventTime):
        surv = as_surv(surv)
    if isinstance(surv, Surv):
        outcome = None
    elif isinstance(surv, Outcome):
        outcome = surv
    elif isinstance(surv, str):
        outcome, rhs = parse_formula(surv)
        if rhs is not None:
            _place_rhs(rhs, rhs_to, designs, labels, estimator, options)
    else:
        raise TypeError(
            "The response must be a `Surv`, an `EventTime`, an `Outcome`, or a formula string "
            "such as "
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
        bound = _bind_plain(surv, data, designs, labels)
    elif data is None:
        raise ValueError(
            "An `Outcome` or formula names columns, so `fit()` needs the frame that holds them. "
            "Pass `data=`."
        )
    else:
        bound = _bind_outcome(outcome, data, designs, labels)
    bound.options = options
    _drop_missing_response(bound)
    if "weights" in labels:
        bound.labels["weights"] = _case_weights(bound.labels["weights"], bound.surv)
    return bound


def _drop_missing_response(bound: BoundInputs) -> None:
    """Drop rows whose response is missing, as R's default `na.action = na.omit` does.

    A `Surv` keeps missing values (from missing inputs, invalid status codes, or backwards
    intervals). Those rows are removed here, together with the matching rows of every row-aligned
    argument and of `data`, and counted in `n_dropped`.
    """
    missing = bound.surv.is_na()
    if not missing.any():
        return
    keep = ~missing
    n = bound.surv.n
    bound.surv = bound.surv[keep]
    for group in (bound.labels, bound.designs):
        for key, value in group.items():
            if value is None or isinstance(value, (str, _GroupTerms)) or _is_name_list(value):
                continue
            if _row_count(value) == n:
                group[key] = _filter_rows(value, keep)
    if bound.data is not None and not isinstance(bound.data, Mapping):
        frame = _to_frame(bound.data)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        if int(frame.shape[0]) == n:
            bound.data = frame.filter(keep.tolist()).to_native()
    bound.n_dropped += int(missing.sum())


def _case_weights(weights: Any, surv: Surv) -> Array | None:
    """Validate case weights given to `fit(weights=)`: finite, strictly positive, one per row."""
    if weights is None:
        return None
    w = np.asarray(as_1d(weights), dtype=np.float64)
    if w.shape[0] != surv.n:
        raise ValueError(f"`weights` has {w.shape[0]} values, but the response has {surv.n} rows.")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("`weights` must be finite and strictly positive.")
    return w


def _place_rhs(
    rhs: str,
    rhs_to: str | None,
    designs: dict[str, Any],
    labels: dict[str, Any],
    estimator: str,
    options: dict[str, Any],
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
        term_name, target, names, term_options = special
        if target not in labels:
            if grouping and term_name == "strata":
                # survfit() treats `strata(x)` as an ordinary grouping term.
                terms += names
                continue
            raise ValueError(f"{estimator} does not support `{term_name}()` terms in a formula.")
        if labels[target] is not None:
            raise ValueError(
                f"`{target}` was given both in the formula and as `{target}=`. Use one."
            )
        labels[target] = _GroupTerms(tuple(names))
        options.update(term_options)

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
