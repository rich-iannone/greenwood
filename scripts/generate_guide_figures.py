#!/usr/bin/env python
"""Regenerate the inline SVG figures in the introductory user-guide pages.

The figures in `user_guide/03-survival-data.qmd` and `user_guide/05-kaplan-meier.qmd` are inline
SVG, drawn here from a few lines of geometry so their coordinates stay consistent. Each page holds
one style block (the CSS custom properties that switch the figures between Quarto's light and dark
themes) and one `::: {#fig-...}` block per figure. This script rewrites those blocks in place.

Run from the repo root after changing a figure here:

    .venv/bin/python scripts/generate_guide_figures.py           # update the pages
    .venv/bin/python scripts/generate_guide_figures.py --check   # fail if the pages are stale
    .venv/bin/python scripts/generate_guide_figures.py --preview out.html  # standalone preview

Design rules (see the dataviz guidance used to build them):

- Colors come from CSS custom properties on `.gw-fig`, defined for the light theme and for
  `body.quarto-dark`. The palette (blue, orange, aqua) was validated for color-vision deficiency
  and contrast against both page surfaces. Aqua is below 3:1 on the light surface, so every aqua
  mark carries a text label.
- Text uses the page ink (`currentColor`) or a muted ink, never a series color.
- Marks differ by shape as well as color: a filled dot is an event, an open circle is censored,
  a shaded band is where the event could be, and a dashed line is time not observed.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "user_guide"
FENCE = "```"

# Palette validated with the dataviz skill's validator (light and dark, all pairs).
STYLE = """<style>
.gw-fig { --gw-event: #2a78d6; --gw-window: #eb6834; --gw-entry: #1baf7a;
  --gw-surface: #fcfeff; --gw-muted: rgba(0, 0, 0, 0.62); --gw-faint: rgba(0, 0, 0, 0.16);
  --gw-box: rgba(0, 0, 0, 0.035);
  margin: 1.5rem 0; }
body.quarto-dark .gw-fig { --gw-surface: #1a1a1a; --gw-event: #3987e5; --gw-window: #d95926;
  --gw-entry: #199e70;
  --gw-muted: rgba(255, 255, 255, 0.66); --gw-faint: rgba(255, 255, 255, 0.2);
  --gw-box: rgba(255, 255, 255, 0.05); }
.gw-fig svg { display: block; width: 100%; height: auto; overflow: visible; }
.gw-fig text { fill: currentColor; font-family: inherit; }
.gw-fig .t-muted { fill: var(--gw-muted); }
.gw-fig .t-mono {
  font-family: var(--bs-font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace); }
.gw-fig .t-bold { font-weight: 600; }
.gw-fig .ink { stroke: currentColor; fill: none; }
.gw-fig .muted { stroke: var(--gw-muted); fill: none; }
.gw-fig .faint { stroke: var(--gw-faint); fill: none; }
.gw-fig .event { fill: var(--gw-event); stroke: none; }
.gw-fig .event-line { stroke: var(--gw-event); fill: none; }
.gw-fig .censor { fill: var(--gw-surface, Canvas); stroke: currentColor; }
.gw-fig .window { fill: var(--gw-window); stroke: none; opacity: 0.45; }
.gw-fig .window-line { stroke: var(--gw-window); fill: none; }
.gw-fig .entry-line { stroke: var(--gw-entry); fill: none; }
.gw-fig .box { fill: var(--gw-box); stroke: var(--gw-faint); }
.gw-fig .arrowhead { fill: var(--gw-muted); }
.gw-fig .arrowhead-window { fill: var(--gw-window); }
body.quarto-dark .gw-fig .window { opacity: 0.75; }
.gw-fig .group { fill: none; stroke: var(--gw-faint); stroke-dasharray: 5 5; }
</style>"""


def svg(width: int, height: int, body: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{label}">\n'
        '<defs><marker id="gw-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrowhead" d="M0,0 L10,5 L0,10 z"/>'
        "</marker>"
        '<marker id="gw-arrow-window" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path class="arrowhead-window" d="M0,0 L10,5 L0,10 z"/>'
        "</marker></defs>\n" + body + "\n</svg>"
    )


def text(
    x: float, y: float, s: str, *, size: float = 14, anchor: str = "start", cls: str = ""
) -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{c}>{s}</text>'


def line(
    x1: float, y1: float, x2: float, y2: float, cls: str = "ink", w: float = 3, extra: str = ""
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" '
        f'stroke-width="{w}" stroke-linecap="round"{extra}/>'
    )


def band(x1: float, x2: float, y: float) -> str:
    """A shaded band from `x1` to `x2`, centred on `y`: where the event could be."""
    return (
        f'<rect x="{x1:.1f}" y="{y - 7:.1f}" width="{x2 - x1:.1f}" height="14" rx="3" '
        'class="window"/>'
    )


def group_frame(x: float, y: float, w: float, h: float) -> str:
    """A dashed frame around a group of alternatives in the flow diagram."""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="group" stroke-width="1.5"/>'
    )


def event_dot(x: float, y: float, r: float = 7) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="event"/>'


def censor_dot(x: float, y: float, r: float = 7) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="censor" stroke-width="2.5"/>'


def axis(
    x0: float, x1: float, y: float, ticks: list[tuple[float, str]], label: str, label_y: float
) -> str:
    parts = [line(x0, y, x1, y, "muted", 1.5)]
    for x, s in ticks:
        parts.append(line(x, y, x, y + 6, "muted", 1.5))
        parts.append(text(x, y + 22, s, size=13, anchor="middle", cls="t-muted"))
    parts.append(text((x0 + x1) / 2, label_y, label, size=13, anchor="middle", cls="t-muted"))
    return "\n".join(parts)


def legend(items: list[tuple[str, str]], x: float, y: float, gap: float = 26) -> str:
    parts = []
    for kind, label in items:
        if kind == "event":
            parts.append(event_dot(x + 7, y - 5, 6))
        elif kind == "censor":
            parts.append(censor_dot(x + 7, y - 5, 6))
        elif kind == "window":
            parts.append(
                f'<rect x="{x:.1f}" y="{y - 11:.1f}" width="18" height="12" rx="2" class="window"/>'
            )
        elif kind == "entry":
            parts.append(line(x, y - 5, x + 18, y - 5, "entry-line", 3, ' stroke-dasharray="4 4"'))
        parts.append(text(x + 26, y, label, size=13))
        x += 26 + 7.4 * len(label) + gap
    return "\n".join(parts)


# -- Figure 1: what follow-up looks like, and what gets recorded --------------------------------


def follow_up() -> str:
    x0, scale = 120, 15.0  # months 0..24 -> x 120..480

    def X(m: float) -> float:
        return x0 + m * scale

    rows = [
        ("Patient A", 8, "event", "8", "event"),
        ("Patient B", 24, "censor", "24", "censored"),
        ("Patient C", 13, "censor", "13", "censored"),
        ("Patient D", 17, "event", "17", "event"),
    ]
    notes = {"Patient B": "study ended", "Patient C": "moved away"}
    top, step = 74, 46
    parts = [
        legend(
            [("event", "Relapse observed"), ("censor", "Censored: relapse-free when last seen")],
            20,
            22,
        ),
        line(X(24), 44, X(24), top + step * 3 + 18, "muted", 1.5, ' stroke-dasharray="5 5"'),
        text(560, 52, "What is recorded", size=14, cls="t-bold"),
        text(560, 72, "time", size=13, cls="t-muted"),
        text(630, 72, "status", size=13, cls="t-muted"),
    ]
    for i, (name, t, kind, rec_t, rec_s) in enumerate(rows):
        y = top + 18 + step * i
        parts.append(text(x0 - 14, y + 5, name, size=14, anchor="end"))
        parts.append(line(X(0), y, X(t), y, "ink", 3))
        if kind == "event":
            parts.append(event_dot(X(t), y))
        else:
            # The true relapse time is unknown and lies somewhere later.
            parts.append(
                line(
                    X(t) + 10,
                    y,
                    X(t) + 48,
                    y,
                    "window-line",
                    2.5,
                    ' stroke-dasharray="3 5" marker-end="url(#gw-arrow-window)"',
                )
            )
            parts.append(censor_dot(X(t), y))
            parts.append(text(X(t) + 56, y + 5, "?", size=15, cls="t-bold"))
            if name in notes:
                parts.append(
                    text(X(t) - 8, y - 12, notes[name], size=12, anchor="end", cls="t-muted")
                )
        parts.append(text(560, y + 5, rec_t, size=14, cls="t-mono"))
        marker = event_dot(636, y, 5) if kind == "event" else censor_dot(636, y, 5)
        parts.append(marker)
        parts.append(text(648, y + 5, rec_s, size=14))
    parts.append(
        text(X(24), top + step * 3 + 34, "end of study", size=12, anchor="middle", cls="t-muted")
    )
    ticks = [(X(m), str(m)) for m in range(0, 25, 4)]
    parts.append(
        axis(X(0), X(24), top + step * 3 + 50, ticks, "Months since enrolment", top + step * 3 + 92)
    )
    height = top + step * 3 + 102
    return svg(
        720,
        int(height),
        "\n".join(parts),
        "Follow-up of four patients and the time and status recorded for each",
    )


# -- Figure 2: the kinds of incomplete observation -------------------------------------------------


def kinds() -> str:
    x0, scale = 230, 14.0  # 0..20 -> 230..510

    def X(m: float) -> float:
        return x0 + m * scale

    head = 30
    rows_y = [head + 52 + 64 * i for i in range(5)]
    parts = [
        text(20, head, "What we know about one subject", size=14, cls="t-bold"),
        text(560, head, "Written as", size=14, cls="t-bold"),
    ]
    labels = [
        ("Exact event", "The event was seen when it happened."),
        ("Right-censored", "Event-free when last seen."),
        ("Left-censored", "Already happened by the first check."),
        ("Interval-censored", "Happened between two visits."),
        ("Late entry", "Only observed after joining the study."),
    ]
    codes = [
        ('status "e"', "event at 8"),
        ('status "r"', "event after 12"),
        ('status "l"', "event before 8"),
        ('status "i"', "event in (4, 12]"),
        ("Surv(t0, t1, event)", "start, stop, event"),
    ]
    for (name, desc), (code, note), y in zip(labels, codes, rows_y, strict=True):
        parts.append(text(20, y - 3, name, size=14, cls="t-bold"))
        parts.append(text(20, y + 15, desc, size=12, cls="t-muted"))
        parts.append(text(560, y - 3, code, size=13, cls="t-mono"))
        parts.append(text(560, y + 15, note, size=12, cls="t-muted"))
        parts.append(line(X(0), y + 24, X(20), y + 24, "faint", 1))
    y1, y2, y3, y4, y5 = rows_y
    # Exact: observed up to the event at 8.
    parts += [line(X(0), y1, X(8), y1), event_dot(X(8), y1)]
    # Right-censored: seen event-free to 12, event somewhere later.
    parts += [
        band(X(12), X(20), y2),
        line(X(0), y2, X(12), y2),
        censor_dot(X(12), y2),
        text(X(16), y2 - 13, "somewhere here", size=12, anchor="middle", cls="t-muted"),
    ]
    # Left-censored: first check at 8 finds the event already happened.
    parts += [
        band(X(0), X(8), y3),
        line(X(8), y3 - 13, X(8), y3 + 13, "ink", 3),
        text(X(8) + 8, y3 + 5, "first check", size=12, cls="t-muted"),
        text(X(4), y3 - 13, "somewhere here", size=12, anchor="middle", cls="t-muted"),
    ]
    # Interval-censored: event-free at the visit at 4, event found at the visit at 12.
    parts += [
        band(X(4), X(12), y4),
        line(X(0), y4, X(4), y4),
        line(X(4), y4 - 13, X(4), y4 + 13, "ink", 3),
        line(X(12), y4 - 13, X(12), y4 + 13, "ink", 3),
        text(X(4), y4 - 18, "visit", size=12, anchor="middle", cls="t-muted"),
        text(X(12), y4 - 18, "visit", size=12, anchor="middle", cls="t-muted"),
    ]
    # Late entry: not observable before joining at 6, then followed to an event at 15.
    parts += [
        line(X(0), y5, X(6), y5, "entry-line", 3, ' stroke-dasharray="4 5"'),
        text(X(3), y5 - 12, "not yet in study", size=12, anchor="middle", cls="t-muted"),
        line(X(6), y5, X(15), y5),
        line(X(6), y5 - 10, X(6), y5 + 10, "ink", 3),
        event_dot(X(15), y5),
    ]
    bottom = rows_y[-1] + 44
    ticks = [(X(m), str(m)) for m in range(0, 21, 4)]
    parts.append(axis(X(0), X(20), bottom, ticks, "Time", bottom + 40))
    parts.append(
        legend(
            [
                ("event", "Event"),
                ("censor", "Censored"),
                ("window", "Where the event could be"),
                ("entry", "Not observed"),
            ],
            20,
            bottom + 72,
            gap=18,
        )
    )
    return svg(
        720,
        int(bottom + 86),
        "\n".join(parts),
        "Exact, right-censored, left-censored, interval-censored, and late-entry observations",
    )


# -- Figure 3: the ways to build a response --------------------------------------------------------


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    sub: str = "",
    *,
    code: bool = False,
    bold: bool = True,
) -> str:
    cx = x + w / 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="box" stroke-width="1.5"/>'
    ]
    if sub:
        parts.append(
            text(cx, y + h / 2 - 5, title, size=14, anchor="middle", cls="t-bold" if bold else "")
        )
        parts.append(
            text(
                cx,
                y + h / 2 + 15,
                sub,
                size=12 if code else 12.5,
                anchor="middle",
                cls="t-mono" if code else "t-muted",
            )
        )
    else:
        parts.append(
            text(cx, y + h / 2 + 5, title, size=14, anchor="middle", cls="t-bold" if bold else "")
        )
    return "\n".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return line(x1, y1, x2, y2, "muted", 1.8, ' marker-end="url(#gw-arrow)"')


def flow() -> str:
    L, R, W = 10, 370, 340  # left/right column x and width
    lc, rc = L + W / 2, R + W / 2
    parts = [
        box(250, 10, 220, 46, "Where is your data?"),
        arrow(300, 56, lc, 96),
        arrow(420, 56, rc, 96),
        box(L, 100, W, 56, "In a data frame", "pandas, Polars, PyArrow, DuckDB, lazy frames"),
        box(R, 100, W, 56, "Already in hand", "lists, NumPy arrays, simulated values"),
        arrow(lc, 156, lc, 180),
        arrow(rc, 156, rc, 180),
        group_frame(L - 4, 184, W + 8, 170),
        group_frame(R - 4, 184, W + 8, 170),
        box(
            L + 8,
            194,
            W - 16,
            58,
            "Name columns in a formula",
            '"Surv(time, status) ~ age"',
            code=True,
            bold=False,
        ),
        text(lc, 272, "or", size=13, anchor="middle", cls="t-muted"),
        box(
            L + 8,
            286,
            W - 16,
            58,
            "Describe it once with an Outcome",
            'gw.Outcome.surv(time="time", event="status")',
            code=True,
            bold=False,
        ),
        box(
            R + 8,
            194,
            W - 16,
            58,
            "Times and an event indicator",
            "gw.Surv(time=t, event=d)",
            code=True,
            bold=False,
        ),
        text(rc, 272, "or", size=13, anchor="middle", cls="t-muted"),
        box(
            R + 8,
            286,
            W - 16,
            58,
            "Event-time codes e, r, l, i",
            "gw.event_time(time=t, status=s)",
            code=True,
            bold=False,
        ),
        arrow(lc, 354, 320, 402),
        arrow(rc, 354, 400, 402),
        text(lc - 8, 384, "evaluated against data=", size=12.5, anchor="end", cls="t-muted"),
        text(rc + 8, 384, "converted with gw.as_surv()", size=12.5, cls="t-muted"),
        box(190, 406, 340, 58, "A Surv response", "right, left, interval, counting, multi-state"),
        arrow(360, 464, 360, 492),
        box(190, 496, 340, 46, "Any estimator's fit()", bold=False),
    ]
    return svg(
        720,
        550,
        "\n".join(parts),
        "Ways to build a survival response, from a data frame or from values",
    )


# -- Kaplan-Meier figures --------------------------------------------------------------------------

KM_DATA = [(3, 1), (5, 0), (7, 1), (9, 1), (11, 1), (14, 0)]  # (months, event)


def km_steps() -> list[tuple[float, float, int, int]]:
    """(time, survival after the time, at risk, events) at each event time."""
    s, at_risk, out = 1.0, len(KM_DATA), []
    for t, e in KM_DATA:
        if e:
            s *= (at_risk - 1) / at_risk
            out.append((t, s, at_risk, 1))
        at_risk -= 1
    return out


def km_swimmer() -> str:
    x0, scale = 130, 30.0  # 0..15 -> 130..580

    def X(m: float) -> float:
        return x0 + m * scale

    top, step = 62, 38
    parts = [text(x0 - 14, 34, "At risk:", size=13, anchor="end", cls="t-bold")]
    n_rows = len(KM_DATA)
    for t, _, at_risk, _ in km_steps():
        parts.append(
            line(
                X(t),
                44,
                X(t),
                top + step * (n_rows - 1) + 16,
                "faint",
                1.5,
                ' stroke-dasharray="4 4"',
            )
        )
        parts.append(text(X(t), 34, str(at_risk), size=14, anchor="middle", cls="t-bold"))
    for i, (t, e) in enumerate(KM_DATA):
        y = top + step * i
        parts.append(text(x0 - 14, y + 5, f"Patient {i + 1}", size=14, anchor="end"))
        parts.append(line(X(0), y, X(t), y))
        parts.append(event_dot(X(t), y) if e else censor_dot(X(t), y))
    axis_y = top + step * (n_rows - 1) + 30
    parts.append(
        axis(
            X(0),
            X(15),
            axis_y,
            [(X(m), str(m)) for m in range(0, 16, 3)],
            "Months since diagnosis",
            axis_y + 40,
        )
    )
    parts.append(
        legend([("event", "Died"), ("censor", "Censored: alive when last seen")], 130, axis_y + 72)
    )
    return svg(
        720,
        int(axis_y + 86),
        "\n".join(parts),
        "Six patients, with the number at risk at each death",
    )


def km_curve() -> str:
    x0, xs = 90, 30.0  # months 0..15 -> 90..540
    y_top, y_span = 30, 270  # survival 1 -> 30, 0 -> 300

    def X(m: float) -> float:
        return x0 + m * xs

    def Y(s: float) -> float:
        return y_top + (1 - s) * y_span

    parts = []
    for s in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(line(X(0), Y(s), X(15), Y(s), "faint", 1))
        parts.append(text(X(0) - 10, Y(s) + 4, f"{s:.2f}", size=12.5, anchor="end", cls="t-muted"))
    parts.append(
        axis(
            X(0),
            X(15),
            Y(0) + 4,
            [(X(m), str(m)) for m in range(0, 16, 3)],
            "Months since diagnosis",
            Y(0) + 46,
        )
    )
    parts.append(
        f'<text x="22" y="{Y(0.5):.1f}" font-size="13" text-anchor="middle" class="t-muted" '
        f'transform="rotate(-90 22 {Y(0.5):.1f})">Probability still alive</text>'
    )
    # The step function.
    pts = [f"M{X(0):.1f},{Y(1):.1f}"]
    for t, s_new, _, _ in km_steps():
        pts.append(f"H{X(t):.1f}")
        pts.append(f"V{Y(s_new):.1f}")
    pts.append(f"H{X(14):.1f}")
    parts.append(
        f'<path d="{"".join(pts)}" class="event-line" stroke-width="3" stroke-linejoin="round"/>'
    )
    # Censoring ticks on the curve.
    for t, e in KM_DATA:
        if not e:
            level = [s for (tt, s, _, _) in km_steps() if tt < t][-1]
            parts.append(line(X(t), Y(level) - 9, X(t), Y(level) + 9, "ink", 2.5))
    # Annotations.
    a = []
    a.append(text(X(3) + 12, Y(1) + 12, "Death at month 3: 1 of 6 at risk", size=12.5))
    a.append(text(X(3) + 12, Y(1) + 28, "1 × 5/6 = 0.83", size=12.5, cls="t-mono"))
    a.append(text(X(0) + 8, Y(0.62), "Censored at month 5:", size=12.5))
    a.append(text(X(0) + 8, Y(0.62) + 16, "no drop, one fewer", size=12.5))
    a.append(text(X(0) + 8, Y(0.62) + 32, "at risk from here on", size=12.5))
    a.append(line(X(0) + 100, Y(0.62) - 12, X(5) - 6, Y(5 / 6) + 12, "muted", 1.2))
    a.append(text(X(7) + 12, Y(0.833) + 26, "Death at month 7: 1 of 4 at risk", size=12.5))
    a.append(text(X(7) + 12, Y(0.833) + 42, "0.83 × 3/4 = 0.63", size=12.5, cls="t-mono"))
    a.append(text(X(9) + 12, Y(0.625) + 26, "Steps get bigger as fewer remain:", size=12.5))
    a.append(
        text(
            X(9) + 12, Y(0.625) + 42, "× 2/3 at month 9, × 1/2 at month 11", size=12.5, cls="t-mono"
        )
    )
    a.append(text(X(11) + 12, Y(0.208) - 34, "Last patient censored at month 14:", size=12.5))
    a.append(text(X(11) + 12, Y(0.208) - 18, "the curve stops at 0.21, not at 0", size=12.5))
    parts += a
    return svg(
        720, int(Y(0) + 60), "\n".join(parts), "Annotated Kaplan-Meier curve for the six patients"
    )


FIGURES: dict[str, Any] = {
    "follow_up": follow_up,
    "kinds": kinds,
    "flow": flow,
    "km_swimmer": km_swimmer,
    "km_curve": km_curve,
}


# Which page holds which figure.
PAGES: dict[str, list[str]] = {
    "03-survival-data.qmd": ["follow_up", "kinds", "flow"],
    "05-kaplan-meier.qmd": ["km_swimmer", "km_curve"],
}

# The Quarto figure label of each figure.
LABELS = {
    "follow_up": "fig-follow-up",
    "kinds": "fig-kinds",
    "flow": "fig-ways",
    "km_swimmer": "fig-km-patients",
    "km_curve": "fig-km-curve",
}

_STYLE_BLOCK = re.compile(
    re.escape(FENCE) + r"\{=html\}\n<style>\n\.gw-fig .*?</style>\n" + re.escape(FENCE), re.S
)


def _figure_block(label: str) -> re.Pattern[str]:
    """The raw-HTML part of a `::: {#label}` figure: from its html fence to the closing fence."""
    return re.compile(
        r"(::: \{#" + re.escape(label) + r"\}\n" + re.escape(FENCE) + r"\{=html\}\n)"
        r"<div class=\"gw-fig\">\n.*?\n</div>\n(" + re.escape(FENCE) + r")",
        re.S,
    )


def render_page(text: str, names: list[str]) -> str:
    """Return `text` with its style block and the named figures regenerated."""
    style = f"{FENCE}{{=html}}\n{STYLE}\n{FENCE}"
    text, n = _STYLE_BLOCK.subn(lambda _: style, text)
    if n != 1:
        raise ValueError("Expected exactly one figure style block.")
    for name in names:
        svg_text = FIGURES[name]()
        pattern = _figure_block(LABELS[name])
        text, n = pattern.subn(
            lambda m, s=svg_text: f'{m.group(1)}<div class="gw-fig">\n{s}\n</div>\n{m.group(2)}',
            text,
        )
        if n != 1:
            raise ValueError(f"Expected exactly one figure block labelled {LABELS[name]!r}.")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if any page is out of date")
    parser.add_argument("--preview", type=pathlib.Path, help="also write a standalone HTML preview")
    args = parser.parse_args()

    stale: list[str] = []
    for page, names in PAGES.items():
        path = GUIDE / page
        before = path.read_text()
        after = render_page(before, names)
        if after != before:
            stale.append(page)
            if not args.check:
                path.write_text(after)
    if args.preview:
        blocks = "".join(
            f"<h3>{name}</h3><div class='gw-fig'>{make()}</div>" for name, make in FIGURES.items()
        )
        args.preview.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>" + STYLE + "</head>"
            "<body class='quarto-light' style='max-width:760px;margin:2rem auto'>"
            + blocks
            + "</body></html>"
        )
    if args.check:
        if stale:
            print("Out of date: " + ", ".join(stale) + ". Run without --check to update.")
            return 1
        print("All guide figures are up to date.")
        return 0
    print("Updated: " + ", ".join(stale) if stale else "Nothing to update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
