"""R-identical formatting of numeric vectors.

`format_real()` reproduces what R's `format()` does with a double vector: it chooses one layout
for the whole vector (fixed or scientific, with a shared number of decimals) so that every value
shows at most `digits` significant digits. This is a port of `scientific()` and `formatReal()` in
R's `src/main/format.c`, following R's plain-double code path (the one R takes on platforms where
`long double` is no wider than `double`, such as macOS on Apple silicon).

It exists so printed output can match R character for character, which the `event_time` port
needs (`format.event_time()` formats every number in a vector together).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["format_real"]

# R's KP_MAX: the largest power of ten held in its lookup table.
_KP_MAX = 22
_TBL = [10.0**k for k in range(_KP_MAX + 1)]

# R's R_dec_min_exponent for doubles (DBL_MIN_10_EXP).
_DEC_MIN_EXPONENT = -307


def _scientific(x: float, digits: int) -> tuple[bool, int, int, bool]:
    """Port of R's `scientific()`.

    Returns `(negative, kpower, nsig, rounding_widens)`: whether `x` is negative, its decimal
    exponent after rounding to `digits` significant digits, how many of those digits are
    significant (trailing zeros dropped), and whether rounding pushed it up to the next power of
    ten in scientific form only.
    """
    if x == 0.0:
        return False, 0, 1, False

    negative = x < 0.0
    r = -x if negative else x

    kp = math.floor(math.log10(r)) - digits + 1
    r_prec = r
    if abs(kp) < 10:
        if kp > 0:
            r_prec /= _TBL[kp]
        elif kp < 0:
            r_prec *= _TBL[-kp]
    elif kp <= _DEC_MIN_EXPONENT:
        r_prec = (r * 1e303) / 10.0 ** (kp + 303)
    else:
        r_prec /= 10.0**kp
    if r_prec < _TBL[digits - 1]:
        r_prec *= 10.0
        kp -= 1

    # nearbyint() under the default rounding mode: round half to even, as Python's round().
    alpha = float(round(r_prec))
    nsig = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1

    # Scientific form can round further than fixed form (9996 with 3 digits is 1e+04 but 9996).
    rgt = min(max(digits - kpower, 0), _KP_MAX)
    fuzz = 0.5 / _TBL[rgt]
    rounding_widens = 0 < kpower <= _KP_MAX and r < _TBL[kpower] - fuzz
    return negative, kpower, nsig, rounding_widens


def _layout(values: Sequence[float], digits: int) -> tuple[int, int]:
    """Port of R's `formatReal()` layout choice. Returns `(decimals, exponent_digits)`.

    `exponent_digits` is `0` for fixed notation, otherwise `1` (two-digit exponent) or `2`
    (three-digit exponent).
    """
    neg = 0
    rgt = mxl = mxsl = mxns = None
    mnl = None
    for x in values:
        if not math.isfinite(x):
            continue
        negative, kpower, nsig, rounding_widens = _scientific(x, digits)
        left = kpower + 1
        if rounding_widens:
            left -= 1
        sleft = int(negative) + (1 if left <= 0 else left)
        right = nsig - left
        if negative:
            neg = 1
        rgt = right if rgt is None else max(rgt, right)
        mxl = left if mxl is None else max(mxl, left)
        mnl = left if mnl is None else min(mnl, left)
        mxsl = sleft if mxsl is None else max(mxsl, sleft)
        mxns = nsig if mxns is None else max(mxns, nsig)

    if mxl is None or rgt is None or mxsl is None or mxns is None or mnl is None:
        # Every value is non-finite.
        return 0, 0

    if mxl < 0:
        mxsl = 1 + neg
    rgt = max(rgt, 0)
    width_fixed = mxsl + rgt + (rgt != 0)

    exponent_digits = 2 if (mxl > 100 or mnl <= -99) else 1
    decimals = mxns - 1
    width_sci = neg + (decimals > 0) + decimals + 4 + exponent_digits
    # R_print.scipen is 0 by default.
    if width_fixed <= width_sci:
        return rgt, 0
    return decimals, exponent_digits


def _encode(x: float, decimals: int, exponent_digits: int) -> str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if exponent_digits:
        return f"{x:.{decimals}e}"
    return f"{x:.{decimals}f}"


def format_real(values: Sequence[float], *, digits: int = 7) -> list[str]:
    """Format numbers together exactly as R's `format(x, trim = TRUE)` does.

    Every value shares one layout, chosen from all finite values: fixed notation with a common
    number of decimals, or scientific notation when that is narrower. Non-finite values format as
    `"NaN"`, `"Inf"`, and `"-Inf"`. (R writes a missing value as `"NA"`; callers map those
    themselves, since Python has no separate missing double.)
    """
    decimals, exponent_digits = _layout(values, digits)
    return [_encode(x, decimals, exponent_digits) for x in values]
