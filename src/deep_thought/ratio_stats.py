"""IAAO ratio statistics - ONE implementation, shared by the notebook and the app.

DON'T PANIC. 🐬  This module is thin on purpose. It does almost no maths.

Why it exists
=============

This project had **three** implementations of COD:

  1. `pyRealEstate.RealEstateMetrics.COD`, which the notebook uses
  2. a module-level `COD()` in `app/NAL_Navigator.py`
  3. a `DOR_COD()` method inside the app's report generator

They were checked against each other across eleven cases - float arrays,
pandas Series, nullable `Int64`, `pd.NA`, a zero divisor, a single row - and
on every realistic input they agreed exactly. Two of them were in fact
character-for-character identical source.

So the duplication had not caused a wrong number yet. It was still worth
removing, because three copies of a formula drift: somebody fixes one.

But they did NOT agree everywhere
=================================

`DOR_COD` opened with a guard the other two did not have::

    if all(i == 0 for i in ap) or all(i == 0 for i in sl):
        return 0

| case | `COD` | `DOR_COD` | `rem.COD` |
|---|---|---|---|
| all appraised values zero | `nan` | **0.0** | `nan` |
| all sale prices zero | `nan` | **0.0** | `nan` |
| empty arrays | `nan` | **0.0** | `nan` |

🧨 **A COD of 0.00 means perfect uniformity.** To an assessor reading a
summary table, `0.00` is not a null - it is the best score obtainable. That
guard turned *"I have nothing to report"* into *"I am flawless"*, and it
fired exactly where it was most likely to be believed: the report generator
deliberately zeroes out value groups with fewer than five sales, so a group
with four sales printed **COD 0.00**.

This module returns `None` instead, and gives you `format_statistic()` to
render that as something an appraiser can read. A missing statistic should
look missing.

What this module does and does not do
=====================================

The arithmetic is **delegated to `pyRealEstate`** - the maintainer's own
package, and the version the notebook has always used. Nothing is
re-derived here, so there is nothing here to drift.

What is added is the honest-absence handling, and argument names you cannot
get backwards.

⚠️ **Argument order is not symmetric.** `cod(sale_prices, appraised_values)`
is not the same as `cod(appraised_values, sale_prices)` - measured on real
data, 14.53 against 15.59. The keyword names exist so that a call site reads
correctly without anybody having to remember which way round it goes.
"""

from __future__ import annotations

import numpy as np

from pyRealEstate import RealEstateMetrics as rem

__all__ = [
    "cod",
    "prd",
    "weighted_mean_ratio",
    "median_ratio",
    "format_statistic",
    "INSUFFICIENT_DATA",
]


# What every function here returns when the statistic cannot honestly be
# computed. `None` rather than 0, rather than nan, and rather than raising:
#
#   0    is a LIE - it reads as a perfect score
#   nan  propagates into arithmetic and formats as "nan" on screen
#   an exception would take down a report over one thin value group
#
# `None` has to be handled deliberately, which is the point.
INSUFFICIENT_DATA = None


def _usable(sale_prices, appraised_values, minimum_sales: int) -> bool:
    """Can a statistic be computed from these two columns at all?

    Deliberately NOT a cleaning step. No rows are dropped, nothing is
    coerced - because that would silently change numbers this project has
    already published. It answers one question: is there anything here?
    """
    sl = np.asarray(sale_prices, dtype="float64").ravel()
    ap = np.asarray(appraised_values, dtype="float64").ravel()

    if sl.size != ap.size:
        raise ValueError(
            f"sale_prices has {sl.size} values and appraised_values has "
            f"{ap.size}. These must line up row for row."
        )

    if sl.size < max(1, int(minimum_sales)):
        return False

    # The case the old guard was reaching for: a value group that was zeroed
    # out upstream because it was too thin to report on.
    if np.all(sl == 0) or np.all(ap == 0):
        return False

    return True


def cod(sale_prices, appraised_values, minimum_sales: int = 1):
    """Coefficient of Dispersion - the uniformity measure assessors live by.

    The average percentage by which individual ratios differ from the median
    ratio. Lower is better: it means similar properties are being valued
    similarly.

    IAAO guidance, roughly: under 10 is excellent for single-family in a
    homogeneous area, 10-15 is acceptable, over 20 suggests a real
    uniformity problem worth investigating.

    Args:
        sale_prices: the sale prices, adjusted however your jurisdiction
            adjusts them. In Florida that is net of the DR-493
            cost-of-sale adjustment.
        appraised_values: the values being tested - your roll, or a model's
            predictions.
        minimum_sales: below this many records, return `INSUFFICIENT_DATA`
            rather than a number. A COD from three sales is noise wearing a
            decimal point.

    Returns:
        The COD as a percentage, or `INSUFFICIENT_DATA` if it cannot be
        computed honestly.
    """
    if not _usable(sale_prices, appraised_values, minimum_sales):
        return INSUFFICIENT_DATA
    return rem.COD(sale_prices, appraised_values)


def prd(sale_prices, appraised_values, minimum_sales: int = 1):
    """Price-Related Differential - a test of vertical equity.

    Mean ratio divided by weighted mean ratio. Above 1.0 suggests
    regressivity: cheaper properties being valued relatively higher than
    expensive ones, which is the common failure and the one that falls
    hardest on the people least able to appeal it.

    IAAO's accepted range is 0.98 to 1.03.

    Returns:
        The PRD, or `INSUFFICIENT_DATA`.
    """
    if not _usable(sale_prices, appraised_values, minimum_sales):
        return INSUFFICIENT_DATA
    return rem.PRD(sale_prices, appraised_values)


def weighted_mean_ratio(sale_prices, appraised_values, minimum_sales: int = 1):
    """Total appraised value divided by total sale price.

    The level of assessment weighted by value, so expensive properties count
    for more than they do in the median. Returns `INSUFFICIENT_DATA` when it
    cannot be computed.
    """
    if not _usable(sale_prices, appraised_values, minimum_sales):
        return INSUFFICIENT_DATA
    return rem.weighted_Mean_Sale_Ratio(sale_prices, appraised_values)


def median_ratio(sale_prices, appraised_values, minimum_sales: int = 1):
    """The median of the individual ratios - the headline level of assessment.

    IAAO's accepted range is 0.90 to 1.10.

    📌 Check this one FIRST when reviewing any ratio study. A median far from
    1.00 usually means the numerator and denominator are not on the same
    footing - one adjusted and the other not - and that mistake leaves COD
    untouched, because COD divides by its own median and is therefore
    scale-free. A healthy-looking COD is no evidence that the level is right.
    """
    if not _usable(sale_prices, appraised_values, minimum_sales):
        return INSUFFICIENT_DATA

    sl = np.asarray(sale_prices, dtype="float64").ravel()
    ap = np.asarray(appraised_values, dtype="float64").ravel()
    return float(np.median(ap / sl))


def format_statistic(value, digits: int = 2, scale: float = 1.0,
                     unavailable: str = "n/a") -> str:
    """Render a statistic for a report cell, or say plainly that there is none.

    This is the other half of returning `None`. A statistic that could not be
    computed has to LOOK uncomputed wherever it is displayed, or the caution
    is lost between here and the reader's eye.

    Args:
        value: the statistic, or `INSUFFICIENT_DATA`.
        digits: decimal places.
        scale: multiplier applied before rounding - pass 100 for the
            statistics this project reports as percentages. Applied here
            rather than at the call site so that `None * 100` can never
            raise.
        unavailable: what to show when there is no number.
    """
    if value is None:
        return unavailable

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return unavailable

    if not np.isfinite(numeric):
        return unavailable

    return str(round(numeric * scale, digits))
