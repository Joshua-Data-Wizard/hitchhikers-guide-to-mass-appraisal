"""A ratio study you can ask for by name - the chat assistant's main tool.

DON'T PANIC. 🐬  The arithmetic is in `ratio_stats.py`. This module decides
WHICH SALES a statistic is computed over, because that is where a ratio study
quietly goes wrong.

Why this cannot just be a sale price column
===========================================

The obvious design is to add a sale price to the parcel table and point COD at
it. Measured on the real Lee County roll, Single Family, untuned:

| | model COD | model PRD |
|---|---|---|
| the obvious design: all sales, `Model_Value`, trimmed | **9.98** | 1.0175 |
| what the model actually scores on sales it never saw | **11.12** | 1.0254 |

So the obvious design flatters the model by more than a full point of COD. Three
separate things go wrong:

  1. **Eight in ten qualified sales trained the model.** Scoring a model on its
     own training sales measures its memory, not its judgement.
  2. **`Model_Value` is not the number that was scored.** It is predicted as of
     the valuation date, for every parcel. The Results tab scores each sale at
     its OWN date. Same rows, different predictions, different COD.
  3. **A parcel can sell twice.** 482 did. Joined onto a parcel table, those
     sales add rows, and "how many parcels are there?" comes back wrong.

So sales get their own table, and the rule that matters is built in rather than
asked for: **`Model_Value_At_Sale_Date` is empty on every training sale.** Not
discouraged, not filtered by default - absent. Nothing a language model writes
can score the model on sales it learned from, because the number is not there.

The other rule: the same rows, or no comparison
===============================================

`compare = "model_vs_county"` scores both sides on the SAME sales: held out from
training, with neither ratio an outlier. Each side trimming on its own ratio
would score each on the subset kindest to it, which can reverse the conclusion.

These are the same figures the Results tab shows under "Model vs county roll, on
identical sales", to four decimals. If they ever stop matching, one of them is
wrong, and `scripts/check_chat_assistant.py` is what notices.

The results are written as labelled text on purpose
====================================================

Every value carries its name and its source on the same line. A small language
model reading a wide table will happily take a number from the wrong column, and
a labelled line gives it far less room to do that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deep_thought.ratio_stats import (
    IAAO_MEDIAN_RATIO_RANGE,
    IAAO_PRD_RANGE,
    RATIO_OUTLIER_HIGH,
    RATIO_OUTLIER_LOW,
    format_statistic,
    ratio_is_in_range,
    ratio_summary,
)

__all__ = [
    "SALES_TABLE",
    "build_sales_rows",
    "run_ratio_study",
    "MINIMUM_SALES_FOR_A_STATISTIC",
    "COMPARE_MODEL_VS_COUNTY",
    "COMPARE_COUNTY_ONLY",
]

SALES_TABLE = "Sales"

COMPARE_MODEL_VS_COUNTY = "model_vs_county"
COMPARE_COUNTY_ONLY = "county_only"

# Below this many sales a group gets a count and no statistics. A COD from three
# sales is noise wearing a decimal point. Five matches the app's DOR report.
MINIMUM_SALES_FOR_A_STATISTIC = 5

# A county can have hundreds of neighbourhoods. Show both ends of the list, so
# "which is worst" and "which is best" can both be answered from one result.
MOST_GROUPS_SHOWN = 30

_GROUP_COLUMNS = {
    "city": "City",
    "neighborhood": "Neighborhood_Code",
    "neighbourhood": "Neighborhood_Code",
    "neighborhood_code": "Neighborhood_Code",
    "neighbourhood_code": "Neighborhood_Code",
    "sale_year": "Sale_Year",
    "year": "Sale_Year",
}


def build_sales_rows(property_type, trained_on, held_out,
                     adjusted_sale_column="Adjusted_Sale_Price",
                     prediction_column="Model_Prediction"):
    """One property type's qualified sales, in the shape of the Sales table.

    Args:
        property_type: the label this model is known by, e.g. "Single Family".
        trained_on: the sales the model was fitted on.
        held_out: the sales kept back to test it, with predictions already made
            at each sale's own date.
        adjusted_sale_column: the sale price net of the cost-of-sale adjustment.
        prediction_column: the model's prediction on `held_out`.

    Returns:
        A frame with one row per sale. 🛑 `Model_Value_At_Sale_Date` is `NaN` on
        every training row, on purpose. See the module docstring.
    """
    columns = {
        "PARCEL_ID": "PARCEL_ID",
        "SALE_YR": "Sale_Year",
        "SALE_MO": "Sale_Month",
        "SALE_PRC": "Sale_Price",
        adjusted_sale_column: "Adjusted_Sale_Price",
        "JV": "Just_Value",
    }

    test_rows = held_out[list(columns)].rename(columns = columns)
    test_rows["Held_Out"] = 1
    test_rows["Model_Value_At_Sale_Date"] = (
        held_out[prediction_column].astype("float64").values
    )

    train_rows = trained_on[list(columns)].rename(columns = columns)
    train_rows["Held_Out"] = 0
    train_rows["Model_Value_At_Sale_Date"] = np.nan

    rows = pd.concat([test_rows, train_rows], ignore_index = True)
    rows.insert(1, "Property_Type", property_type)
    return rows


def _as_code(value):
    """A neighbourhood code as text, however it was stored.

    SQLite hands numeric codes back as `1234.0`, while somebody asking about
    one types `1234`. Other counties use text codes, which pass straight through.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number == int(number) else text


def _clean_text(series):
    return series.astype("string").fillna("").str.strip().str.upper()


def _against_range(value, limits):
    """Whether a statistic is inside a range, decided here and written down.

    📌 Measured: asked about a PRD of 1.0298, both the small local model AND the
    hosted one said it was above the 1.03 limit. Comparing 1.0298 with 1.03 is
    exactly the kind of thing to do in code and hand over as a fact.
    """
    if value is None or not np.isfinite(value):
        return ""
    low, high = limits
    if value < low:
        where = "below"
    elif value > high:
        where = "above"
    else:
        where = "inside"
    return " (%s the IAAO range of %.2f to %.2f)" % (where, low, high)


def _statistics_text(summary):
    """One value source's statistics, every number labelled."""
    return ("median ratio %s%s, weighted mean ratio %s, COD %s, PRD %s%s"
            % (format_statistic(summary["median_ratio"], digits = 4),
               _against_range(summary["median_ratio"], IAAO_MEDIAN_RATIO_RANGE),
               format_statistic(summary["weighted_mean_ratio"], digits = 4),
               format_statistic(summary["cod"], digits = 4),
               format_statistic(summary["prd"], digits = 4),
               _against_range(summary["prd"], IAAO_PRD_RANGE)))


def _model_vs_county(rows, minimum_sales):
    """Both sides, on the rows where BOTH ratios are in range. Same rows."""
    held_out = rows[rows["Held_Out"] == 1]
    keep = (ratio_is_in_range(held_out["Adjusted_Sale_Price"],
                              held_out["Model_Value_At_Sale_Date"])
            & ratio_is_in_range(held_out["Adjusted_Sale_Price"],
                                held_out["Just_Value"]))
    kept = held_out[keep]
    sale_prices = kept["Adjusted_Sale_Price"].values.astype("float64")
    return (
        len(kept),
        ratio_summary(sale_prices, kept["Model_Value_At_Sale_Date"].values.astype("float64"),
                      minimum_sales),
        ratio_summary(sale_prices, kept["Just_Value"].values.astype("float64"),
                      minimum_sales),
    )


def _county_only(rows, minimum_sales):
    """The county alone, on every qualified sale whose ratio is in range."""
    keep = ratio_is_in_range(rows["Adjusted_Sale_Price"], rows["Just_Value"])
    kept = rows[keep]
    return (
        len(kept),
        ratio_summary(kept["Adjusted_Sale_Price"].values.astype("float64"),
                      kept["Just_Value"].values.astype("float64"),
                      minimum_sales),
    )


def _who_is_ahead(model, county):
    """Which side is ahead on each statistic, as a sentence.

    🛑 Measured: left to work this out, the small model wrote that the county
    "performs slightly better across all three metrics", when the model's median
    ratio was the one closer to 1.00. Like the IAAO ranges, this is a comparison
    to make in code and hand over as a fact. The closing words are there for the
    same reason: level, uniformity and vertical equity are separate questions.
    """
    def closer_to_one(statistic):
        if model[statistic] is None or county[statistic] is None:
            return None
        gap_model, gap_county = abs(model[statistic] - 1), abs(county[statistic] - 1)
        if gap_model == gap_county:
            return "the two are level"
        return "the model" if gap_model < gap_county else "the county roll"

    def lower_cod():
        if model["cod"] is None or county["cod"] is None:
            return None
        if model["cod"] == county["cod"]:
            return "the two are level"
        return "the model" if model["cod"] < county["cod"] else "the county roll"

    level, uniformity, equity = closer_to_one("median_ratio"), lower_cod(), closer_to_one("prd")
    if None in (level, uniformity, equity):
        return ""
    return ("Who is ahead on these sales: on level (median ratio closest to 1.00), %s. "
            "On uniformity (lowest COD), %s. On vertical equity (PRD closest to 1.00), "
            "%s. These are separate questions, so there is no overall winner."
            % (level, uniformity, equity))


def _whole_group_report(property_type, rows, compare, minimum_sales):
    limits = "%s to %s" % (RATIO_OUTLIER_LOW, RATIO_OUTLIER_HIGH)

    if compare == COMPARE_COUNTY_ONLY:
        count, county = _county_only(rows, minimum_sales)
        return [
            "RATIO STUDY - %s - the county roll on its own" % property_type,
            "Sales used: %s qualified sales with a Just_Value ratio from %s, "
            "using sale prices adjusted for cost of sale. These include sales "
            "the model trained on, which does not matter here because the "
            "model plays no part in these figures."
            % (format(count, ","), limits),
            "County roll (Just_Value), %s sales: %s."
            % (format(count, ","), _statistics_text(county)),
            "To compare the county with the model, ask for model_vs_county, "
            "which uses fewer sales so that both sides are scored on the same "
            "ones. Do not compare these figures with the model's.",
        ]

    count, model, county = _model_vs_county(rows, minimum_sales)
    return [
        "RATIO STUDY - %s - the model and the county roll on the SAME sales"
        % property_type,
        "Sales used: %s held-out sales the model never saw in training, with "
        "both ratios from %s, using sale prices adjusted for cost of sale. "
        "These match the 'Model vs county roll, on identical sales' section "
        "of the Results tab. The model is never scored here on the sales it "
        "was trained on." % (format(count, ","), limits),
        "Model (benchmark estimate), %s sales: %s."
        % (format(count, ","), _statistics_text(model)),
        "County roll (Just_Value), %s sales: %s."
        % (format(count, ","), _statistics_text(county)),
        _who_is_ahead(model, county),
    ]


def _by_group_report(property_type, rows, compare, group_column, minimum_sales):
    limits = "%s to %s" % (RATIO_OUTLIER_LOW, RATIO_OUTLIER_HIGH)
    entries, too_thin = [], 0

    for group_value, group_rows in rows.groupby(group_column, dropna = False, sort = False):
        if group_column == "Neighborhood_Code":
            name = _as_code(group_value) or "(no code)"
        else:
            name = "(blank)" if pd.isna(group_value) or str(group_value).strip() == "" \
                   else str(group_value).strip()

        if compare == COMPARE_COUNTY_ONLY:
            count, county = _county_only(group_rows, minimum_sales)
            model = None
            sort_key = county["cod"]
        else:
            count, model, county = _model_vs_county(group_rows, minimum_sales)
            sort_key = model["cod"]

        if count < minimum_sales or sort_key is None or not np.isfinite(sort_key):
            too_thin += 1
            continue

        if model is None:
            line = "%s, %s sales: county %s." % (name, format(count, ","),
                                                 _statistics_text(county))
        else:
            line = ("%s, %s sales: model %s. County %s."
                    % (name, format(count, ","), _statistics_text(model),
                       _statistics_text(county)))
        entries.append((sort_key, line))

    entries.sort(key = lambda entry: entry[0], reverse = True)

    if compare == COMPARE_COUNTY_ONLY:
        heading = ("RATIO STUDY BY %s - %s - the county roll on its own"
                   % (group_column, property_type))
        which = ("Sales used in each group: qualified sales with a Just_Value "
                 "ratio from %s. Sorted highest (worst) county COD first." % limits)
    else:
        heading = ("RATIO STUDY BY %s - %s - the model and the county roll on "
                   "the SAME sales" % (group_column, property_type))
        which = ("Sales used in each group: held-out sales the model never saw "
                 "in training, with both ratios from %s. Sorted highest (worst) "
                 "model COD first." % limits)

    lines = [heading, which]
    if not entries:
        lines.append("No group has the %d sales needed for a statistic."
                     % minimum_sales)
    elif len(entries) <= MOST_GROUPS_SHOWN:
        lines.extend(line for _, line in entries)
    else:
        half = MOST_GROUPS_SHOWN // 2
        lines.extend(line for _, line in entries[:half])
        lines.append("... %s groups in the middle of the list are not shown. "
                     "Ask about one by name to see it. The lowest (best) COD "
                     "groups follow."
                     % format(len(entries) - MOST_GROUPS_SHOWN, ","))
        lines.extend(line for _, line in entries[-half:])

    if too_thin:
        lines.append("%s group(s) have fewer than %d sales and get no statistics."
                     % (format(too_thin, ","), minimum_sales))
    return lines


def run_ratio_study(sales, compare = "", group_by = "", property_type = "",
                    city = "", neighborhood_code = "",
                    minimum_sales = MINIMUM_SALES_FOR_A_STATISTIC):
    """Answer a ratio study question, as text a person or a model can quote.

    Args:
        sales: the Sales table as a frame.
        compare: `"model_vs_county"` (the default) or `"county_only"`.
        group_by: optional. `"city"`, `"neighborhood"` or `"sale_year"`.
        property_type: optional. Blank reports every modelled type separately.
            📌 Types are never pooled: a COD across single family homes and
            commercial buildings describes nothing anybody values together.
        city: optional, any case.
        neighborhood_code: optional.
        minimum_sales: fewer sales than this gets no statistics.

    Returns:
        The report, or a sentence saying what was wrong with the request. It
        does not raise on a bad argument: a language model chose these, and an
        explanation it can read lets it correct itself.
    """
    wanted_compare = (compare or "").strip().lower()
    if wanted_compare == "" or "vs" in wanted_compare or "model" in wanted_compare:
        wanted_compare = COMPARE_MODEL_VS_COUNTY
    elif any(word in wanted_compare for word in ("county", "roll", "just")):
        wanted_compare = COMPARE_COUNTY_ONLY
    else:
        return ("compare must be '%s' or '%s', not %r."
                % (COMPARE_MODEL_VS_COUNTY, COMPARE_COUNTY_ONLY, compare))

    group_column = None
    wanted_group = (group_by or "").strip().lower()
    if wanted_group and wanted_group != "property_type":
        group_column = _GROUP_COLUMNS.get(wanted_group)
        if group_column is None:
            return ("group_by must be 'city', 'neighborhood' or 'sale_year', "
                    "not %r." % group_by)

    rows = sales
    available_types = sorted(rows["Property_Type"].dropna().unique())

    if (property_type or "").strip():
        rows = rows[_clean_text(rows["Property_Type"]) == property_type.strip().upper()]
        if rows.empty:
            return ("There are no sales for property_type %r. The property types "
                    "modelled in this session are: %s."
                    % (property_type, ", ".join(available_types)))

    if (city or "").strip():
        before = rows
        rows = rows[_clean_text(rows["City"]) == city.strip().upper()]
        if rows.empty:
            known = sorted(c for c in before["City"].dropna().astype(str).unique()
                           if c.strip())
            return ("There are no sales for city %r. The cities with sales are: %s."
                    % (city, ", ".join(known[:60])))

    if str(neighborhood_code or "").strip():
        wanted_code = _as_code(neighborhood_code)
        rows = rows[rows["Neighborhood_Code"].map(_as_code) == wanted_code]
        if rows.empty:
            return ("There are no sales for neighborhood_code %r. Use "
                    "group_by='neighborhood' to see the codes that have sales."
                    % neighborhood_code)

    blocks = []  # one per property type, never pooled
    for one_type in sorted(rows["Property_Type"].dropna().unique()):
        type_rows = rows[rows["Property_Type"] == one_type]
        if group_column:
            lines = _by_group_report(one_type, type_rows, wanted_compare,
                                     group_column, minimum_sales)
        else:
            lines = _whole_group_report(one_type, type_rows, wanted_compare,
                                        minimum_sales)
        blocks.append("\n".join(lines))

    if not blocks:
        return "No sales matched that request."

    ## 🛑 PRB and PRD are one letter apart. Asked for PRB, a small model that had
    ## just been handed a PRD reported that instead. So the result says so, at the
    ## moment it is reading the figures.
    blocks.append("PRB is not in these results. This app does not compute it, and "
                  "PRD is a different statistic.")
    return "\n\n".join(blocks)
