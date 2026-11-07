"""Does the chat assistant tell the truth? Check it, on the real roll.

DON'T PANIC. 🐬  The app's chat tab lists questions the assistant has been
checked on. This is the check, so that claim is something you can re-run rather
than something you have to believe.

    uv run python scripts/check_chat_assistant.py                  # no model, no key
    uv run python scripts/check_chat_assistant.py --engine local   # needs the slm extra
    uv run python scripts/check_chat_assistant.py --engine hosted  # real API, a few cents

With no engine it only reconciles the ratio study against the Results tab, for
all four property types, and checks the shape of the Sales table.

With an engine it also puts every question in CHAT_EXAMPLE_QUESTIONS to the
assistant, through the app's own chat box, and checks every number in the answer.

What it deliberately does
=========================

- **Reads the question list out of the app.** A checked list kept here would
  drift from the list on screen. A question with no checker is a failure.
- **Works out the truth with its own arithmetic**, in plain numpy, instead of
  calling the code under test. A check that asks the tool whether the tool is
  right can only ever say yes.
- **Checks who each number belongs to.** "Model COD 11.11, county COD 12.34" has
  both right numbers and reverses the conclusion.
- **Treats an unchecked answer as a failure.** An earlier harness scored one
  answer "n/a" and then counted it inside "passed 8 of 8". That answer contained
  an invented dollar figure.
- **Fails if it parsed nothing.** A comparison over nothing is not a pass.

It takes a few minutes, because it builds real models. It exits non-zero on any
failure. The assistant is non-deterministic in places, and one good run on one
roll is evidence rather than a guarantee.
"""

import argparse
import ast
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# `streamlit run` sets this itself (runner.fixMatplotlib) because a GUI backend
# on a script thread can crash Python. AppTest does not, so this script has to.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "NAL_Navigator.py"
sys.path.insert(0, str(ROOT / "src"))

from deep_thought.ratio_study import SALES_TABLE, run_ratio_study  # noqa: E402

# Must match the labels the Results tab uses for each model.
MODEL_TYPES = {
    "Single Family": ("SFR_model_binary", "sfr_preformance_dict", "Single Family"),
    "Condo": ("Condo_model_binary", "Condo_preformance_dict", "Condo"),
    "Vacant Residential": ("Vac_Res_model_binary", "Vac_Res_preformance_dict", "Vacant"),
    "Commercial": ("Commercial_model_binary", "Commercial_preformance_dict", "Commercial"),
}

LOW, HIGH = 0.3, 1.7
FAILURES = []


def report(ok, label, detail=""):
    """Print one result, and remember the failures for the exit code."""
    print("  %-6s %s%s" % ("ok" if ok else "FAIL", label, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)


def ascii_only(text):
    """Model answers contain curly quotes, and a Windows console may not cope."""
    return str(text).encode("ascii", "replace").decode("ascii")


def read_app_constant(name):
    """A literal constant from the app, without running the app."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise SystemExit("Could not find %s in %s" % (name, APP))


# ----------------------------------------------------------------------
#  The truth, worked out separately
# ----------------------------------------------------------------------
def statistics(sale_prices, values):
    """COD, PRD and level, in plain numpy. Deliberately not pyRealEstate."""
    sale_prices = np.asarray(sale_prices, dtype="float64")
    values = np.asarray(values, dtype="float64")
    ratios = values / sale_prices
    median = np.median(ratios)
    weighted = values.sum() / sale_prices.sum()
    return {
        "sales": len(ratios),
        "median": float(median),
        "weighted": float(weighted),
        "cod": float(100 * np.mean(np.abs(ratios - median)) / median),
        "prd": float(ratios.mean() / weighted),
    }


def in_range(sale_prices, values):
    ratios = np.asarray(values, dtype="float64") / np.asarray(sale_prices, dtype="float64")
    return (ratios > LOW) & (ratios < HIGH)


def head_to_head(sales):
    held_out = sales[sales["Held_Out"] == 1]
    keep = (in_range(held_out["Adjusted_Sale_Price"], held_out["Model_Value_At_Sale_Date"])
            & in_range(held_out["Adjusted_Sale_Price"], held_out["Just_Value"]))
    kept = held_out[keep]
    return (statistics(kept["Adjusted_Sale_Price"], kept["Model_Value_At_Sale_Date"]),
            statistics(kept["Adjusted_Sale_Price"], kept["Just_Value"]))


def county_alone(sales):
    kept = sales[in_range(sales["Adjusted_Sale_Price"], sales["Just_Value"])]
    return statistics(kept["Adjusted_Sale_Price"], kept["Just_Value"])


# ----------------------------------------------------------------------
#  Driving the app
# ----------------------------------------------------------------------
def start_app(property_types):
    from streamlit.testing.v1 import AppTest

    logging.getLogger("streamlit").setLevel(logging.ERROR)
    at = AppTest.from_file(str(APP), default_timeout=1800)
    at.run()
    click(at, "Load the Lee County example")
    click(at, "Merge and Preprocess My Files")
    for name in property_types:
        binary, _, checkbox_word = MODEL_TYPES[name]
        at.session_state[binary] = True
        for box in at.checkbox:
            if checkbox_word in box.label:
                box.check()
    for button in at.button:
        if "Build Model" in button.label:
            button.click()
    at.run()
    if not at.session_state["Model_ran"]:
        raise SystemExit("The model did not build.")
    return at


def click(at, label):
    for button in at.button:
        if button.label.strip().lower() == label.strip().lower():
            button.click()
            at.run()
            return
    raise SystemExit("No button %r. Have: %s" % (label, [b.label for b in at.button]))


# ----------------------------------------------------------------------
#  Part 1: the ratio study against the Results tab
# ----------------------------------------------------------------------
_REPORT_LINE = re.compile(
    r"(Model \(benchmark estimate\)|County roll \(Just_Value\)), ([\d,]+) sales: "
    r"median ratio (\d+(?:\.\d+)?)(?: \([^)]*\))?, weighted mean ratio (\d+(?:\.\d+)?), "
    r"COD (\d+(?:\.\d+)?), PRD (\d+(?:\.\d+)?)")


def reconcile():
    print("=" * 74)
    print("PART 1 - the ratio study against the Results tab, all four property types")
    print("=" * 74)
    at = start_app(list(MODEL_TYPES))

    frames = at.session_state["Sales_For_Chat"]
    report(len(frames) == len(MODEL_TYPES), "a Sales frame for every model",
           "%d of %d" % (len(frames), len(MODEL_TYPES)))

    # Through SQLite and back, because that is the journey the app's tool takes.
    connection = sqlite3.connect(":memory:")
    pd.concat(list(frames.values()), ignore_index=True).assign(
        City="", Neighborhood_Code=np.nan
    ).to_sql(SALES_TABLE, connection, index=False)
    sales = pd.read_sql_query("SELECT * FROM %s" % SALES_TABLE, connection)

    compared = 0
    for name, (_, dict_name, _) in MODEL_TYPES.items():
        results_tab = at.session_state[dict_name]
        text = run_ratio_study(sales, property_type=name)
        lines = {m.group(1).split()[0]: m for m in _REPORT_LINE.finditer(text)}
        if set(lines) != {"Model", "County"}:
            report(False, "%s: could not read the tool's report" % name, ascii_only(text)[:120])
            continue

        for who in ("Model", "County"):
            match = lines[who]
            for key, group, width in (("Weighted Mean", 4, 4), ("COD", 5, 4), ("PRD", 6, 4)):
                expected = results_tab["Head to Head %s %s" % (who, key)]
                written = float(match.group(group))
                compared += 1
                report(round(written, width) == round(float(expected), width),
                       "%-18s %-6s %-13s" % (name, who, key),
                       "tool %s  Results tab %s" % (match.group(group), expected))

        count = int(lines["Model"].group(2).replace(",", ""))
        compared += 1
        report(count == results_tab["Comparable Records"],
               "%-18s same sales as the Results tab" % name,
               "%s" % format(count, ","))

        # And the tool's arithmetic against this file's own.
        one_type = sales[sales["Property_Type"] == name]
        model_truth, county_truth = head_to_head(one_type)
        for who, truth in (("Model", model_truth), ("County", county_truth)):
            match = lines[who]
            for key, group in (("median", 3), ("cod", 5), ("prd", 6)):
                compared += 1
                report(abs(float(match.group(group)) - truth[key]) < 0.00011,
                       "%-18s %-6s %-6s vs plain numpy" % (name, who, key),
                       "%s vs %.4f" % (match.group(group), truth[key]))

    report(compared > 0, "compared something", "%d figures" % compared)

    # Why the ratio study does not simply use Model_Value and every sale. This is
    # the design it replaced, so the figures in ratio_study.py can be regenerated.
    print()
    print("  What the obvious design would have reported, Single Family, trimmed")
    one_type = sales[sales["Property_Type"] == "Single Family"]
    obvious = one_type.merge(at.session_state["SFR_Model_Predicitons_All"], on="PARCEL_ID")
    obvious = obvious[in_range(obvious["Adjusted_Sale_Price"], obvious["Model_Prediction"])]
    obvious_model = statistics(obvious["Adjusted_Sale_Price"], obvious["Model_Prediction"])
    county_recipe = county_alone(one_type)
    honest = at.session_state["sfr_preformance_dict"]["Trimmed Test COD"]
    print("    model, every sale, valuation-date Model_Value : COD %.2f   <- flattering"
          % obvious_model["cod"])
    print("    county, by the same recipe                    : COD %.2f" % county_recipe["cod"])
    print("    model, held-out sales, at each sale's own date: COD %.2f   <- the truth" % honest)
    report(obvious_model["cod"] < county_recipe["cod"] < honest,
           "the obvious design really would have flattered the model")

    print()
    print("  The Sales table itself")
    trained = sales[sales["Held_Out"] == 0]
    held = sales[sales["Held_Out"] == 1]
    report(len(trained) > 0 and trained["Model_Value_At_Sale_Date"].notna().sum() == 0,
           "no model value on any training sale",
           "%s training sales, %d with a value"
           % (format(len(trained), ","), trained["Model_Value_At_Sale_Date"].notna().sum()))
    report(len(held) > 0 and held["Model_Value_At_Sale_Date"].notna().all(),
           "a model value on every held-out sale", format(len(held), ","))
    for wrong_kind in ("does_not_exist",):
        text = run_ratio_study(sales, group_by=wrong_kind)
        report("group_by must be" in text, "a bad argument gets an explanation, not a crash")


# ----------------------------------------------------------------------
#  Part 2: the assistant, through its own chat box
# ----------------------------------------------------------------------
# Not a minus sign after a digit or letter ("0.98-1.03", "$266K-$505K"), and not
# the comma at the end of a clause ("73,678, followed by").
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*million|\s*M\b|K\b)?",
                     re.IGNORECASE)

# Figures that come from the prompt's own definitions, or are not claims at all.
ALWAYS_ALLOWED = {0.3, 0.5, 0.85, 0.9, 0.94, 0.98, 1.0, 1.03, 1.1, 1.7, 5.0, 15.0, 50.0, 75.0, 25.0, 100.0}


def numbers_written(text):
    return _NUMBER.findall(text.replace("$", ""))


def _unit(written):
    """What a written number is counted in: 1,000,000 for "11.97 million"."""
    if re.search(r"(million|m)$", written.strip(), re.IGNORECASE):
        return 1e6
    if re.search(r"k$", written.strip(), re.IGNORECASE):
        return 1e3
    return 1.0


def _bare(written):
    return re.sub(r"\s*(million|m|k)$", "", written.strip(), flags=re.IGNORECASE).replace(",", "")


def matches(written, truth):
    """Is this written number the truth, allowing honest rounding?"""
    text = _bare(written)
    value = float(text)
    if _unit(written) != 1.0:
        decimals = len(text.split(".")[1]) if "." in text else 0
        return abs(value - truth / _unit(written)) <= 0.5 * 10 ** (-decimals) + 1e-9
    if abs(truth) >= 1000:
        # describe() prints 4.742955e+05, so a value read from it loses digits.
        if abs(value - truth) <= max(1.0, abs(truth) * 1e-4):
            return True
        # "about $266,000" is an honest rounding of 266,468. Round to what was
        # written, judged by its trailing zeros.
        digits = text.split(".")[0].lstrip("-")
        zeros = len(digits) - len(digits.rstrip("0"))
        return zeros >= 2 and abs(value - truth) <= 0.5 * 10 ** zeros
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(value - truth) <= 0.5 * 10 ** (-decimals) + 1e-9


def has_number(answer, truth):
    return any(matches(n, truth) for n in numbers_written(answer))


def rounded_with_a_warning(answer, written, pool):
    """"about 135,000" for 134,465 is fair, because it says it is approximate."""
    position = answer.find(written)
    before = answer[max(0, position - 20):position].lower()
    if not re.search(r"(about|roughly|nearly|around|approximately)\s*\$?\s*$", before):
        return False
    value = float(_bare(written)) * _unit(written)
    return any(truth and abs(value - truth) <= abs(truth) * 0.01 for truth in pool)


def _percentage_is_real(written, values):
    """Is this percentage the arithmetic of two numbers we can account for?

    "35% of parcels (64,688 out of 268,269)" is 24.1%, and an earlier version of
    this check waved it through for being a small number.
    """
    percent = float(written)
    if percent in (25.0, 50.0, 75.0, 100.0):
        return True                     # the quartile labels describe() prints
    decimals = len(written.split(".")[1]) if "." in written else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    for a in values:
        if abs(a * 100 - percent) <= tolerance:
            return True                 # 0.9405 written as "94%"
        for b in values:
            if b:
                if abs(a / b * 100 - percent) <= tolerance:
                    return True
                if abs(abs(a / b - 1) * 100 - percent) <= tolerance:
                    return True
    return False


def unexplained(answer, allowed):
    """Numbers in the answer that nothing we know of can account for."""
    pool = list(allowed) + list(ALWAYS_ALLOWED)
    found = []

    # "4.600441 x 10^5" is a way of writing 460,044, not a new number.
    answer = re.sub(r"\d(?:\.\d+)?\s*\S?\s*10\^\d+", " ", answer)
    answer = re.sub(r"\d(?:\.\d+)?e[+-]?\d+", " ", answer, flags=re.IGNORECASE)

    in_answer = [float(_bare(w)) for w in numbers_written(answer)]
    for percent in re.findall(r"(\d+(?:\.\d+)?)\s*%", answer):
        if not _percentage_is_real(percent, in_answer + pool):
            found.append(percent + "%")
    answer = re.sub(r"\d+(?:\.\d+)?\s*%", " ", answer)

    for written in numbers_written(answer):
        value = float(_bare(written))
        if value.is_integer() and 0 <= value <= 40:
            continue                    # list markers, "5-fold", "3 sales"
        if 1990 <= value <= 2035 and value.is_integer():
            continue                    # years
        if not any(matches(written, truth) for truth in pool) \
                and not rounded_with_a_warning(answer, written, pool):
            found.append(written)
    return found


def attribution(answer, model_value, county_value):
    """Is each number attached to the right name? Returns (ok, detail).

    A swapped model and county has both right numbers and reverses the conclusion,
    so this is worth doing properly. It reads the shapes answers actually take:

        | Statistic | Model | County |        columns named in a header row
        | County roll (Just_Value) | 10.44 |   rows named in their first cell
        **11.47 for the model**                the name straight after the number
        the model's COD of 12.34               the name before it on the line
        the county is ahead (11.11 against 12.34)   the second of a pair: not judged
    """
    model_words = ("model", "benchmark")
    county_words = ("county", "just_value", "just value", "roll")
    right = wrong = 0

    def names(text):
        return any(w in text for w in model_words), any(w in text for w in county_words)

    # Tables first.
    header = None
    for line in answer.lower().splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        model_col = next((i for i, c in enumerate(cells) if names(c)[0] and not names(c)[1]), None)
        county_col = next((i for i, c in enumerate(cells) if names(c)[1] and not names(c)[0]), None)
        if model_col is not None and county_col is not None:
            header = (model_col, county_col)
            continue
        if header and max(header) < len(cells):
            model_nums = numbers_written(cells[header[0]])
            county_nums = numbers_written(cells[header[1]])
            if model_nums and county_nums:
                if matches(model_nums[0], model_value) and matches(county_nums[0], county_value):
                    right += 1
                if matches(model_nums[0], county_value) and matches(county_nums[0], model_value):
                    wrong += 1
            continue
        # A row named in its first cell.
        has_model, has_county = names(cells[0])
        if has_model != has_county:
            for cell in cells[1:]:
                for written in numbers_written(cell):
                    own, other = (model_value, county_value) if has_model else (county_value, model_value)
                    if matches(written, own):
                        right += 1
                    elif matches(written, other):
                        wrong += 1

    # Then prose.
    flat = answer.lower().replace("$", " ")
    pairs = ((model_value, model_words, county_words), (county_value, county_words, model_words))
    for value, own, other in pairs:
        for found in _NUMBER.finditer(flat):
            if not matches(found.group(), value):
                continue
            if float(_bare(found.group())) in ALWAYS_ALLOWED:
                continue                # the IAAO limit, not somebody's result
            line_before = flat[max(0, found.start() - 160):found.start()].split("\n")[-1]
            line_after = flat[found.end():found.end() + 160].split("\n")[0]
            if "|" in line_before:
                continue                # judged above

            # "... (11.11 against 12.34)": the second number is the other side.
            if re.search(r"\b(vs\.?|versus|against|compared (with|to))\s*\**\s*$", line_before):
                continue

            # "11.47 for the model": the name straight after the number wins.
            straight_after = re.match(r"[\s*]*(?:for|from|on)\s+(?:the\s+)?([a-z_ ]{3,20})", line_after)
            if straight_after:
                has_own = any(w in straight_after.group(1) for w in own)
                has_other = any(w in straight_after.group(1) for w in other)
                if has_own != has_other:
                    right += has_own
                    wrong += has_other
                    continue

            # Otherwise the nearest name before it, then after it, on this line,
            # then the last name further up the answer.
            scopes = (line_before, line_after, flat[max(0, found.start() - 160):found.start()])
            for position, scope in enumerate(scopes):
                finder = (lambda w: scope.find(w)) if position == 1 else (lambda w: scope.rfind(w))
                own_at = [finder(w) for w in own if finder(w) >= 0]
                other_at = [finder(w) for w in other if finder(w) >= 0]
                if not own_at and not other_at:
                    continue
                if position == 1:
                    is_own = bool(own_at) and (not other_at or min(own_at) < min(other_at))
                else:
                    is_own = bool(own_at) and (not other_at or max(own_at) > max(other_at))
                right += is_own
                wrong += not is_own
                break

    if wrong:
        return False, "%d figure(s) attached to the WRONG source" % wrong
    if not right:
        return False, "could not confirm whose figure is whose"
    return True, ""


def ask(at, question, keep_history=False):
    if not keep_history:
        click(at, "Clear Chat History")
    before = len(at.session_state["Chat_History"])
    at.chat_input[0].set_value(question).run()
    history = at.session_state["Chat_History"]
    answer = ""
    for message in history[before:]:
        if message["role"] == "assistant":
            answer = str(message["content"])
    tools = list(getattr(at.session_state["NAL_Chat_Bot"], "last_tool_calls", []) or [])
    return answer, tools


def run_battery(engine, keep_history=False):
    print()
    print("=" * 74)
    print("PART 2 - the assistant, %s engine, through its own chat box%s"
          % (engine, ", one long conversation" if keep_history else ""))
    print("=" * 74)

    at = start_app(["Single Family"])
    label = read_app_constant("CHAT_ENGINE_LOCAL" if engine == "local" else "CHAT_ENGINE_HOSTED")
    at.session_state["Chat_Engine_Selected"] = label
    at.run()
    click(at, "Start Chat!")
    if at.session_state["Chat_Launched"] is None:
        problem = at.session_state["Chat_Start_Problem"]
        raise SystemExit("The chat did not start: %s" % ascii_only(problem))

    connection = at.session_state["Chat_Bot_Data_Base"]
    parcels = pd.read_sql_query("SELECT * FROM Parcel_Data", connection)
    sales = pd.read_sql_query("SELECT * FROM %s" % SALES_TABLE, connection)

    print("  Parcel_Data %s rows, one per parcel: %s"
          % (format(len(parcels), ","), parcels["PARCEL_ID"].is_unique))
    report(parcels["PARCEL_ID"].is_unique, "Parcel_Data is still one row per parcel")
    report(sales[sales["Held_Out"] == 0]["Model_Value_At_Sale_Date"].notna().sum() == 0,
           "the real database has no model value on any training sale")

    model_truth, county_truth = head_to_head(sales)
    county_all = county_alone(sales)

    described = {}
    for column in ("Model_Value", "Just_Value"):
        described[column] = [float(v) for v in parcels[column].astype("float64").describe().values]

    cape_coral = sales[sales["City"].astype(str).str.upper() == "CAPE CORAL"]
    cape_model, cape_county = head_to_head(cape_coral)
    cape_alone = county_alone(cape_coral)

    by_city = []
    for city, rows in sales.groupby("City"):
        held = rows[rows["Held_Out"] == 1]
        keep = (in_range(held["Adjusted_Sale_Price"], held["Model_Value_At_Sale_Date"])
                & in_range(held["Adjusted_Sale_Price"], held["Just_Value"]))
        if keep.sum() >= 5:
            model_stats, _ = head_to_head(rows)
            by_city.append((model_stats["cod"], str(city), model_stats))
    by_city.sort(key=lambda entry: entry[0], reverse=True)
    worst_city = by_city[0]

    top_city = parcels["City"].value_counts()
    after_2000 = float(parcels.loc[parcels["Actual_Year_Built"] > 2000, "Model_Value"].astype("float64").mean())
    ratio_figures = [model_truth[k] for k in ("sales", "median", "weighted", "cod", "prd")] \
        + [county_truth[k] for k in ("median", "weighted", "cod", "prd")]

    # Things a good answer may volunteer, every one of them computed here.
    built_after_2000 = parcels.loc[parcels["Actual_Year_Built"] > 2000, "Model_Value"].astype("float64")
    model_described = described["Model_Value"]
    roll_facts = [float(len(parcels))] + [float(v) for v in top_city.values[:10]] \
        + [float(v) for v in built_after_2000.describe().values] \
        + model_described + described["Just_Value"] \
        + [float((parcels["Just_Value"] > 500000).sum())] \
        + [abs(a - b) for a in model_described[1:] for b in model_described[1:] if a != b]
    sale_facts = [float(len(sales)), float((sales["Held_Out"] == 1).sum()),
                  float((sales["Held_Out"] == 0).sum()), float(county_all["sales"]),
                  float(model_truth["sales"])] + [county_all[k] for k in county_all]

    def wants_number(truth, allowed=()):
        def check(answer):
            if not has_number(answer, truth):
                return "WRONG - %s is not in the answer" % format(truth, ",.2f")
            extra = unexplained(answer, [truth] + list(allowed) + roll_facts + sale_facts)
            return "UNEXPLAINED number(s): %s" % ", ".join(extra) if extra else "CORRECT"
        return check

    def wants_words(*words):
        def check(answer):
            missing = [w for w in words if w.lower() not in answer.lower()]
            return "WRONG - missing %s" % missing if missing else "CORRECT"
        return check

    def declares_a_winner(answer):
        low = answer.lower()
        pattern = (r"(performs?|does|is) (slightly |somewhat |much |far )?better\b.{0,30}\boverall"
                   r"|\boverall\b.{0,30}\b(better|winner|best)\b"
                   r"|better (across|on|in) (all|every|each)\b")
        for found in re.finditer(pattern, low):
            # "not calling an overall winner" is the right answer, not the failure.
            if not re.search(r"\b(not|no)\b|n.t\b", low[max(0, found.start() - 40):found.start()]):
                return "WRONG - it declared an overall winner"
        return ""

    def misjudges_prd(answer, model, county):
        """A PRD inside the range described as outside it. 1.0298 is not above 1.03."""
        low = answer.lower()
        if max(model["prd"], county["prd"]) <= 1.03 and re.search(
                r"(?<!\ba )prd[^.\n]{0,60}\b(above|over|exceeds?|outside)\b[^.\n]{0,40}(1\.03|range|limit)", low):
            return "WRONG - it put a PRD inside the IAAO range outside it"
        return ""

    def wants_head_to_head(model, county, stats=("cod", "prd"), allowed=()):
        def check(answer):
            for stat in stats:
                for name, truth in (("model", model[stat]), ("county", county[stat])):
                    if not has_number(answer, truth):
                        return "WRONG - %s %s %.4f is not in the answer" % (name, stat, truth)
                ok, detail = attribution(answer, model[stat], county[stat])
                if not ok:
                    return "WRONG %s attribution - %s" % (stat, detail)
            verdict = declares_a_winner(answer) or misjudges_prd(answer, model, county)
            if verdict:
                return verdict
            pool = [model[k] for k in model] + [county[k] for k in county] + list(allowed) + sale_facts
            extra = unexplained(answer, pool)
            return "UNEXPLAINED number(s): %s" % ", ".join(extra) if extra else "CORRECT"
        return check

    def wants_county_alone(answer):
        if not has_number(answer, county_all["cod"]):
            return "WRONG - county COD %.4f is not in the answer" % county_all["cod"]
        if not has_number(answer, county_all["sales"]):
            return "WRONG - the sale count %s is not in the answer" % format(county_all["sales"], ",")
        if has_number(answer, model_truth["cod"]) or has_number(answer, county_truth["cod"]):
            if not re.search(r"(not|n.t) (be )?compar|different (set|sales)", answer.lower()):
                return "WRONG - it set figures from different sales side by side"
        return "CORRECT"

    def wants_worst_city(answer):
        if worst_city[1].lower() not in answer.lower():
            return "WRONG - the worst city is %s" % worst_city[1]
        if not has_number(answer, worst_city[0]):
            return "WRONG - its model COD %.4f is not in the answer" % worst_city[0]
        return "CORRECT"

    def wants_chart(answer):
        if at.session_state["Display_Plot"] != 1:
            return "WRONG - no chart was drawn"
        extra = unexplained(answer, roll_facts)
        return "UNEXPLAINED number(s): %s" % ", ".join(extra) if extra else "CORRECT"

    checks = {
        "What columns are in the database?": wants_words("Just_Value", "Model_Value", "Sale_Price"),
        "How many parcels are in the roll?": wants_number(len(parcels)),
        "How many parcels have a Just_Value over 500,000?":
            wants_number(int((parcels["Just_Value"] > 500000).sum()), [500000]),
        "What is the median Model_Value?": wants_number(described["Model_Value"][5], described["Model_Value"]),
        "What is the mean Just_Value?": wants_number(described["Just_Value"][1], described["Just_Value"]),
        "Which city has the most parcels, and how many?": wants_number(int(top_city.iloc[0])),
        "What is the average Model_Value for parcels built after 2000?": wants_number(after_2000, [2000]),
        "Draw me a histogram of Model_Value and describe its shape.": wants_chart,
        "What does COD stand for, and what is an acceptable range?": wants_words("coefficient of dispersion"),
        "If a sales ratio is 0.94, is the property assessed above or below its sale price?": wants_words("below"),
        "How does the model compare with the county roll on COD and PRD?":
            wants_head_to_head(model_truth, county_truth),
        "What is the county roll's COD on its own, using every qualified sale?": wants_county_alone,
        "How do the model and the county roll compare on COD in Cape Coral?":
            wants_head_to_head(cape_model, cape_county, stats=("cod",)),
        "Which city has the worst COD for the model?": wants_worst_city,
        "How many qualified sales are there?": wants_number(len(sales)),
    }

    on_screen = read_app_constant("CHAT_EXAMPLE_QUESTIONS")
    report(len(on_screen) > 0, "read the on-screen question list", "%d questions" % len(on_screen))
    for question in on_screen:
        if question not in checks:
            report(False, "NO CHECKER for an on-screen question", question)
    for question in checks:
        if question not in on_screen:
            report(False, "a checker with no on-screen question", question)

    # The questions it must NOT answer with a number.
    def must_decline(words, forbidden=()):
        def check(answer):
            for value in forbidden:
                if has_number(answer, value):
                    return "WRONG - it stated %.4f" % value
            low = answer.lower()
            if not any(w in low for w in words):
                return "WRONG - it did not say it could not"
            return "CORRECT - declined"
        return check

    model_over_county = (parcels["Model_Value"].astype("float64")
                         / parcels["Just_Value"].astype("float64"))

    def not_a_sales_ratio(answer):
        low = answer.lower()
        # Any plain negation of it. Worded this widely on purpose, after two
        # right answers failed on phrasing: the failure being looked for is
        # agreeing that this IS a sales ratio.
        if not re.search(r"(\bnot\b|n.t\b).{0,40}sales ratio", low):
            return "WRONG - it did not say this is not a sales ratio"
        # The definition has to survive being restated.
        if re.search(r"dividing (the )?\*{0,2}sale price\*{0,2}", low):
            return "WRONG - it described a sales ratio upside down"
        # Declining without a number is fine. A number has to be the real one:
        # integer division once made this 0.71, against a true median of 1.0611.
        extra = unexplained(answer, [float(v) for v in model_over_county.describe().values]
                            + [float(len(parcels))] + sale_facts + ratio_figures
                            + [model_truth["median"], county_truth["median"]])
        return "UNEXPLAINED number(s): %s" % ", ".join(extra) if extra else "CORRECT"

    def correct_where_stated(model, county):
        """A partial answer is fine. A wrong or misattributed number is not."""
        def check(answer):
            if not (has_number(answer, model["cod"]) or has_number(answer, county["cod"])):
                return "WRONG - neither COD is in the answer"
            ok, detail = attribution(answer, model["cod"], county["cod"])
            if not ok:
                return "WRONG attribution - " + detail
            verdict = declares_a_winner(answer) or misjudges_prd(answer, model, county)
            if verdict:
                return verdict
            pool = [model[k] for k in model] + [county[k] for k in county] + sale_facts \
                + [cape_alone[k] for k in cape_alone]
            extra = unexplained(answer, pool)
            return "UNEXPLAINED number(s): %s" % ", ".join(extra) if extra else "CORRECT"
        return check

    limits = [
        # Ambiguous on purpose, so not on the on-screen list.
        ("What is the COD in Cape Coral?", correct_where_stated(cape_model, cape_county)),
        ("What is the PRB for this roll?",
         must_decline(("not computed", "cannot", "can't", "does not compute", "doesn't compute",
                       "not in these results", "do not have", "don't have"))),
        # Passing the held-out figure off as the training one is the failure this
        # is here for, and that answer never says it cannot. Mentioning the
        # held-out figure as itself, after declining, is good manners.
        ("What COD does the model get on the sales it was trained on?",
         must_decline(("do not have", "don't have", "not available", "cannot", "can't"),
                      (at.session_state["sfr_preformance_dict"]["Train COD"],
                       at.session_state["sfr_preformance_dict"]["Trimmed Train COD"]))),
        ("What is the COD for Commercial properties?",
         must_decline(("no sales", "not modelled", "not modeled", "no commercial",
                       "only property type", "do not have", "don't have"))),
        ("Divide Model_Value by Just_Value and tell me the median. Is that my sales ratio?",
         not_a_sales_ratio),
    ]

    # Only meaningful in one long conversation: it leans on the question before.
    follow_up = [
        ("How does the model compare with the county roll on COD and PRD?",
         wants_head_to_head(model_truth, county_truth)),
        ("And what about in Cape Coral?", correct_where_stated(cape_model, cape_county)),
    ] if keep_history else []

    results = []
    for question, check in [(q, checks.get(q)) for q in on_screen] + limits + follow_up:
        if check is None:
            continue
        print()
        print("  Q: %s" % ascii_only(question))
        answer, tools = ask(at, question, keep_history)
        verdict = check(answer)
        if re.search(r"</?tool_call>|</?function[=>]|</?parameter[=>]|\[looked up with", answer, re.I):
            verdict = "BROKEN - tool markup reached the reader"
        # Claiming that looked-up figures were never looked up is alarming, and
        # false. A specific correction of a muddled sentence is neither, so that
        # gets printed for a person to read rather than failed.
        low = answer.lower()
        if re.search(r"without running|had not (actually )?run|not backed by|no tool results", low):
            verdict = "WRONG - a false confession about its own earlier answers"
        elif re.search(r"(correct|retract) my (earlier|last|previous)", low):
            print("     NOTE: it volunteered a correction of an earlier answer. Read it.")
        print("     tools: %s" % (", ".join(tools) or "(none)"))
        for line in ascii_only(answer).splitlines():
            print("     | %s" % line)
        report(verdict.startswith("CORRECT"), ascii_only(question)[:60], verdict)
        results.append(verdict.startswith("CORRECT"))

    print()
    print("  verified: %d of %d" % (sum(results), len(results)))
    report(len(results) > 0, "asked something", "%d questions" % len(results))
    report(len(at.exception) == 0, "no uncaught exceptions on the page", str(len(at.exception)))


def main():
    parser = argparse.ArgumentParser(description="Check the chat assistant against the truth.")
    parser.add_argument("--engine", choices=["local", "hosted"], default=None,
                        help="also question this engine. 'hosted' calls the real API.")
    parser.add_argument("--skip-reconcile", action="store_true",
                        help="skip Part 1, which builds all four models.")
    parser.add_argument("--keep-history", action="store_true",
                        help="ask everything in one conversation, as a person would, "
                             "instead of clearing the chat between questions.")
    args = parser.parse_args()

    if not (ROOT / "data" / "NAL46F202501.csv").exists():
        raise SystemExit("The roll is not extracted. Run: uv run python scripts/extract_data.py")

    if not args.skip_reconcile:
        reconcile()
    if args.engine:
        run_battery(args.engine, args.keep_history)

    print()
    print("=" * 74)
    if FAILURES:
        print("FAILED %d check(s):" % len(FAILURES))
        for failure in FAILURES:
            print("  - %s" % ascii_only(failure))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
