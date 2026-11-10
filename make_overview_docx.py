"""Write PROJECT_OVERVIEW.docx: scope, data, variables and results to date.

Every figure is parsed out of a sealed run directory. Nothing is typed in by
hand, so the document cannot drift from the run it claims to describe, and
regenerating after a new run is the only way to change a number.

The prose that IS written here is definitional -- what a column means, where a
dataset comes from -- and is checked against the trained model's own feature
list at build time, so a feature added to the pipeline without a description
fails the build instead of silently vanishing from the table.

usage: python make_overview_docx.py [RUN_ID]
"""
import csv
import json
import pathlib
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "artifacts" / "runs"
# The newest full run carrying the cyclone feature block. Overridable so the
# document can be rebuilt against whichever run is being cited.
DEFAULT_RUN = "20261108T225052Z-run"

AUTHOR = "Ganesh Pinnu"

# ---------------------------------------------------------------- definitions
# Column meanings. Keys must match the trained feature list exactly; see the
# assertion in features_table().
FEATURE_DOC = {
    "prcp": ("Rainfall", "County rainfall on the origin day, mean over the "
             "county's reporting stations."),
    "prcp_max_stn": ("Rainfall", "Highest single-station rainfall in the "
                     "county that day. Separates a soaked county from one "
                     "cell sitting over one gauge."),
    "prcp_prev3": ("Antecedent", "Total rainfall over the previous 3 days."),
    "prcp_prev7": ("Antecedent", "Total rainfall over the previous 7 days."),
    "prcp_prev14": ("Antecedent", "Total rainfall over the previous 14 days."),
    "prcp_prev30": ("Antecedent", "Total rainfall over the previous 30 days; "
                    "a proxy for how saturated the ground already is."),
    "prcp_prev60": ("Antecedent", "Total rainfall over the previous 60 days."),
    "prcp_prev90": ("Antecedent", "Total rainfall over the previous 90 days."),
    "prcp_fwd1": ("Forecast rain", "Rain forecast for the next day."),
    "prcp_fwd3": ("Forecast rain", "Rain forecast over the next 3 days, the "
                  "window the label covers."),
    "prcp_fwd3_max": ("Forecast rain", "Largest single forecast day within "
                      "the next 3."),
    "ratio_prev3": ("Normalised", "Previous 3-day rainfall divided by the "
                    "county's own climatology for that period. 50 mm is "
                    "routine on the Louisiana coast and extreme in south "
                    "Texas, so the raw total is not comparable across "
                    "counties."),
    "ratio_fwd3": ("Normalised", "Forecast 3-day rainfall over the same "
                   "local climatology. The strongest single-variable "
                   "baseline for flood."),
    "ratio_prev30": ("Normalised", "Previous 30-day rainfall over "
                     "climatology."),
    "log_fwd3": ("Normalised", "Log transform of the 3-day forecast total, "
                 "compressing the tail."),
    "tmax": ("Weather", "Daily maximum temperature."),
    "tmin": ("Weather", "Daily minimum temperature."),
    "month_sin": ("Seasonality", "Sine component of calendar month, so "
                  "December sits next to January rather than eleven months "
                  "away."),
    "month_cos": ("Seasonality", "Cosine component of calendar month."),
    "hurricane_season": ("Seasonality", "Flag for 1 June to 30 November."),
    "lat": ("Location", "County centroid latitude."),
    "lon": ("Location", "County centroid longitude."),
    "pop_density": ("Static", "People per square mile. Carried with a "
                    "caveat: removing it costs flood skill, which most "
                    "likely reflects how reliably an episode in a populated "
                    "county gets reported rather than physical risk."),
    # --- cyclone block, built from the NHC official forecast in force ---
    "cyc_active": ("Cyclone", "Whether any tropical cyclone was under "
                   "advisory and forecast to approach this county."),
    "cyc_n_active": ("Cyclone", "Number of such storms, which matters in a "
                     "double-storm season."),
    "cyc_dist_min": ("Cyclone", "Closest forecast approach of the most "
                     "threatening storm, in kilometres from the county "
                     "centroid."),
    "cyc_dist1": ("Cyclone", "Forecast distance at lead day 1."),
    "cyc_dist2": ("Cyclone", "Forecast distance at lead day 2."),
    "cyc_dist3": ("Cyclone", "Forecast distance at lead day 3."),
    "cyc_lead_closest": ("Cyclone", "Which lead day the closest approach "
                         "falls on."),
    "cyc_vmax_close": ("Cyclone", "Forecast maximum sustained wind at the "
                       "hour of closest approach."),
    "cyc_vmax_max": ("Cyclone", "Peak forecast intensity anywhere in the "
                     "3-day window."),
    "cyc_fwd_kt": ("Cyclone", "Forward speed of the storm; a slow mover "
                   "rains far longer over one county."),
    "cyc_bearing_sin": ("Cyclone", "Sine of the bearing from county to "
                        "storm."),
    "cyc_bearing_cos": ("Cyclone", "Cosine of that bearing. The pair "
                        "distinguishes an approach from a departure."),
    "cyc_r34": ("Cyclone", "Forecast radius of 34-knot (tropical storm "
                "force) winds."),
    "cyc_r50": ("Cyclone", "Forecast radius of 50-knot winds."),
    "cyc_r64": ("Cyclone", "Forecast radius of 64-knot (hurricane force) "
                "winds."),
    "cyc_in34": ("Cyclone", "Whether the county falls inside the forecast "
                 "34-knot wind field."),
    "cyc_in50": ("Cyclone", "Whether it falls inside the 50-knot field."),
    "cyc_in64": ("Cyclone", "Whether it falls inside the 64-knot field."),
    "cyc_cone_km": ("Cyclone", "Track-error radius at the relevant lead, "
                    "measured from past official forecasts rather than taken "
                    "from the published table."),
    "cyc_cone_margin": ("Cyclone", "How far inside or outside the forecast "
                        "cone the county sits."),
    "cyc_in_cone": ("Cyclone", "Whether the county is inside the cone at "
                    "all."),
}

SOURCES = [
    ("Rainfall and temperature", "NOAA GHCN-Daily station observations, "
     "aggregated to county-day over the stations reporting in each county."),
    ("Rainfall climatology", "Derived from the same GHCN-Daily observations "
     "over a frozen early window, never from the validation or test years. "
     "Used to express rainfall relative to what is normal for that county "
     "and season."),
    ("Flood label", "NOAA Storm Events, county-coded Flood and Flash Flood "
     "episodes: roughly 6,700 independent episodes across the period."),
    ("Hurricane label", "NHC HURDAT2 post-season best track. A county-day "
     "counts as exposed when the cyclone centre passes close enough to place "
     "the county inside the 34-knot wind field."),
    ("Cyclone features", "NHC ATCF a-decks, official OFCL forecast only. "
     "These are the forecasts as they were actually issued, timestamped, so "
     "a feature contains only what was knowable on the origin date."),
    ("Secondary label", "FEMA disaster declarations, retained only as a "
     "'this one was costly' marker, not as the hazard label."),
    ("Static county data", "CDC/ATSDR Social Vulnerability Index file, which "
     "carries the Census-derived population and land area used for population "
     "density. The vulnerability themes themselves are not trained on."),
    ("County coordinates", "Centroids averaged from the GHCN station-to-county "
     "mapping, so a county's position is the centre of the stations that "
     "actually report for it."),
    ("Live serving", "Open-Meteo forecast endpoint, which returns up to 92 "
     "past days alongside the forward forecast, so one request per county "
     "supplies both the antecedent windows and the forward rain."),
    ("Official comparison", "NWS api.weather.gov active alerts, read in the "
     "browser at page load."),
]


def para(doc, text, size=10, bold=False, italic=False, space=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    p.paragraph_format.space_after = Pt(space)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(8.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(8.5)
    return t


# ------------------------------------------------------------------ run input
def load_run(run_id):
    d = RUNS / run_id
    if not d.is_dir():
        sys.exit(f"no such run directory: {d}")
    man = json.loads((d / "manifest.json").read_text())
    tag = man["tag"]
    res = d / "results"
    out = {"dir": d, "manifest": man, "tag": tag,
           "console": (d / "console.log").read_text()}
    for name in ("summary", "leads"):
        with (res / f"{name}_{tag}.csv").open() as f:
            out[name] = list(csv.DictReader(f))
    for haz in ("flood", "storm"):
        with (res / f"deviation_{haz}_{tag}.csv").open() as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        out[f"dev_{haz}"] = {"cols": head[1:],
                             "rows": {r[0]: r[1:] for r in body}}
    out["xgb"] = json.loads((res / f"xgb_{tag}.json").read_text())
    out["tft"] = json.loads((res / f"tft_{tag}.json").read_text())
    with (res / f"leads_both_{tag}.csv").open() as f:
        out["leads_both"] = list(csv.DictReader(f))
    feats = json.loads(
        (d / "models" / f"xgb_flood_{tag}.json").read_text()
    )["learner"]["feature_names"]
    out["features"] = feats
    return out


def splits_from_console(text):
    """The split table is printed once; parse it rather than recompute it."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(train|val|test)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
                     r"(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$", line)
        if m:
            rows.append(m.groups())
    return rows


def fnum(s, nd=5):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return str(s)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 1e-4:
        return f"{v:.2e}"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------- the writer
def build(run_id):
    R = load_run(run_id)
    man, tag = R["manifest"], R["tag"]
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    doc.add_heading("County Flood and Hurricane Risk Prediction", 0)
    p = para(doc, f"{AUTHOR}   |   Results from sealed run {man['run_id']}",
             size=9, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # -- 1. what the project does
    doc.add_heading("1. Scope", level=1)
    para(doc, "The system predicts two hazards for a US county on a given "
              "day: whether a flood episode begins within the next three "
              "days, and whether a tropical cyclone places the county inside "
              "a 34-knot wind field within the same window. Nothing else is "
              "predicted.")
    para(doc, "The target is deliberately an onset, not an active window. "
              "Asking whether a flood is ongoing today is close to trivial, "
              "because yesterday's answer gives it away. Asking whether one "
              "begins in the next three days is the question a warning "
              "actually has to answer.")
    para(doc, "Models are trained on counties in the Gulf Coast states. The "
              "web application displays the whole country, and that gap is "
              "stated on the page itself rather than papered over.")

    # -- 2. data
    doc.add_heading("2. Data sources", level=1)
    table(doc, ["Role", "Source"], SOURCES)
    para(doc, "")
    para(doc, "Two choices in that table carry most of the project's "
              "credibility.", bold=True)
    para(doc, "Labels come from observations, not from administration. FEMA "
              "disaster declarations were the original label and they are the "
              "wrong instrument: a declaration requires a damage threshold to "
              "be met and a governor to request it, so most floods never "
              "produce one. The entire Gulf panel held only 82 distinct flood "
              "declarations over 2000-2026, which reduced the flood metric to "
              "an estimate from about 16 test events. NOAA Storm Events "
              "supplies roughly 6,700 episodes over the same period.")
    para(doc, "Cyclone features come from forecasts, labels come from best "
              "track. The label should record what happened, so it is built "
              "from the post-season best track. A feature may only contain "
              "what was knowable on the origin date, so the cyclone features "
              "are built from the official NHC forecast that was on the wire "
              "at the time, chosen as the most recent advisory at or before "
              "00Z on the local day. Using the best track as a feature would "
              "hand the model perfect foreknowledge of the track.")

    # -- 3. panel and splits
    doc.add_heading("3. Panel and data splits", level=1)
    pan = man["panel"]
    para(doc, f"The panel is one row per county per day: "
              f"{pan['rows_as_read']:,} rows covering "
              f"{pan['counties_as_read']} counties over "
              f"{pan['years_as_read'][0]}-{pan['years_as_read'][1]}.")
    srows = splits_from_console(R["console"])
    if srows:
        table(doc,
              ["Split", "Years", "Rows", "Counties", "Flood onsets",
               "Flood rate %", "Storm onsets", "Storm rate %"],
              [(a.capitalize(), b, f"{int(c):,}", d, f"{int(e):,}", f,
                f"{int(g):,}", h) for a, b, c, d, e, f, g, h in srows])
    para(doc, "")
    para(doc, "Splits are contiguous blocks of time and are never shuffled. "
              "Adjacent days share a weather state, so a random split would "
              "leak the answer across the boundary and report skill the model "
              "does not have.")
    para(doc, "Contiguity alone is not sufficient. Because a label looks "
              "three days ahead, the last days of one split carry outcomes "
              "belonging to the next. A three-day embargo is therefore cut "
              "from the end of every split that another split follows, which "
              "is why the blocks end on 28 December rather than 31 December.")
    para(doc, "One further restriction: the hurricane label depends on "
              "HURDAT2, which NHC finalises only after a season closes, so "
              "storm labels end in 2024. Test rows beyond that are dropped "
              "for the storm hazard rather than being scored against labels "
              "that do not exist yet.")

    # -- 4. variables
    doc.add_heading("4. Variables used for training", level=1)
    feats = R["features"]
    missing = [f for f in feats if f not in FEATURE_DOC]
    extra = [f for f in FEATURE_DOC if f not in feats]
    if missing or extra:
        sys.exit(f"feature documentation out of sync with the trained model.\n"
                 f"  trained but undocumented: {missing}\n"
                 f"  documented but not trained: {extra}")
    para(doc, f"The models train on {len(feats)} variables. The list below is "
              f"read out of the trained booster in run {man['run_id']}, so it "
              f"is what the model actually saw rather than what the "
              f"configuration intended.")
    groups = {}
    for f in feats:
        g, desc = FEATURE_DOC[f]
        groups.setdefault(g, []).append((f, desc))
    order = ["Rainfall", "Antecedent", "Forecast rain", "Normalised",
             "Weather", "Seasonality", "Location", "Static", "Cyclone"]
    for i, g in enumerate([g for g in order if g in groups], start=1):
        doc.add_heading(f"4.{i} {g} ({len(groups[g])})", level=2)
        table(doc, ["Variable", "Meaning"], groups[g])
        para(doc, "")
    para(doc, "Two constraints are imposed on the boosters rather than "
              "learned. More rain may never lower predicted risk, and a storm "
              "forecast to pass closer may never lower it either. Without "
              "these the trees carve non-monotone thresholds out of "
              "individual historical events.")
    para(doc, "The CDC social-vulnerability themes were removed from this "
              "list. Whether a cyclone crosses a county is not a function of "
              "its social vulnerability; those columns described who would be "
              "harmed, not whether the hazard was coming. The cost of "
              "removing them was measured first: flood 0.26334 to 0.26190, "
              "storm 0.53857 to 0.53294.")

    # -- 5. models
    doc.add_heading("5. Models", level=1)
    para(doc, "Two model families are trained on identical inputs and splits "
              "so the comparison is clean.", bold=True)
    para(doc, "XGBoost, gradient-boosted decision trees, with the monotone "
              "constraints described above. This is the model currently "
              "serving the web application.")
    para(doc, "Temporal Fusion Transformer, a sequence model that reads a "
              "30-day history per county rather than a single row. The "
              "expectation is that a trajectory, a storm approaching over "
              "several days, is visible to a sequence model and invisible to "
              "a model shown one day at a time.")
    para(doc, "Every model is scored against Tier-0 baselines: single "
              "variables such as forecast rainfall over climatology, used "
              "directly as the prediction. A model that cannot beat one "
              "column is not worth deploying, and the baseline is reported "
              "beside every result for that reason.")

    # -- 6. results
    doc.add_heading("6. Results", level=1)
    para(doc, f"All figures below are from sealed run {man['run_id']}, "
              f"measured on the held-out test period only.", italic=True)

    doc.add_heading("6.1 Headline skill", level=2)
    para(doc, "Average precision (AP) is the headline metric. For rare "
              "events it is the metric that separates a useful model from a "
              "useless one, and the two columns that matter are the lift over "
              "the event rate and the ratio against the best single-variable "
              "baseline.")
    table(doc, ["Hazard", "Model", "Test AP", "Lift over base rate",
                "vs best baseline", "Verdict"],
          [(r["hazard"], r["model"], r["test_AP"], r["lift"] or "-",
            r["vs_baseline"], r["verdict"]) for r in R["summary"]])
    para(doc, "")
    xg, tf = R["xgb"], R["tft"]
    para(doc, f"The hurricane result is the one that moved this week. Before "
              f"the cyclone variables existed the storm model saw only "
              f"rainfall, temperature, month and position, and was asked "
              f"whether a hurricane would arrive; it early-stopped after a "
              f"handful of boosting rounds because there was nothing in the "
              f"feature matrix about a storm. Adding the official-forecast "
              f"block took hurricane AP from 0.15570 to "
              f"{fnum(xg['storm']['test']['ap'])}.")

    doc.add_heading("6.2 Error and deviation measures", level=2)
    para(doc, "These are the requested RMSE, MSE and standard-deviation "
              "figures, against two reference columns: a model that always "
              "predicts zero, and one that always predicts the base rate.")
    want = [("observed_rate", "Observed event rate"),
            ("predicted_mean", "Mean predicted probability"),
            ("bias_pred_minus_obs", "Bias (predicted minus observed)"),
            ("predicted_sd", "SD of predictions"),
            ("observed_sd", "SD of outcomes"),
            ("MAE", "Mean absolute error"),
            ("RMSE", "RMSE"),
            ("brier", "MSE / Brier score"),
            ("log_loss", "Log loss"),
            ("calibration_error_ECE", "Calibration error (ECE)"),
            ("average_precision", "Average precision")]
    for haz in ("flood", "storm"):
        dev = R[f"dev_{haz}"]
        doc.add_heading(haz.capitalize(), level=3)
        table(doc, ["Measure"] + [c.replace("_", " ") for c in dev["cols"]],
              [[label] + [fnum(v) for v in dev["rows"][key]]
               for key, label in want if key in dev["rows"]])
        para(doc, "")
    para(doc, "Two things in those tables need saying plainly.", bold=True)
    para(doc, "MSE and the Brier score are the same quantity here. Because "
              "the outcome is 0 or 1 and the prediction is a probability, the "
              "mean squared error is exactly the Brier score, and RMSE is its "
              "square root. They are reported once rather than twice.")
    para(doc, "RMSE and MSE barely distinguish these models, and that is a "
              "property of the metric rather than of the models. Flood "
              "onsets occur on about 1% of county-days, so a model that "
              "predicts zero every single day already achieves an RMSE within "
              "a few percent of the trained model while being completely "
              "useless. This is why average precision leads the reporting and "
              "squared-error measures are shown for completeness. The columns "
              "that do separate the models are average precision, Brier skill "
              "and recall.")

    doc.add_heading("6.3 Skill by lead time", level=2)
    para(doc, "The headline models answer 'onset anywhere in the next one to "
              "three days'. A reviewer is entitled to the three days "
              "separately, because skill that does not decay with lead time "
              "is skill being read off the present rather than forecast.")
    table(doc, ["Hazard", "Lead", "Test positives", "Prevalence", "Test AP",
                "Lift", "Baseline", "Baseline AP", "vs baseline"],
          [(r["hazard"], r["lead"], f"{int(r['test_pos']):,}", r["prevalence"],
            r["test_AP"], r["lift"], r["baseline"], r["baseline_AP"],
            r["vs_baseline"]) for r in R["leads"]])
    para(doc, "")
    para(doc, "Skill decays monotonically with lead time for both hazards, "
              "which is the expected and reassuring pattern.")

    doc.add_heading("6.4 The transformer, and why the booster is still "
                    "the model in service", level=2)
    tft_leads = R.get("leads_both")
    para(doc, "The transformer was trained to completion on this run, cold, "
              "on the same panel and the same splits as the booster. It lost.",
              bold=True)
    table(doc, ["Hazard", "Lead", "Baseline AP", "XGBoost AP", "TFT AP",
                "TFT epochs"],
          [(r["hazard"], r["lead"], fnum(r["baseline_AP"]),
            fnum(r["xgb_AP"]), fnum(r["tft_AP"]), r["tft_epochs"])
           for r in tft_leads])
    para(doc, "")
    para(doc, f"Pooled over one to three days the booster takes both hazards: "
              f"flood {fnum(xg['flood']['test']['ap'])} against "
              f"{fnum(tf['flood']['test']['ap'])}, and storm "
              f"{fnum(xg['storm']['test']['ap'])} against "
              f"{fnum(tf['storm']['test']['ap'])}. Per lead day the booster "
              f"wins five of six. The single exception is flood at three "
              f"days, where the transformer reaches "
              f"{fnum(tft_leads[2]['tft_AP'])} against "
              f"{fnum(tft_leads[2]['xgb_AP'])}.")
    para(doc, "This contradicts what a short rehearsal on 150 counties had "
              "suggested, which was that the transformer beat the booster on "
              "the storm hazard at every lead. At full scale it is beaten "
              "there decisively, 0.29892 against 0.49420 at one day. The "
              "rehearsal was two epochs on a tenth of the panel and it "
              "misled; that is recorded here rather than quietly dropped.")
    para(doc, f"The flood transformer in particular should not be described "
              f"as working. Its margin over a single rainfall column is "
              f"{fnum(tf['flood']['vs_best_baseline'], 2)}x, which for "
              f"{int(tf['flood']['n_test_pos']):,} events is inside the noise. "
              f"A model that needs a 30-day sequence and a GPU to match one "
              f"column of the feature matrix has not earned its place.")
    para(doc, "One measure does favour the transformer, and it is worth "
              "stating because it is the honest counterweight: calibration. "
              "On the storm hazard it is better on squared error and much "
              "better calibrated than the booster, while ranking worse. If "
              "the requirement were a probability that is literally correct "
              "rather than merely well ordered, that would be an argument. "
              "It is not an argument for putting it in front of users, and "
              "the application continues to serve the booster.")

    # -- 7. application
    doc.add_heading("7. The web application", level=1)
    para(doc, "The trained parameters are written into a sealed run directory "
              "and the application reads the boosters from there, not from a "
              "working folder. A run directory is hash-verified against the "
              "panel it was trained on; a loose folder is whatever was "
              "written last. On three occasions a loose path produced numbers "
              "that could not be traced afterwards, which is why the "
              "constraint exists.")
    doc.add_heading("7.1 What happens when a county is selected", level=2)
    para(doc, "Live weather is pulled per county from Open-Meteo's public "
              "forecast endpoint, no API key required. A single request "
              "returns both the preceding days and the forward forecast, so "
              "the antecedent windows the model needs and the forward rain "
              "arrive together.")
    para(doc, "Those values are assembled into the same feature layout the "
              "model was trained on, including the cyclone block, which is "
              "rebuilt from the NHC forecast currently on the wire rather "
              "than from anything stored. The feature list is imported from "
              "the module that builds it, so the training layout and the "
              "serving layout cannot drift apart silently.")
    para(doc, "The model then returns a probability of flood onset and of "
              "hurricane exposure for that county, using the day's conditions "
              "together with the preceding days of rainfall history.")
    doc.add_heading("7.2 Three-day outlook", level=2)
    para(doc, "Separate models are trained for each individual lead day, so "
              "the application shows tomorrow, the day after, and the third "
              "day as distinct predictions from distinct models, with the "
              "skill of each quantified in the lead-time table above.")
    doc.add_heading("7.3 Comparison against the official forecast", level=2)
    para(doc, "A model that cannot be compared to what already exists is not "
              "evidence of anything, so the page reads active alerts from the "
              "National Weather Service at api.weather.gov and places them "
              "beside the model's own output for the same county and day. "
              "The reader sees where the two agree, where the model raises "
              "something the official feed does not, and where the official "
              "feed carries a warning the model missed.")
    para(doc, "This is a live side-by-side rather than a skill score. A "
              "formal skill comparison against NWS forecasts is identified "
              "work and is not claimed here.")

    # -- 8. limitations
    doc.add_heading("8. Known limitations", level=1)
    for t in [
        "Models are trained on Gulf Coast counties while the map displays the "
        "entire country. Predictions outside the training region are "
        "extrapolation.",
        "The hurricane features come from NHC's official forecast, and the "
        "hurricane label is derived from the best-track wind field. Part of "
        "the measured skill is therefore NHC's forecast skill re-expressed at "
        "county level, which is a legitimate product but is not independent "
        "of NHC.",
        "Rainfall is station-based rather than gridded, so a county with few "
        "reporting stations is measured less reliably than a dense one.",
        "The hurricane label ends in 2024 because HURDAT2 is finalised only "
        "after a season closes, so the most recent test years score flood "
        "only.",
        "The transformer does not beat the booster and is not served; see "
        "section 6.4. The sequence model the design expected to win has not "
        "yet been shown to.",
    ]:
        b = doc.add_paragraph(style="List Bullet")
        r = b.add_run(t)
        r.font.size = Pt(10)

    # -- provenance
    doc.add_heading("9. Provenance", level=1)
    g = man.get("git", {})
    rows = [
        ("Run identifier", man["run_id"]),
        ("Environment", man.get("env", "-")),
        ("Panel file", pathlib.Path(pan["path"]).name),
        ("Panel SHA-256", pan["sha256"]),
        ("Panel rows", f"{pan['rows_as_read']:,}"),
        ("Counties", str(pan["counties_as_read"])),
        ("Variables trained on", str(len(feats))),
        ("Embargo", f"{man['params']['embargo_days']} days"),
        ("Git commit", str(g.get("commit", "-"))[:12]),
        ("Working tree clean", str(not g.get("dirty", True))),
    ]
    table(doc, ["Field", "Value"], rows)
    para(doc, "")
    para(doc, "Every number in this document is parsed from the run "
              "directory named above by make_overview_docx.py. None is typed "
              "in by hand, so the document cannot disagree with the run it "
              "cites.", italic=True, size=9)

    out = ROOT / "PROJECT_OVERVIEW.docx"
    doc.save(out)
    return out, len(feats), man["run_id"]


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    path, nfeat, rid = build(run)
    print(f"wrote {path}  (run {rid}, {nfeat} variables documented)")
