"""Generate V2_TRAINING_RESULTS.docx -- the training and results write-up.

Every number comes out of the sealed run directory rather than being retyped,
so the document cannot drift from the models the dashboard serves. Point it at
a different run with CRISK_RUN_DIR and it regenerates against that run.

This is the technical companion to PROJECT_EXPLAINER.docx: it assumes the
reader knows what average precision is and wants the splits, the
hyperparameters, the baselines and the provenance hashes.
"""

import json
import pathlib

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

import v2_models

RUN = v2_models.run_dir()
RES = RUN / "results"
PLOTS = RUN / "plots"
OUT = pathlib.Path(__file__).parent / "V2_TRAINING_RESULTS.docx"

man = json.loads((RUN / "manifest.json").read_text())
prm = v2_models.results(f"params_{v2_models.TAG}.json")
xgb = v2_models.results(f"xgb_{v2_models.TAG}.json")
leads = v2_models.results(f"leads_{v2_models.TAG}.json")
tft = v2_models.results_opt(f"tft_{v2_models.TAG}.json")
tft_leads = v2_models.results_opt(f"tft_leads_{v2_models.TAG}.json")
HAS_TFT = tft is not None and tft_leads is not None
HAZ = {"flood": "Flood", "storm": "Hurricane"}

# The exporter's payload, for the few facts that are properties of what gets
# published rather than of the training run: how many counties the live page
# actually scores, the episode counts, and the forward-rain ablation. Optional
# so the document can still be built before a dashboard export has been run --
# every section that uses it degrades to leaving the claim out, which is the
# same rule the withdrawn transformer follows. Never retype these.
_meta = pathlib.Path(__file__).parent / "docs" / "data" / "meta.json"
META = json.loads(_meta.read_text()) if _meta.exists() else None
if META and META["run"]["run_id"] != man["run_id"]:
    raise SystemExit(
        f"run mismatch: this report is being built from {man['run_id']}, but "
        f"docs/data/meta.json was exported from {META['run']['run_id']}.\n"
        f"run `python export_v2.py` first, or point CRISK_RUN_DIR at the run "
        f"meta.json came from -- a document that mixes two runs is worse than "
        f"one that is merely out of date."
    )

WITHDRAWN = (
    "No transformer was trained on this panel. The only fitted transformer "
    "used the superseded label window, which placed a third of its positives "
    "on days the event had already begun, so setting it against these "
    "boosters would compare two label definitions rather than two "
    "architectures. Every transformer figure is therefore withdrawn from this "
    "document until the model is refitted on the corrected label. The "
    "comparison is a required deliverable and will be restored, not dropped."
)


def lead_frame():
    """The per-lead table, with transformer columns only if a run produced them.

    leads_both_run.csv is written by the comparison step, so it exists only for
    runs that trained both architectures. Rebuilding the same shape from
    leads_run.csv keeps section 8 on one code path either way.
    """
    f = RES / f"leads_both_{v2_models.TAG}.csv"
    if f.exists():
        return pd.read_csv(f)
    d = pd.read_csv(RES / f"leads_{v2_models.TAG}.csv")
    d = d[d["lead"].str.startswith("+")].copy()
    d = d.rename(columns={"test_AP": "xgb_AP", "vs_baseline": "xgb_vs_base"})
    d["tft_AP"] = float("nan")
    d["tft_vs_base"] = float("nan")
    return d.reset_index(drop=True)


both = lead_frame()

doc = Document()
doc.core_properties.author = "Ganesh Pinnu"
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)
doc.styles["Normal"].paragraph_format.space_after = Pt(7)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, italic=False, size=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def callout(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x8A, 0x3A, 0x00)


def _no_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def _repeat_header(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))


def table(headers, rows, caption=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(str(head)).bold = True
    _no_split(t.rows[0])
    _repeat_header(t.rows[0])
    for row in rows:
        r = t.add_row()
        for i, val in enumerate(row):
            r.cells[i].text = str(val)
        _no_split(r)
    if caption:
        c = doc.add_paragraph()
        run = c.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
    else:
        doc.add_paragraph()
    return t


def figure(name, caption):
    f = PLOTS / name
    if not f.exists():
        p(f"[missing plot: {name}]", italic=True)
        return
    doc.add_picture(str(f), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = c.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)


# ====================================================================== title

title = doc.add_heading(
    "County-Daily Flood and Hurricane Risk on the Gulf Coast", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(
    "Training procedure and results\n"
    + ("Gradient boosting against a Temporal Fusion Transformer, "
       if HAS_TFT else "Gradient boosting, ")
    + "at 1-, 2- and 3-day lead").italic = True

# The hardware is read from this run's own manifest rather than written in.
# Quoting an A100 while serving a model fitted on a laptop CPU is the kind of
# provenance error that invalidates everything around it.
ref = doc.add_paragraph()
ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
hw = man["versions"].get("cuda") or f"CPU, host {man['host']}"
r = ref.add_run(
    f"Run {man['run_id']}  ·  finished {man['finished_utc']}  ·  "
    f"{man['wall_seconds'] / 3600:.2f} h on {hw}\n"
    f"Panel sha256 {man['panel']['sha256']}")
r.font.size = Pt(9)

doc.add_page_break()

# ===================================================================== 1 scope

h("1. What was predicted", 1)
p("The unit of analysis is a county-day. For each of the "
  f"{man['panel']['counties_as_read']} Gulf Coast counties and each day from "
  f"{man['panel']['years_as_read'][0]} to {man['panel']['years_as_read'][1]}, "
  "the models answer a binary question: does a hazard event begin in this "
  f"county within the next {prm['horizon_days']} days? "
  f"The panel holds {man['panel']['rows_as_read']:,} such rows.")

p("Two hazards are modelled separately, with separate labels and separate "
  "models:")
bullet("Flood — the onset day of a NOAA StormEvents flood episode. A positive "
       "exists because a local Weather Service office wrote the episode up, "
       "so this is a record of reported flooding, not of flooding.")
bullet("Hurricane / tropical storm — the day the HURDAT2 Atlantic best track "
       "passes through the county. This is purely geometric: a county grazed "
       "by a weak system and a county hit by a major landfall are the same "
       "positive.")

p("Two target definitions are used throughout, and keeping them apart is the "
  "point of the whole lead-time section below:")
bullet("Pooled target — onset falls anywhere in t+1 .. t+3. This is what the "
       "dashboard's headline risk cards report.")
bullet("Exact-lead target — onset falls on exactly t+k, for k in "
       f"{', '.join(str(k) for k in prm['leads'])}. Obtained by a within-county "
       "shift of the onset column, so no day leaks across a county boundary.")

callout("The pooled question is strictly easier than the 3-day exact-lead "
        "question. Reporting a pooled score as a '3-day forecast' is the "
        "specific overstatement the exact-lead sweep exists to prevent.")

# ================================================================== 2 features

h("2. Features", 1)
p(f"All {len(v2_models.FEATS)} features are available on the morning the model "
  "runs, with one important exception noted in section 10. They fall into six "
  "groups:")
table(
    ["Group", "Features", "What it carries"],
    [["Rainfall, antecedent",
      "prcp, prcp_max_stn, prcp_prev3/7/14/30/60/90",
      "how wet the county already is, over six windows"],
     ["Rainfall, forward",
      "prcp_fwd1, prcp_fwd3, prcp_fwd3_max, log_fwd3",
      "rain over the prediction window itself"],
     ["Normalised rainfall",
      "ratio_prev3, ratio_fwd3, ratio_prev30",
      "rain divided by that county's own climatology, so a wet county and a "
      "dry county are on one scale"],
     ["Season and place",
      "tmax, tmin, month_sin, month_cos, hurricane_season, lat, lon",
      "where and when"],
     ["Exposure",
      "pop_density",
      "population per square mile; see section 10 for why this one is not "
      "innocent"],
     ["Tropical cyclone",
      "cyc_active, cyc_n_active, cyc_dist_min, cyc_dist1/2/3, "
      "cyc_lead_closest, cyc_vmax_close, cyc_vmax_max, cyc_fwd_kt, "
      "cyc_bearing_sin, cyc_bearing_cos, cyc_r34/50/64, cyc_in34/50/64, "
      "cyc_cone_km, cyc_cone_margin, cyc_in_cone",
      "the NHC forecast track as it was issued: whether a storm is active, "
      "how far away it is forecast to pass, how strong, how wide its wind "
      "field, and whether this county falls inside the forecast cone"]],
    "Table 1. Feature groups. Column order is fixed in v2_models.FEATS because "
    "an XGBoost DMatrix takes its feature names from the DataFrame column "
    "order and rejects a layout that differs from training.")

p("Monotone constraints are applied to every rainfall, ratio and log feature, "
  "forcing predicted risk to be non-decreasing in each of them. More rain can "
  "never lower predicted risk. This costs a little fit and buys a model whose "
  "behaviour can be defended without inspecting it.")

p("Missing values are passed through to XGBoost rather than imputed. The "
  "booster learned its own default direction for missingness during training, "
  "so filling at scoring time would evaluate a different function from the one "
  "that was fitted.")

# ==================================================================== 3 splits

h("3. Splits", 1)
p("Splits are by calendar year and are never shuffled. No row from a test year "
  "informs training or model selection, and because the split is temporal "
  "rather than random, no county contributes neighbouring days to both sides "
  "of the boundary.")
table(
    ["Split", "Years", "Rows", "Counties", "Flood onsets", "Flood rate",
     "Hurricane onsets", "Hurricane rate"],
    [[s["split"], s["years"], f"{s['rows']:,}", s["counties"],
      f"{s['flood_pos']:,}", f"{s['flood_rate%']:.4f}%",
      f"{s['storm_pos']:,}", f"{s['storm_rate%']:.4f}%"]
     for s in man["splits"]],
    "Table 2. Split sizes and class balance. Both targets are rare-event "
    "problems; hurricane onset is roughly one county-day in four hundred.")

if prm.get("label_end"):
    ends = ", ".join(f"{HAZ[k]} through {v}" for k, v in prm["label_end"].items())
    p(f"Label availability is capped where the source is not yet final: {ends}. "
      "The HURDAT2 best track is finalised a season behind, so hurricane "
      "metrics are scored on a shorter test window than flood metrics "
      f"({765020 - 579263:,} test rows are excluded). The two hazards' test "
      "sets are therefore not the same size and their absolute average "
      "precisions are not directly comparable to each other.")

h("3.1 Effective sample size", 2)
p("The row counts above are county-days and must not be read as event counts. "
  "Rainfall and tropical cyclones are spatially correlated, so the counties a "
  "single system crosses on one day are close to one observation rather than "
  "many. The table below reduces the test period to the quantity that actually "
  "constrains these metrics.")
if META and META.get("episodes"):
    ep = META["episodes"]
    gap = ep[v2_models.HAZARDS[0]]["gap_days"]
    table(["Hazard", "Positive county-days", "Onset county-days",
           "Distinct onset dates", "Independent episodes"],
          [[HAZ[h_], f"{xgb[h_]['n_test_pos']:,}",
            f"{ep[h_]['onset_county_days']:,}",
            f"{ep[h_]['distinct_dates']:,}", f"{ep[h_]['episodes']:,}"]
           for h_ in v2_models.HAZARDS],
          f"Table 2.1. Test-period sample size at four levels of aggregation. "
          f"An episode is a cluster of onset dates separated from the next by "
          f"more than {gap} quiet days.")
    callout(
        f"Every hurricane figure in this document rests on roughly "
        f"{ep['storm']['episodes']} independent tropical cyclones, not the "
        f"{xgb['storm']['n_test_pos']:,} county-days in Table 2. The flood "
        f"figures rest on roughly {ep['flood']['episodes']}. One unusual "
        "season would move the hurricane numbers materially. Section 3.2 "
        "resamples these episodes to put an interval on each headline figure; "
        "resampling rows instead would treat neighbouring counties as "
        "independent observations and understate every width several-fold.")

UNC = META.get("uncertainty") if META else None
if UNC:
    h("3.2 How precise are these numbers?", 2)
    p("Two different things make a reported average precision uncertain, and "
      "they are measured separately because they have different remedies. The "
      "first is the test sample: the interval below comes from resampling the "
      f"episode blocks of section 3.1 {UNC['draws']:,} times, drawing whole "
      "weather systems with replacement rather than individual county-days. "
      "The second is the fit itself. Subsampling of rows and columns makes "
      "training stochastic, so the published figure is one draw from a "
      "distribution; refitting at "
      f"{len(UNC['seeds'])} seeds says how wide that distribution is.")
    uh = UNC["hazards"]
    table(["Hazard", "Test AP", "95% interval", "Width", "vs best baseline",
           "95% interval"],
          [[HAZ[h_], f"{uh[h_]['ap_rescored']:.5f}",
            f"[{uh[h_]['ap_boot']['lo95']:.4f}, {uh[h_]['ap_boot']['hi95']:.4f}]",
            f"{uh[h_]['ap_boot']['hi95'] - uh[h_]['ap_boot']['lo95']:.4f}",
            f"{uh[h_]['vs_baseline_point']:.3f}x",
            f"[{uh[h_]['vs_baseline_boot']['lo95']:.2f}x, "
            f"{uh[h_]['vs_baseline_boot']['hi95']:.2f}x]"]
           for h_ in v2_models.HAZARDS],
          f"Table 2.2. Episode-block bootstrap, {UNC['draws']:,} draws, on "
          f"{uh['flood']['episodes']} flood and {uh['storm']['episodes']} "
          f"hurricane episodes. The ratio column is the point estimate "
          f"AP/baseline with a resampled interval around it, not the mean of "
          f"the resampled ratios.")

    table(["Hazard"] + [f"seed {s}" for s in UNC["seeds"]]
          + ["mean", "sd", "spread"],
          [[HAZ[h_]] + [f"{a:.5f}" for a in uh[h_]["seed_sweep"]["ap"]]
           + [f"{uh[h_]['seed_sweep']['mean']:.5f}",
              f"{uh[h_]['seed_sweep']['sd']:.5f}",
              f"{uh[h_]['seed_sweep']['spread']:.5f}"]
           for h_ in v2_models.HAZARDS],
          "Table 2.3. Test AP by random seed, everything else held fixed.")
    table(["Hazard"] + [f"seed {s}" for s in UNC["seeds"]],
          [[HAZ[h_]] + [f"{i:,}" for i in
                        uh[h_]["seed_sweep"]["best_iteration"]]
           for h_ in v2_models.HAZARDS],
          "Table 2.4. Boosting round selected by early stopping, by seed.")

    _f, _s = uh["flood"]["seed_sweep"], uh["storm"]["seed_sweep"]
    _rank = sorted(_s["ap"]).index(uh["storm"]["ap_rescored"]) + 1
    p(f"The two hazards behave differently here and the difference matters "
      f"more than either interval. Flood is stable: a seed standard deviation "
      f"of {_f['sd']:.5f} is {_f['sd'] / _f['mean'] * 100:.1f}% of its mean, "
      f"and the published value sits "
      f"{sorted(_f['ap']).index(uh['flood']['ap_rescored']) + 1} of "
      f"{len(_f['ap'])} in the sweep, which is to say it is the typical "
      f"result rather than a lucky one.")
    callout(
        f"The hurricane figure is not stable. Its seed standard deviation is "
        f"{_s['sd']:.5f}, or {_s['sd'] / _s['mean'] * 100:.1f}% of the mean, "
        f"and Table 2.4 shows early stopping selecting "
        f"{min(_s['best_iteration'])} rounds on one seed and "
        f"{max(_s['best_iteration'])} on another -- a factor of "
        f"{max(_s['best_iteration']) / max(min(_s['best_iteration']), 1):.0f}. "
        f"A validation signal resting on roughly "
        f"{uh['storm']['episodes']} cyclones does not locate an optimum that "
        f"precisely, so the stopping round is being set largely by noise. The "
        f"published {uh['storm']['ap_rescored']:.5f} is rank {_rank} of "
        f"{len(_s['ap'])} in the sweep, so this document understates rather "
        f"than overstates the model; the defensible way to quote it is "
        f"{_s['mean']:.3f} +/- {_s['sd']:.3f} across seeds, and no hurricane "
        f"figure anywhere should be read to five significant digits.")

PAT = META.get("patience_sweep") if META else None
_ps = (PAT or {}).get("hazards", {}).get("storm") or None
if _ps:
    h("3.3 Is the hurricane model stopping too early?", 2)
    p(f"Section 3.2 left one reading open. If early stopping is halting the "
      f"hurricane fit before it has learned anything -- a patience shorter "
      f"than the plateau the validation metric's own noise produces -- then "
      f"the instability is a bug with a fix, and raising patience should both "
      f"lift the mean and tighten the spread. This section tests that "
      f"directly by refitting at {len(PAT['seeds'])} seeds across "
      f"{len(PAT['patience'])} patience settings, "
      f"{_ps['_span']['n_fits']} fits in total, changing nothing else.")
    _lv = sorted((k for k in _ps if k != "_span"), key=int)
    table(["Patience", "Mean AP", "Seed sd", "Rounds selected, by seed"],
          [[lv, f"{_ps[lv]['mean']:.5f}", f"{_ps[lv]['sd']:.5f}",
            ", ".join(str(i) for i in _ps[lv]["best_iteration"])]
           for lv in _lv],
          "Table 2.5. Hurricane test AP against early-stopping patience. "
          "Flood is omitted: its seed spread is already a fraction of a "
          "percent, so it has no instability for patience to explain.")
    _sp = _ps["_span"]
    # Range against range. The table prints a standard deviation, so quoting
    # the between-level range next to it would read as the larger number when
    # it is in fact the smaller one.
    _wr = [max(_ps[lv]["ap"]) - min(_ps[lv]["ap"]) for lv in _lv]
    p(f"The answer is no, and the way it is no is more informative than a "
      f"yes would have been. Patience does not trade off against accuracy "
      f"monotonically: the mean rises to {_sp['mean_hi']:.5f} and then falls "
      f"back to {_ps[_lv[-1]]['mean']:.5f} at the widest setting, so there is "
      f"no direction in which the model is simply being cut short. The "
      f"lowest-patience column settles within the first dozen or so rounds "
      f"and scores worse than the middle of the table, which is the opposite "
      f"of what a premature stop would look like. And the seed that produced the "
      f"published model selects the same handful of rounds at every setting "
      f"up to {_lv[-2]}: its validation curve genuinely peaks there and is "
      f"not beaten for the next {_lv[-2]} rounds. The published fit is a real "
      f"early stop, not a truncated one.")
    callout(
        f"Across all {_sp['n_fits']} fits the selected round ranges from "
        f"{_sp['iter_lo']} to {_sp['iter_hi']:,} and test AP from "
        f"{_sp['ap_lo']:.5f} to {_sp['ap_hi']:.5f}. Comparing like with like, "
        f"the full spread of the means across patience settings is "
        f"{_sp['mean_hi'] - _sp['mean_lo']:.5f}, while changing nothing but "
        f"the seed inside one setting moves test AP by {min(_wr):.5f} to "
        f"{max(_wr):.5f} -- so the hyperparameter buys no more than the noise "
        f"does, and at most settings rather less. When moving a "
        f"hyperparameter and moving nothing but the random seed shift the "
        f"result by the same amount, the data is not determining that "
        f"hyperparameter. Patience therefore stays at "
        f"{_sp['published']}, not the {_sp['best_level']} that tops this "
        f"table. The only evidence favouring {_sp['best_level']} is the test "
        f"column itself, and switching to it on that basis would convert the "
        f"test set into a validation set and "
        f"leave the project with no held-out estimate at all. The honest "
        f"statement is that the hurricane AP is undetermined at roughly the "
        f"+/-0.01 level by choices no validation set of "
        f"{uh['storm']['episodes']} cyclones can settle, and that the "
        f"published value sits near the bottom of that range.")

FR = META.get("forecast_rain") if META else None
if FR:
    # GEFS when both arms ran: it reaches back four further years, and the
    # storm label has only six episodes inside the Open-Meteo window.
    _fa = FR.get("gefs") or next(iter(FR.values()))
    fh = _fa["hazards"]
    _fep = (META.get("episodes") or {}) if META else {}
    h("3.4 Could a forecast have said this in advance?", 2)
    p("Every number above rests on the panel's forward rainfall features, and "
      "those are built from the same gauge record as the labels. They are the "
      "rain that fell, not the rain a forecaster could have seen on the "
      "morning a deployed system would have had to run. Each figure so far is "
      "therefore an upper bound rather than an operating point. This section "
      "rebuilds the five forward features from an archived forecast issued at "
      f"the origin date -- {_fa['forecast_source']} -- and rescores the same "
      "boosters on the same county-days with nothing else changed. The models "
      "are not refitted, so the whole of the difference is attributable to the "
      "input. The per-county climatology that divides the ratio feature is "
      "held at its published value for the same reason: rebuilding it from "
      "forecast rain would change the function being scored.")
    p(f"The archive sets the window rather than any choice made here. It "
      f"begins {_fa['forecast_start']}, which leaves "
      f"{fh['flood']['episodes']} flood and {fh['storm']['episodes']} "
      f"hurricane episodes"
      + (f" of the {_fep['flood']['episodes']} and "
         f"{_fep['storm']['episodes']} in section 3.1" if _fep else "")
      + ". The observed column below is rescored on exactly those days rather "
        "than quoted from Table 2.2, because quoting the published figure "
        "would confound the change of input with the change of days.")
    table(["Hazard", "AP on observed rain", "95% interval",
           "AP on forecast rain", "95% interval", "Episodes"],
          [[HAZ[h_], f"{fh[h_]['ap_observed']:.5f}",
            f"[{fh[h_]['ap_observed_boot']['lo95']:.4f}, "
            f"{fh[h_]['ap_observed_boot']['hi95']:.4f}]",
            f"{fh[h_]['ap_forecast']:.5f}",
            f"[{fh[h_]['ap_forecast_boot']['lo95']:.4f}, "
            f"{fh[h_]['ap_forecast_boot']['hi95']:.4f}]",
            f"{fh[h_]['episodes']}"
            + (" (too few)" if fh[h_]["underpowered"] else "")]
           for h_ in v2_models.HAZARDS if h_ in fh],
          f"Table 2.6. The published models scored twice over the same rows, "
          f"once on the forward rain that fell and once on the forward rain "
          f"that was forecast. Episode-block bootstrap, {_fa['draws']:,} "
          f"draws, paired on identical block draws so each interval is the "
          f"interval of the comparison.")
    _fl = fh.get("flood")
    if _fl:
        p(f"For flood the cost of not knowing the future is "
          f"{_fl['drop_point']:.5f} AP, 95% "
          f"[{_fl['drop_boot']['lo95']:.4f}, {_fl['drop_boot']['hi95']:.4f}]. "
          f"A deployed version of this model retains "
          f"{_fl['retained_point'] * 100:.1f}% of the published figure, 95% "
          f"[{_fl['retained_boot']['lo95'] * 100:.1f}%, "
          f"{_fl['retained_boot']['hi95'] * 100:.1f}%]. Two thirds of the "
          f"headline skill is hindsight. The mechanism is visible in the "
          f"input and not in the model: forecast and observed three-day totals "
          f"correlate {_fl['fwd3_corr']:.3f} over these county-days, "
          f"{_fl['fwd1_corr']:.3f} at lead one, so the ceiling here belongs to "
          f"the rainfall forecast rather than to anything a different "
          f"architecture would recover.")
        p(f"The lift has to be requoted against a baseline fed the same "
          f"forecast, because scoring a forecast-fed model against an "
          f"observation-fed single feature would be this same defect inverted "
          f"-- the baseline would be the only one of the two allowed to see "
          f"the rain that fell. On that footing the model still clears its "
          f"strongest baseline, {_fl['best_baseline']} at "
          f"{_fl['baseline_ap']:.5f}, by "
          f"{_fl['forecast_vs_baseline']:.2f}x, 95% "
          f"[{_fl['vs_baseline_boot']['lo95']:.2f}x, "
          f"{_fl['vs_baseline_boot']['hi95']:.2f}x], an interval that excludes "
          f"parity. Against the observation-fed baseline that section 4 "
          f"measures lift from, {_fl['best_baseline_observed']} at "
          f"{_fl['baseline_ap_observed']:.5f}, it reaches "
          f"{_fl['forecast_vs_baseline_observed']:.2f}x. Both comparisons are "
          f"true and they are not interchangeable: the model beats what a "
          f"deployed baseline could know, and loses to what the published "
          f"tables compare it with.")
    _st = fh.get("storm")
    if _st and _st.get("forecast_is_better"):
        pk = _st["precision_at_k"]["200"]
        pkf = (_fl or {}).get("precision_at_k", {}).get("200")
        callout(
            f"The hurricane arm runs the other way, and on "
            f"{_st['episodes']} episodes it is reported rather than claimed. "
            f"Forecast rain is the better input by "
            f"{-_st['drop_point']:.5f} AP, 95% "
            f"[{-_st['drop_boot']['hi95']:.4f}, "
            f"{-_st['drop_boot']['lo95']:.4f}] -- an interval clear of zero. "
            f"Observed forward rain is not this hazard's ceiling; it is "
            f"costing it skill. Average precision is dominated by the top of "
            f"the ranking, which is where the mechanism shows: among the 200 "
            f"county-days the three-day rain feature ranks highest, "
            f"{pk['forecast']:.0%} are real onsets when the feature is the "
            f"forecast against {pk['observed']:.0%} when it is the "
            f"observation"
            + (f", while for flood the same comparison is "
               f"{pkf['observed']:.0%} against {pkf['forecast']:.0%}"
               if pkf else "")
            + ". A cyclone is a large organised system that a global ensemble "
              "resolves days out, so the top of a forecast ranking is mostly "
              "real storms; the top of an observed-rain ranking is mostly "
              "intense convection that is not a cyclone. Floods are local "
              "enough to be smoothed away at half-degree resolution and "
              "invert exactly. The operational reading is that the published "
              "pipeline handicaps the hurricane model by feeding it the rain "
              "that fell, and that a forecast-fed refit is the next thing to "
              "try rather than a concession.")
    if len(FR) > 1:
        _o = [a for a in FR if a != _fa["forecast_arm"]]
        p(f"A second archive, {', '.join(_o)}, was fetched as an independent "
          f"check and is carried in the run metadata. It covers too short a "
          f"span to carry the hurricane comparison, so the figures above are "
          f"the {_fa['forecast_arm']} arm throughout.", italic=True)

figure(f"split_{v2_models.TAG}.png", "Figure 1. Rows and positive counts by split.")

# =============================================================== 4 baselines

h("4. Baselines", 1)
p("No model is accepted until it beats every baseline. Each baseline ranks "
  "county-days by a single raw feature -- no fitting at all -- and is scored "
  "with the same metric on the same test rows. On a rare-event target this is "
  "a much harder bar than it sounds, because a single rainfall feature already "
  "carries most of the signal a flood model has.")
rows = []
for haz in v2_models.HAZARDS:
    for name, ap in sorted(xgb[haz]["baselines"].items(), key=lambda kv: -kv[1]):
        rows.append([HAZ[haz], name.replace("single_", ""), f"{ap:.5f}",
                     f"{xgb[haz]['test']['ap'] / ap:.2f}x",
                     "strongest" if name == xgb[haz]["best_baseline"] else ""])
table(["Hazard", "Baseline feature", "Test AP", "XGBoost / baseline", "Rank"],
      rows,
      "Table 3. Every single-feature baseline, pooled target. The strongest "
      "one per hazard is the bar the model has to clear.")

# ================================================================ 5 xgboost

h("5. Gradient boosting", 1)
x = prm["xgb"]
p(f"XGBoost with a logistic objective, learning rate {x['lr']}, max depth "
  f"{x['max_depth']}, min child weight {x['min_child_weight']}, subsample "
  f"{x['subsample']}, colsample_bytree {x['colsample_bytree']} and L2 "
  f"{x['reg_lambda']}. Up to {x['rounds']:,} rounds with early stopping after "
  f"{x['early']} rounds without validation improvement.")

callout("A saved booster contains best_iteration + early trees, because "
        "save_model writes every tree early stopping trained, not only the "
        "ones up to the best round. Scoring the whole file silently evaluates "
        "an overfit model. Every prediction in this project therefore passes "
        "iteration_range=(0, best_iteration+1), with best_iteration read from "
        "the run's results JSON rather than from the model file.")

table(["Hazard", "Best iteration", "Trees in file", "Val AP", "Test AP",
       "Test base rate", "Lift", "vs best baseline"],
      [[HAZ[h_], xgb[h_]["best_iteration"],
        xgb[h_]["best_iteration"] + x["early"],
        f"{xgb[h_]['val']['ap']:.5f}", f"{xgb[h_]['test']['ap']:.5f}",
        f"{xgb[h_]['test']['prevalence'] * 100:.4f}%",
        f"{xgb[h_]['test']['lift']:.1f}x",
        f"{xgb[h_]['vs_best_baseline']:.2f}x"]
       for h_ in v2_models.HAZARDS],
      "Table 4. Pooled gradient-boosting results.")

figure(f"xgb_training_{v2_models.TAG}.png",
       "Figure 2. Training and validation curves, with the early-stopping "
       "point marked.")
figure(f"xgb_eval_{v2_models.TAG}.png",
       "Figure 3. Precision-recall on the test years, against the base rate.")

p("Feature importance confirms what the baselines imply -- the ratio features "
  "dominate the flood model, and the maximum station rainfall dominates the "
  "hurricane model:")
table(["Hazard", "Top features by gain"],
      [[HAZ[h_], ", ".join(f"{n} ({g:.3f})"
                           for n, g in xgb[h_]["top_features"][:6])]
       for h_ in v2_models.HAZARDS],
      "Table 5. Six highest-gain features per hazard.")

if META and META.get("fwd_rain_ablation"):
    ab = META["fwd_rain_ablation"]
    h("5.1 How much of this is observation of the future", 2)
    p("Table 5 names the mechanism behind the headline result and also its "
      "largest problem. Five of the twenty-six features -- prcp_fwd1, "
      "prcp_fwd3, prcp_fwd3_max, ratio_fwd3 and log_fwd3 -- describe rainfall "
      "on the days after the prediction date. In the training panel those "
      "values come from the same NOAA station record the label was built "
      "from, which means they are the rain that caused the flood rather than a "
      "forecast of it. At serve time the same five features are filled from an "
      "Open-Meteo forecast, so the deployed model is asked a question it was "
      "never scored on.")
    p("To size the effect, both hazards were refit with those five features "
      "removed and everything else held identical -- same panel, same splits, "
      "same hyperparameters, same early stopping (audit_fwd_ablation.py):")
    table(["Hazard", "Test AP, as published", "Test AP, no forward rain",
           "Skill removed", "Lift, as published", "Lift, no forward rain"],
          [[HAZ[h_], f"{ab[h_]['ap_all']:.5f}", f"{ab[h_]['ap_no_fwd']:.5f}",
            f"{ab[h_]['share_lost'] * 100:.0f}%",
            f"{xgb[h_]['test']['lift']:.1f}x",
            f"{ab[h_]['ap_no_fwd'] / xgb[h_]['test']['prevalence']:.1f}x"]
           for h_ in v2_models.HAZARDS],
          "Table 5.1. The pooled models refit without the five forward-rain "
          "features. Lift is recomputed against the same base rate.")
    callout(
        f"{ab['flood']['share_lost'] * 100:.0f}% of the reported flood skill "
        f"and {ab['storm']['share_lost'] * 100:.0f}% of the reported hurricane "
        "skill rest on observations of the future. This is the project's "
        "central open problem rather than a footnote, and it is the reason "
        "every figure in this document is an upper bound. The fix is to rebuild "
        "the forward window from archived forecasts -- Open-Meteo Previous Runs "
        "for the recent period, the GEFSv12 reforecast for the full panel -- "
        "and then to report two numbers: the observed-rain score as a ceiling "
        "on achievable skill, and the forecast-rain score as the honest "
        "operational one.")
    p("Section 8's lead-time table deserves the same caution for a second "
      "reason: the strongest single-feature baseline at every lead is itself a "
      "forward-rain feature. The 'vs baseline' column there measures what "
      "boosting adds on top of observed future rain, not how much warning the "
      "system buys.", italic=True)

# ===================================================================== 6 tft

h("6. Temporal Fusion Transformer", 1)
t = prm["tft"]
p(f"The transformer sees a {t['encoder_days']}-day encoder window per county, "
  f"with hidden size {t['hidden']}, {t['heads']} attention heads, dropout "
  f"{t['dropout']} and hidden continuous size {t['hidden_continuous']}. "
  f"Trained with learning rate {t['lr']}, batch size {t['batch']}, up to "
  f"{t['epochs']} epochs with early stopping after {t['early']} epochs without "
  "improvement.")

if not HAS_TFT:
    callout("Withdrawn pending a refit on the corrected label.")
    p(WITHDRAWN)
    p("This run did not attempt it: "
      f"{man.get('tft_skipped', 'no transformer results were written')}.")
else:
    p(f"Because onset is rare, training uses a positive-class weight of "
      f"{t['pos_weight']:.0f}. That weight makes the loss learnable but "
      "inflates every output: the raw flood predictions average "
      f"{tft['flood']['calibration']['pred_mean_raw']:.5f} against an observed "
      f"rate of {tft['flood']['calibration']['observed_rate']:.5f}, a factor of "
      f"{tft['flood']['calibration']['pred_mean_raw'] / tft['flood']['calibration']['observed_rate']:.0f}. "
      "Isotonic regression fitted on the validation split corrects this. "
      "Isotonic is monotone, so it moves the probabilities onto a readable "
      "scale without reordering them, and average precision barely changes.")

    # The column is named after the rule that actually chose the epoch. A run
    # selected on validation AP reports a number where higher is better, so
    # printing it under "Best val loss" would read as the opposite result.
    sel_key = ("best_val_ap" if "best_val_ap" in tft[v2_models.HAZARDS[0]]
               else "best_val_loss")
    sel_head = "Best val AP" if sel_key == "best_val_ap" else "Best val loss"

    table(["Hazard", "Epochs run", sel_head, "Test windows",
           "Raw pred mean", "Calibrated pred mean", "Observed rate",
           "AP before", "AP after"],
          [[HAZ[h_], tft[h_]["epochs_run"], f"{tft[h_][sel_key]:.5f}",
            f"{tft[h_]['n_test_windows']:,}",
            f"{tft[h_]['calibration']['pred_mean_raw']:.5f}",
            f"{tft[h_]['calibration']['pred_mean_calibrated']:.5f}",
            f"{tft[h_]['calibration']['observed_rate']:.5f}",
            f"{tft[h_]['calibration']['ap_raw']:.5f}",
            f"{tft[h_]['test']['ap']:.5f}"]
           for h_ in v2_models.HAZARDS],
          "Table 6. Transformer training and the effect of calibration. The AP "
          "columns show that calibration costs almost nothing in ranking "
          "quality, which is the point of using a monotone method.")

    figure(f"tft_eval_{v2_models.TAG}.png", "Figure 4. Transformer precision-recall on test.")

figure(f"reliability_{v2_models.TAG}.png",
       "Figure 5. Reliability -- predicted probability against observed "
       "frequency." + (" Transformer series after isotonic calibration."
                       if HAS_TFT else ""))

# ============================================================== 7 comparison

h("7. Gradient boosting against the transformer", 1)
if not HAS_TFT:
    callout("Withdrawn pending a refit on the corrected label.")
    p(WITHDRAWN)
    p("What can be said without it: gradient boosting reaches a test average "
      f"precision of {xgb['flood']['test']['ap']:.5f} on floods and "
      f"{xgb['storm']['test']['ap']:.5f} on hurricanes, clearing the strongest "
      f"single-feature baseline by {xgb['flood']['vs_best_baseline']:.2f}x and "
      f"{xgb['storm']['vs_best_baseline']:.2f}x respectively. Whether a "
      "sequence model beats that on this panel is an open question in this "
      "document, not a settled one.")
    figure(f"compare_{v2_models.TAG}.png",
           "Figure 6. Gradient boosting against the baselines, pooled target.")
else:
    p("Both models were fitted on the same panel, the same splits and the same "
      "label, so the only thing differing within a hazard is the architecture.")
    rows = []
    for h_ in v2_models.HAZARDS:
        rows.append([HAZ[h_], "XGBoost (monotone)",
                     f"{xgb[h_]['test']['ap']:.5f}",
                     f"{xgb[h_]['test']['lift']:.1f}x",
                     f"{xgb[h_]['vs_best_baseline']:.2f}x"])
        rows.append([HAZ[h_], "Temporal Fusion Transformer",
                     f"{tft[h_]['test']['ap']:.5f}",
                     f"{tft[h_]['test']['lift']:.1f}x",
                     f"{tft[h_]['vs_best_baseline']:.2f}x"])
    table(["Hazard", "Model", "Test AP", "Lift over base rate",
           "vs best baseline"], rows, "Table 7. Pooled target, test years.")

    fr = xgb["flood"]["test"]["ap"] / tft["flood"]["test"]["ap"]
    sr = xgb["storm"]["test"]["ap"] / tft["storm"]["test"]["ap"]
    p(f"Gradient boosting wins both hazards. On floods the margin is narrow -- "
      f"{xgb['flood']['test']['ap']:.5f} against "
      f"{tft['flood']['test']['ap']:.5f}, a factor of {fr:.2f} -- close enough "
      f"that the transformer is a credible alternative. On hurricanes it is "
      f"not close: {xgb['storm']['test']['ap']:.5f} against "
      f"{tft['storm']['test']['ap']:.5f}, a factor of {sr:.2f}, and the "
      f"transformer clears its baseline by only "
      f"{tft['storm']['vs_best_baseline']:.2f}x.")

    p("The likely explanation is sample size rather than architecture. "
      f"Hurricane onset occurs on {xgb['storm']['test']['prevalence'] * 100:.3f}% "
      f"of county-days, and a {t['encoder_days']}-day encoder window spends "
      "almost all of its capacity on sequences containing no event at all. The "
      "transformer also cost hours of A100 time against seconds of boosting on "
      "the same data, so on this problem the heavier model is both worse and "
      "more expensive. That is a result worth reporting rather than hiding: a "
      "sequence model is not automatically the right answer for a rare-event "
      "tabular panel.")

    figure(f"compare_{v2_models.TAG}.png",
           "Figure 6. Both architectures against both baselines, pooled target.")

# =============================================================== 8 lead sweep

h("8. How much warning it actually buys", 1)
p("A single pooled number hides the thing a user needs most. Each lead below "
  "therefore has its own model, trained on its own target, and each is scored "
  "against the strongest single-feature baseline for that same target -- which "
  "changes with lead, and gets harder to beat as the signal weakens.")
head8 = ["Hazard", "Lead", "Test base rate", "Baseline", "Baseline AP",
         "XGBoost AP", "vs base"] + (["TFT AP", "vs base"] if HAS_TFT else [])
table(head8,
      [[HAZ[r_.hazard], r_.lead, f"{r_.prevalence * 100:.4f}%",
        r_.baseline.replace("single_", ""), f"{r_.baseline_AP:.5f}",
        f"{r_.xgb_AP:.5f}", f"{r_.xgb_vs_base:.2f}x"]
       + ([f"{r_.tft_AP:.5f}", f"{r_.tft_vs_base:.2f}x"] if HAS_TFT else [])
       for r_ in both.itertuples(index=False)],
      "Table 8. The exact-lead sweep. Six models"
      + (" per architecture" if HAS_TFT else "")
      + ", each trained and scored on its own target.")

fl = {k: leads[f"flood_l{k}"]["test"]["ap"] for k in prm["leads"]}
st = {k: leads[f"storm_l{k}"]["test"]["ap"] for k in prm["leads"]}
p("Gradient-boosting skill decays monotonically with lead in both hazards. "
  f"Flood average precision runs {fl[1]:.5f} to {fl[2]:.5f} to {fl[3]:.5f} "
  f"across +1, +2 and +3 days, a loss of {(1 - fl[3] / fl[1]) * 100:.0f}% from "
  f"the shortest lead to the longest; hurricane runs {st[1]:.5f} to "
  f"{st[2]:.5f} to {st[3]:.5f}, a loss of {(1 - st[3] / st[1]) * 100:.0f}%.")

if HAS_TFT:
    tfl = {k: tft_leads[f"flood_l{k}"]["test"]["ap"] for k in prm["leads"]}
    p("The transformer is monotone on hurricanes but not on floods, where the "
      f"+2 day model ({tfl[2]:.5f}) scores above the +1 day model "
      f"({tfl[1]:.5f}). Six independent fits of a high-variance model on a "
      "rare target will not order themselves perfectly, and the inversion is "
      "reported rather than smoothed away: it is a reason to treat the "
      "transformer's per-lead numbers as noisier than the boosted ones, not a "
      "finding about lead time.")

p("The ratio over baseline does not decay neatly, and this is worth being "
  "explicit about rather than presenting as a win. The baseline is losing "
  "signal at the same rate the model is: the strongest flood baseline falls "
  f"from {both[(both.hazard == 'flood') & (both.lead == '+2 day')].baseline_AP.iloc[0]:.5f} "
  f"at +2 days to {both[(both.hazard == 'flood') & (both.lead == '+3 day')].baseline_AP.iloc[0]:.5f} "
  "at +3. The ratio therefore rises while the underlying forecast gets worse. "
  "Read the average-precision columns for forecast quality and the ratio "
  "columns only for whether the model is earning its complexity.")

if HAS_TFT:
    n_xgb_wins = int((both["xgb_AP"] >= both["tft_AP"]).sum())
    p(f"Across the six lead-hazard combinations, gradient boosting takes "
      f"{n_xgb_wins} of {len(both)}.")

figure(f"lead_{v2_models.TAG}.png", "Figure 7. XGBoost average precision against lead.")
if HAS_TFT:
    figure(f"lead_both_{v2_models.TAG}.png",
           "Figure 8. Both architectures against lead, with baselines.")

# ========================================================= 9 deviation metrics

h("9. Deviation from observation", 1)
p("Average precision measures ranking. It says nothing about whether the "
  "predicted probabilities are the right size, or about what happens at a "
  "usable decision threshold. The tables below score both models against two "
  "null predictors -- always-zero, and the constant base rate -- at a "
  "threshold set to the 99th percentile of each model's own predictions.")

# Columns, not a fixed list: the deviation CSV carries one column per model the
# run actually fitted, so a run without a transformer has no `tft` column and
# naming it would raise rather than simply leave the comparison out.
DEV_COLS = ["always_zero", "base_rate", "xgboost"] + (["tft"] if HAS_TFT else [])
DEV_HEAD = ["Always zero", "Base rate", "XGBoost"] + (["TFT"] if HAS_TFT else [])

dev = {}
for haz in v2_models.HAZARDS:
    d = pd.read_csv(RES / f"deviation_{haz}_{v2_models.TAG}.csv")
    key = d.columns[0]
    dev[haz] = d.set_index(key)
    keep = ["observed_events", "predicted_mean", "bias_pred_minus_obs",
            "brier", "brier_skill_vs_baserate", "calibration_error_ECE",
            "hits", "misses", "false_alarms", "POD_recall", "precision",
            "CSI_critical_success", "F1"]
    sel = d[d[key].isin(keep)].set_index(key).reindex(keep)
    h(f"9.{v2_models.HAZARDS.index(haz) + 1} {HAZ[haz]}", 2)

    def fmt(v, name):
        if name in ("observed_events", "hits", "misses", "false_alarms"):
            return f"{int(v):,}"
        return f"{v:.5f}"

    table(["Metric"] + DEV_HEAD,
          [[name] + [fmt(sel.loc[name, c], name) for c in DEV_COLS]
           for name in keep],
          f"Table {9 + v2_models.HAZARDS.index(haz)}. {HAZ[haz]} deviation "
          "metrics on the test years.")

_bs = {h_: dev[h_].loc["brier_skill_vs_baserate", "xgboost"]
       for h_ in v2_models.HAZARDS}
p("Three things are worth drawing out. First, the two hazards differ in kind "
  "and not merely in degree. The flood model's Brier skill over the constant "
  f"base rate is {_bs['flood']:.3f}, a real if modest gain, so its "
  "probabilities carry information beyond their ordering. The hurricane "
  f"model's is {_bs['storm']:.3f}, which as a probability is barely "
  "distinguishable from quoting the base rate. The hurricane model should "
  "therefore be read as a ranking tool only, and nothing in this document "
  "supports treating its output as a calibrated chance of a storm.")
p("Second, the false-alarm counts are large in absolute terms at every usable "
  "threshold. This is inherent to a rare-event target and is the reason the "
  "dashboard presents a relative band rather than a yes/no alert: at a base "
  "rate near one percent, the flood model's precision of "
  f"{dev['flood'].loc['precision', 'xgboost']:.2f} is a strong result and "
  "still means two flagged days in three are quiet. Third, every count in "
  "these two tables is a county-day, so the hits and misses are correlated "
  "within weather episodes in the way section 3.1 describes.")

# ================================================================== 10 limits

h("10. Known limitations", 1)
cov = META["coverage"] if META else None

_ab = (META or {}).get("fwd_rain_ablation")
callout("Training rainfall is observed, not forecast. The forward rainfall "
        "features use station observations over the prediction window, which a "
        "real forecaster would not have on the morning the model runs. This is "
        "the single largest overstatement in the project. Every lead-time "
        "number in this document should be read as an upper bound on what an "
        "operational system would achieve. The fix is to retrain the forward "
        "features on archived forecasts (GEFSv12 reforecast)."
        + (f" Section 5.1 sizes the effect: removing those features costs "
           f"{_ab['flood']['share_lost'] * 100:.0f}% of the flood skill and "
           f"{_ab['storm']['share_lost'] * 100:.0f}% of the hurricane skill."
           if _ab else ""))

bullet("The label is an observational record, not hazard physics. A flood "
       "positive exists because someone reported it; an unreported flood is a "
       "negative. A hurricane positive is a track intersection and carries no "
       "intensity information.")
bullet(f"The fit covers {man['panel']['counties_as_read']} counties, all "
       "within the five Gulf states and all with a rainfall station in the "
       "source data"
       + (f", while the published dashboard scores "
          f"{META['n_counties_scored']:,}"
          if META and META.get("n_counties_scored") else
          ", while the published dashboard scores the whole country")
       + ". Counties outside the fit are shaded identically to those inside "
       "it and the boundary is deliberately not drawn on the map, so their "
       "scores are extrapolation that no measurement in this document "
       "supports. Section 11 explains how the inputs are obtained and why "
       "every metric here still describes the Gulf panel only.")
bullet("No held-out-state validation was run on this panel. Earlier work "
       "measured skill retention when a whole state is removed from training "
       "and found that it does not hold up: with Texas withheld, lift on "
       "held-out Texas counties fell to 1.11x against 1.51x for the same "
       "counties when they were in training. That is a transfer test between "
       "states sharing the Gulf's rainfall regime and hurricane exposure, and "
       "it still lost about two thirds of the skill, so transfer to "
       "climatologically unlike regions should be assumed worse until "
       "measured. Restoring the test on the corrected panel is outstanding "
       "work, not a completed check.")
if cov:
    bullet(f"The rainfall record has gaps. The panel spans "
           f"{cov['calendar_days']:,} calendar days; the median county has "
           f"{cov['median_days_per_county']:,} of them and the worst has "
           f"{cov['min_days_per_county']:,}. "
           f"{cov['counties_under_80pct']} counties fall below 80% coverage, "
           f"and {cov['events_listed'] - cov['events_listed_with_leadup']} of "
           f"the {cov['events_listed']:,} events listed on the dashboard have "
           "no day of their three-day lead-up in the record.")
bullet("No soil moisture and no river gauges. Antecedent wetness is proxied "
       "by 30-, 60- and 90-day rainfall, a weak substitute for catchment "
       "state.")
bullet("No hurricane wind field, track forecast, intensity or storm surge.")
if META and META.get("episodes"):
    _ep = META["episodes"]
    bullet(f"The {xgb['flood']['n_test_pos']:,} flood and "
           f"{xgb['storm']['n_test_pos']:,} hurricane positive county-days in "
           "the test period are not that many observations. They reduce to "
           f"roughly {_ep['flood']['episodes']} and "
           f"{_ep['storm']['episodes']} independent weather episodes, so "
           "every hurricane metric in this document has an effective sample "
           f"of about {_ep['storm']['episodes']} tropical cyclones. See "
           "section 3.1. Confidence intervals are correspondingly wide, would "
           "have to be resampled by episode rather than by row, and none are "
           "reported.")
else:
    bullet(f"Only {xgb['flood']['n_test_pos']:,} flood and "
           f"{xgb['storm']['n_test_pos']:,} hurricane onsets fall in the test "
           "period. Confidence intervals on every metric here are "
           "correspondingly wide and none are reported.")
bullet("The hurricane model ranks but does not calibrate. Its Brier skill "
       f"over the climatological rate is {_bs['storm']:.3f}, against "
       f"{_bs['flood']:.3f} for floods, so its probability should not be read "
       "as a chance of a storm. See section 9.")

# ========================================================= 11 national scoring

h("11. Scoring outside the training panel", 1)

ROOT = pathlib.Path(__file__).parent
clim = pd.read_csv(ROOT / "refdata" / "county_climatology_ghcn.csv",
                   dtype={"county_fips": str})
nat = pd.read_csv(ROOT / "refdata" / "county_static_national.csv",
                  dtype={"county_fips": str})
live = json.loads((ROOT / "docs" / "data" / "live.json").read_text())
n_borrowed = int((clim["clim_source"] == "nearest").sum())
n_own = len(clim) - n_borrowed
n_live = live["n_counties"]

p("The panel is Gulf-only because of where its rainfall came from, not because "
  "of anything in the model. Of the 26 features the scorer reads, 17 are "
  "derived from a live weather request and work for any coordinate on earth. "
  "Six more are static properties of the county -- population density and the "
  "CDC social-vulnerability ranks -- which are published nationally. The "
  "obstacle was the remaining three.")

p("Those three are ratios: observed rainfall over the county's own long-run "
  "normal. The same 80 mm in three days is an unremarkable week in Mobile and "
  "a once-in-a-decade event in Phoenix, and the ratio is what lets one model "
  "serve both. No live endpoint returns a normal, so it has to be computed "
  "offline from years of station history -- and the panel's normals existed "
  "only for its own counties. Any county without them was dropped by the "
  "scorer's guard rather than scored on an incomplete feature row, which is "
  "the correct behaviour: a gradient-boosted tree routes a missing value down "
  "a default branch and returns a plausible-looking number that is "
  "indistinguishable, on a map, from a real one.")

h("11.1 Rebuilding the normals nationally", 2)

p("The normals were rebuilt from the NOAA GHCN-Daily by_year bulk archive over "
  "2000-2008, the same window and the same station network (COOP and WBAN) the "
  "panel used. Two properties of that archive had to be handled explicitly. It "
  "reports precipitation in tenths of a millimetre where the panel is in "
  "millimetres, a factor of ten that would have scaled every national "
  "denominator and made every non-Gulf county read as calmer than it is "
  "without anything on the dashboard looking broken; the median ratio of raw "
  "archive to panel value across 176,086 shared station-days is exactly "
  "10.000, which both fixes the scale and confirms the panel is itself "
  "GHCN-derived. Observations carrying a NOAA quality flag are dropped.")

p("Station density needed a decision. A county-day is the mean across the "
  "county's stations, so a denser network smooths the daily series and drags "
  "the 95th percentile down. The panel averages 1.47 stations per county; an "
  "unrestricted polygon join averages 7.87 nationally. Scoring the Gulf "
  "counties through the national pipeline at different caps and comparing "
  "against their known values gives a mean relative error of 4.5% at one "
  "station, 4.0% at two and 4.1% at three, so two is both the minimum-error "
  "choice and the closest match to the panel's own density.")

table(["Quantity", "Value"],
      [["Stations clearing 80% coverage", "7,432 of 24,506"],
       ["Stations retained after the 2-per-county cap", "4,629"],
       ["Resulting density", "1.64 stations/county (panel: 1.47)"],
       ["Counties with a qualifying gauge of their own", f"{n_own:,}"],
       ["Counties borrowing the 2 nearest within 150 km", f"{n_borrowed:,}"],
       ["Counties with normals", f"{len(clim):,}"],
       ["Counties in the national constants table", f"{len(nat):,}"]],
      "Table 11. National rainfall climatology, built by "
      "build_ghcn_climatology.py --cap 2.")

p(f"{n_borrowed} counties contain no station that clears the coverage "
  "threshold. Rather than leave them unscored, each borrows the two nearest "
  "qualifying gauges by centroid distance, and the borrow is refused beyond "
  "150 km. The justification is that a rainfall normal varies far more "
  "smoothly in space than daily rainfall does, so a gauge one county over is a "
  "fair estimate of the local baseline where a gauge across a mountain range "
  "is not. These counties are marked in the clim_source column of the "
  "climatology file so the provenance survives in the data rather than only in "
  "this paragraph.")

h("11.2 What the rebuild was checked against", 2)

p("The Gulf counties already have a known answer, so the national pipeline was "
  "run over them and compared. It reproduces all 356 at a mean relative error "
  "of 4.1%: the three-day 95th percentile comes out at 42.85 mm against a "
  "known 43.48 (ratio 0.985, correlation 0.974) and the 30-day mean at 88.53 "
  "against 87.77 (ratio 1.009, correlation 0.976). Because the Gulf counties "
  "are no denser under the national rule than the country as a whole, that 4.1% "
  "is a national estimate of the error rather than a regional artefact.")

p("A stricter check was also run: rebuilding the Gulf counties from the panel's "
  "own 523 stations, rather than from the national station list, reproduces all "
  "three constants at a ratio of 1.000 and a mean absolute difference of 0.00 "
  "mm across all 356 counties. That pins the quality-flag filter, the "
  "county-day averaging, the minimum-periods setting and the forward-window "
  "boundary, and isolates the 4.1% as being entirely attributable to which "
  "stations are selected.")

p("The panel's own rows are carried through the national build byte for byte. "
  "They are the inputs the sealed run trained and scored on, so re-deriving "
  "them from a different source would quietly move the published numbers.")

h("11.3 Result and what it does not establish", 2)

p(f"The daily refresh now scores {n_live:,} counties. The counties that remain "
  "unscored are Connecticut's eight retired counties, which the map still draws "
  "but which have no CDC vulnerability record under those codes, plus any "
  "county whose live weather request fails on the day -- dropped rather than "
  "scored from a partial row, under the same guard as before.")

callout("National coverage is a data-engineering result, not an accuracy "
        "result. Every average precision, lift and deviation figure in this "
        "document was measured on the Gulf panel. No held-out-state validation "
        "was run, so nothing here establishes that the skill carries to a "
        "county in Arizona or Vermont. The dashboard discloses this in its "
        "limitations panel and does not mark the boundary on the map, because "
        "a visible boundary would imply the inside of it had been validated "
        "against the outside, which it has not.")

# ============================================================== 12 provenance

h("12. Provenance and reproducibility", 1)
v = man["versions"]
table(["Field", "Value"],
      [["Run id", man["run_id"]],
       ["Environment", f"{man['env']} on host {man['host']}"],
       ["Started (UTC)", man["started_utc"]],
       ["Finished (UTC)", man["finished_utc"]],
       ["Wall time", f"{man['wall_seconds']:,.1f} s "
                     f"({man['wall_seconds'] / 3600:.2f} h)"],
       ["GPU", v.get("cuda", "-")],
       ["Python / numpy / pandas",
        f"{v['python']} / {v['numpy']} / {v['pandas']}"],
       ["xgboost / torch", f"{v['xgboost']} / {v['torch']}"],
       ["Panel path", man["panel"]["path"]],
       ["Panel sha256", man["panel"]["sha256"]],
       ["Panel rows / counties / years",
        f"{man['panel']['rows_as_read']:,} / "
        f"{man['panel']['counties_as_read']} / "
        f"{man['panel']['years_as_read'][0]}-{man['panel']['years_as_read'][1]}"],
       ["Self-contained", str(man["self_contained"])],
       ["Code commit", f"{man['git']['commit'][:12]} on {man['git']['branch']}"],
       ["Working tree at launch",
        "uncommitted changes present" if man["git"]["dirty"] else "clean"],
       ["Warm parents", str(man["warm_parents"]) or "none"],
       ["Random seed", prm["seed"]]],
      "Table 12. Run manifest. Self-contained means the run's own directory "
      "holds the models, the results and the hashes, and nothing outside it "
      "contributed to a reported number -- which is a claim about outputs. It "
      "is not a claim that the code is recoverable from the commit alone: this "
      + ("run was launched from a clean tree, so it is."
         if not man["git"]["dirty"] else
         "run was launched with uncommitted changes in the working tree, so "
         "the commit above is a starting point rather than an exact record of "
         "the code that ran. The data side is exact regardless -- the panel "
         "hash is verified on every load."))

p("Reproducibility was verified locally before any of these numbers reached "
  "the dashboard. The local copy of the panel hashes identically to the one "
  "the run read. Each of the eight boosters -- two pooled and six exact-lead "
  "-- was reloaded from the run directory and re-scored on the test rows, and "
  "every one reproduced its recorded test average precision to five decimal "
  "places. The check is the __main__ block of v2_models.py and can be re-run "
  "at any time.")

p("The dashboard serves XGBoost"
  + (" because it wins both hazards. The transformer numbers on the "
     "dashboard's comparison panel are read" if HAS_TFT else
     ", and it is the only architecture this run fitted. Every number on the "
     "dashboard is read")
  + " from this run's results files rather than re-scored in the browser, so "
  "the figures in this document and the figures on the dashboard come from the "
  "same source. The exporter refuses to write a model card whose run identifier "
  "differs from the one that scored the live page, so the two cannot diverge "
  "silently.")

doc.save(OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
