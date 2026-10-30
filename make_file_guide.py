"""Generate FILE_GUIDE.docx: what every file in this project is, and why it
exists.

Written for someone opening the repository for the first time, or answering
"what is this file for?" in front of an audience. Deliberately plainer than
PROJECT_EXPLAINER.docx, which covers the method and the results instead.

Counts are read from the artifacts so the document cannot drift from the repo.
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor

HERE = Path(__file__).parent
METRICS = json.loads((HERE / "artifacts" / "proto_metrics.json").read_text())
META = json.loads((HERE / "docs" / "data" / "meta.json").read_text())

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(8)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, bold=False, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    return par


def _no_split(row):
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:cantSplit"))


def _repeat_header(row):
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:tblHeader"))


def filetable(rows):
    """One row per file: name in bold, then what it is and why it exists."""
    t = doc.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(("File", "What it is", "Why we need it")):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(head).bold = True
    _no_split(t.rows[0])
    _repeat_header(t.rows[0])
    for name, what, why in rows:
        r = t.add_row()
        run = r.cells[0].paragraphs[0].add_run(name)
        run.bold = True
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        r.cells[1].text = what
        r.cells[2].text = why
        _no_split(r)
    doc.add_paragraph()
    return t


def callout(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x8A, 0x3A, 0x00)


# ====================================================================== title

title = doc.add_heading("County Flood & Hurricane Risk", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("Guide to every file in the project — what it is, and why it exists").italic = True

# ======================================================================= read

h("How to read this guide", 1)
p(
    "This project is a chain, not a pile. Each script writes a file that the "
    "next script reads, and the last few write the files the web page loads. "
    "Nothing here is a framework and nothing is hidden: if you can follow the "
    "chain below, you can follow the whole project."
)
p(
    "The guide is grouped by the job a file does. Within the first group the "
    "order is the order you would actually run things."
)

h("The chain, in one line", 2)
p(
    "raw weather and disaster records  →  build_daily.py  →  a table of "
    "county-days  →  train_proto.py  →  two trained models  →  "
    "transfer_proto.py  →  honesty check  →  export_proto.py and "
    "stage4_geo.py  →  small files the browser can load  →  the dashboard."
)

callout(
    "If you only remember one thing: build_daily.py makes the data, "
    "train_proto.py makes the models, transfer_proto.py checks whether they "
    "are trustworthy somewhere new, and the docs/ folder is the website."
)

# =================================================================== pipeline

h("1. The pipeline — the files that do the work", 1)
p(
    "Run these in the order shown. Each one depends on the output of the one "
    "before it."
)

filetable([
    (
        "config.py",
        "One small file holding settings everything else reads: where the data "
        "lives, which years count as training versus testing, and which FEMA "
        "incident types count as a storm.",
        "So a setting is changed in one place instead of twelve. It also "
        "freezes the 'normal rainfall' reference period to 1990–2010. If that "
        "window were widened to cover the test years, the model would be "
        "judging recent years against a normal that already included them — "
        "which quietly inflates every score.",
    ),
    (
        "build_daily.py",
        f"Builds the main data table: one row per county per day, "
        f"{METRICS['n_rows']:,} rows in total across "
        f"{METRICS['n_counties']} counties. Each row holds that county's "
        "rainfall history, the rain forecast for the next three days, the "
        "season, and whether a federal disaster actually began there.",
        "This is the dataset. Everything downstream trains on it or scores "
        "against it. It also fixes the question being asked: does a disaster "
        "BEGIN in the next three days. The easier question — is a disaster "
        "happening right now — can be answered by copying yesterday's answer, "
        "so it would score well and mean nothing.",
    ),
    (
        "proto_models.py",
        "Defines the two models and, separately, the one piece of code that "
        "knows how to run them to get a prediction.",
        "Four different scripts need to produce predictions: training, the "
        "transfer check, the live weather fetch, and the website export. If "
        "each had its own copy of the scoring logic, they would eventually "
        "disagree and nobody would notice. Here there is exactly one copy.",
    ),
    (
        "train_proto.py",
        "Trains both models and writes down how well they did, next to how "
        "well several deliberately simple methods did.",
        "The simple comparisons are the point. Without them a score is just a "
        "number with nothing to measure it against. This file is what proved "
        "the complicated flood model was worse than a single rainfall "
        "measurement — so the simple one was kept.",
    ),
    (
        "transfer_proto.py",
        "Removes one state from training completely, retrains, and tests on "
        "that state. Repeats for all five. Also scores each test year "
        "separately.",
        "The strongest result in the project, and the only one that answers "
        "the question an audience actually asks: does this work anywhere it "
        "has not already seen? Testing on places the model trained on measures "
        "memory. This measures skill.",
    ),
    (
        "export_proto.py",
        "Writes the three data files the web page loads: county information, "
        "past disasters, and model risk by date.",
        "The full results are around two million rows — far too much to send "
        "to a browser. This picks out only the dates a visitor will actually "
        "look at: the windows around real disasters, plus the most recent two "
        "months. That is what makes a free, server-less website possible.",
    ),
    (
        "stage4_geo.py",
        f"Downloads public map outlines and converts them into drawing "
        f"instructions for all {META['n_counties_mapped']:,} counties in the "
        "lower 48 states.",
        "Doing the map maths here, once, in Python, means the website needs no "
        "mapping library at all — it just draws the shapes it is given. It "
        "also works out a point inside each county to ask for weather, which "
        "matters for coastal counties where the obvious centre point lands in "
        "the sea.",
    ),
    (
        "fetch_live.py",
        "Fetches real weather for each county from a free public service and "
        "scores today plus the next three days.",
        "This is what makes the dashboard current rather than historical. The "
        "weather service returns the past three months alongside the forecast "
        "in a single request, which conveniently supplies both the 'how wet "
        "has it been' and 'how wet will it be' numbers the models need.",
    ),
    (
        "serve.py",
        "A small local web server for viewing the dashboard during "
        "development.",
        "Browsers aggressively remember files. This one tells the browser not "
        "to, so a change to the page shows up immediately instead of hours "
        "later. Not used by the public site.",
    ),
])

# ================================================================== dashboard

h("2. The dashboard — the files that are the website", 1)
p(
    "Everything in the docs/ folder is published directly as the live site. "
    "That is a GitHub convention, not a choice we made."
)

filetable([
    (
        "docs/index.html",
        "The structure of the page: the controls at the top, the map, the "
        "county panel on the right, and the model explanation underneath.",
        "The skeleton. It contains no data and no logic — only the named "
        "slots that the code fills in.",
    ),
    (
        "docs/app.js",
        "All of the behaviour: drawing the map, responding to clicks, "
        "colouring counties by risk, fetching live weather and official "
        "warnings, and listing past disasters.",
        "This is the application. It is plain JavaScript with no framework, so "
        "it will still run unchanged in several years.",
    ),
    (
        "docs/styles.css",
        "All of the appearance, including both the dark and light colour "
        "themes.",
        "Every colour the map uses is defined once here as a named value, so "
        "switching theme simply recolours the existing map instead of "
        "rebuilding it.",
    ),
    (
        "docs/data/geo.json",
        "The map itself: county shapes, state outlines, names, and a weather "
        "query point for each county.",
        "Produced by stage4_geo.py. Without it there is no map.",
    ),
    (
        "docs/data/meta.json",
        "Model scores, the transfer results, the thresholds for the risk "
        "labels, and the list of every county with a flag for whether the "
        "model covers it.",
        "Lets the page tell the difference between 'this county is outside the "
        "modelled area' and 'this county is covered but has no data for the "
        "date you picked'. Those are very different statements and conflating "
        "them would be misleading.",
    ),
    (
        "docs/data/history.json",
        "Model risk for each county on the dates the site covers.",
        "What the map colours in when you look at a past event.",
    ),
    (
        "docs/data/events.json",
        "Real federal disaster declarations for each county.",
        "The clickable history. Choosing one jumps the map to that date, so "
        "you can see what the model would have said before it happened. This "
        "is the most persuasive thing on the site because the outcome is "
        "already known.",
    ),
    (
        "docs/data/live.json",
        "Today and the next three days, scored from the live forecast.",
        "What the map colours in by default. Rebuilt automatically every day.",
    ),
    (
        ".github/workflows/refresh-live.yml",
        "An automatic job that re-runs the live weather fetch once a day and "
        "saves the result.",
        "Without it the 'live' view would quietly freeze on whatever day it "
        "was last run by hand, while still looking current. That would be "
        "worse than having no live view at all.",
    ),
])

# ================================================================== artifacts

h("3. Data and saved models", 1)

filetable([
    (
        "refdata/station_county_mapping.csv",
        "A lookup table matching each weather station to its county, with "
        "coordinates.",
        "Small and derived, so it is included in the repository. The raw "
        "rainfall records it was built from are far too large to publish and "
        "are not ours to redistribute.",
    ),
    (
        "artifacts/proto_flood.json",
        "The saved flood model — a scaling factor, one weight and an offset.",
        "Small enough to read with your own eyes and check. That is a direct "
        "consequence of the model being simple, and it is a genuine advantage.",
    ),
    (
        "artifacts/proto_storm.json",
        "The saved hurricane model.",
        "Loaded by everything that produces a hurricane prediction.",
    ),
    (
        "artifacts/proto_metrics.json",
        "Every score from the training run, including all the simple "
        "comparison methods.",
        "The website reads its numbers from this file rather than having them "
        "typed into the page. It is therefore impossible for the website to "
        "claim a score the training run did not produce.",
    ),
    (
        "artifacts/proto_transfer.json",
        "The per-state and per-year results from the transfer check.",
        "Same reason as above: the headline '77% retained' figure on the site "
        "is read from this file, not retyped.",
    ),
    (
        "requirements.txt",
        "The list of Python packages needed.",
        "So someone else can install the right versions in one command.",
    ),
    (
        ".gitignore",
        "A list of files deliberately excluded from the repository.",
        "Keeps the large raw data files out. Everything the website and the "
        "daily refresh need is still included, so a fresh copy works "
        "immediately.",
    ),
])

# =================================================================== document

h("4. Documentation", 1)

filetable([
    (
        "README.md",
        "The front page of the repository.",
        "The first thing anyone sees on the code site. Covers the question, "
        "the results, and this same file list in shorter form.",
    ),
    (
        "PROJECT_EXPLAINER.docx",
        "A plain-language walkthrough of the method, the results, the limits, "
        "and the questions an audience tends to ask.",
        "For presenting the project without having built it. Every number in "
        "it is pulled from the saved results, so it cannot disagree with them.",
    ),
    (
        "make_explainer.py",
        "The script that generates the document above.",
        "Generating it rather than writing it by hand is exactly what makes "
        "the 'cannot disagree' guarantee true.",
    ),
    (
        "FILE_GUIDE.docx",
        "This document.",
        "So the question 'what is this file for?' has a written answer.",
    ),
    (
        "make_file_guide.py",
        "The script that generates this document.",
        "Same reasoning: the file counts quoted here are read from the project "
        "rather than typed, so they stay correct.",
    ),
])

# ====================================================================== older

h("5. Files from the earlier approach", 1)
p(
    "An earlier version of this project asked a different question: given that "
    "a county had just had its heaviest rainfall in a hundred days, would it be "
    "declared a disaster area within two weeks? That design was set aside "
    "because it worked on monthly data, and monthly data cannot support a "
    "dashboard that answers 'what about today'."
)
p(
    "These files are kept so the path from the old approach to the current one "
    "can be inspected. They are not part of the current pipeline and are not "
    "needed to reproduce anything on the live site.",
    italic=True,
)

filetable([
    (
        "build_panel.py",
        "Built the monthly table the earlier model used.",
        "Kept as the record of what the data looked like before daily.",
    ),
    (
        "stage2_feasibility.py",
        "A counting exercise run before any modelling, to see whether there "
        "were enough real disasters to learn from.",
        "The most useful file in this group. It is the reason the original "
        "design was abandoned — roughly 126 flood disasters across 115 "
        "distinct dates is simply not enough. Counting first and modelling "
        "second saved a great deal of wasted work.",
    ),
    (
        "stage2.py",
        "The earlier model: given a severe rainfall day, which counties were "
        "declared?",
        "The previous generation of the current flood model.",
    ),
    (
        "stage3.py",
        "Converted that model's raw output into honest percentages and tested "
        "it state by state.",
        "The earlier version of today's transfer check.",
    ),
    (
        "stage4_export.py",
        "Froze that model into a standalone file of plain arithmetic and "
        "verified it reproduced the original exactly.",
        "Showed that the website and the analysis could not drift apart.",
    ),
    (
        "stage4_live.py",
        "The earlier live-weather path, using different public data sources.",
        "Superseded by fetch_live.py, which needs only one service.",
    ),
    (
        "baselines.py",
        "Simple comparison methods, including 'how often does this county get "
        "a disaster in this month of the year'.",
        "That comparison is not a strawman. Knowing that Louisiana in "
        "September is dangerous is most of what a risk model is asked to say, "
        "so any real model has to beat it.",
    ),
    (
        "daily_model.py, train_xgb.py",
        "Early experiments on daily rainfall.",
        "Where the current daily design came from.",
    ),
    (
        "tft_colab.py, gen_notebook.py",
        "An experiment with a Temporal Fusion Transformer, a neural network "
        "for time series, plus the script that generates its notebook.",
        "Predicts the range of possible rainfall rather than a yes/no answer. "
        "Kept as the comparison point for whether a larger model is worth it.",
    ),
    (
        "stage.ipynb, stage1-complete.ipynb",
        "Exploratory notebooks.",
        "The original working-out.",
    ),
    (
        "artifacts/stage2, stage3, stage4",
        "Saved results from the files above.",
        "Evidence that those numbers were produced, not asserted.",
    ),
    (
        "docs/data/model.json, replay.json",
        "Data files the earlier dashboard loaded.",
        "No longer read by the current page.",
    ),
])

# ======================================================================= tail

h("Two things worth knowing", 1)
p(
    "First, the raw weather and disaster source tables are not in this "
    "repository. They are large, and they belong to the agencies that publish "
    "them. Only the first script needs them; everything after that runs from "
    "files that are included."
)
p(
    "Second, nothing in the pipeline requires an API key, a paid service or a "
    "login. The weather and official warnings the dashboard shows are fetched "
    "live from free public services at the moment you click a county."
)

doc.core_properties.author = "Ganesh Pinnu"
doc.core_properties.last_modified_by = "Ganesh Pinnu"

OUT = HERE / "FILE_GUIDE.docx"
doc.save(OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
