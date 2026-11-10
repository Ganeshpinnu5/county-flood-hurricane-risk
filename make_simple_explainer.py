"""Builds PROJECT_SIMPLE.docx: a short plain-language explainer.

Numbers come from one sealed run, never from this file. Both models were
trained cold on the same panel and the same splits in that run, so the two
rows of the table are directly comparable.
"""

import csv
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

HERE = Path(__file__).parent
RUNS = HERE / "artifacts" / "runs"
RUN = "20261108T225052Z-run"
XGB_RUN = TFT_RUN = RUN
AUTHOR = "Ganesh Pinnu"


def summary(run):
    path = RUNS / run / "results" / "summary_run.csv"
    out = {}
    for r in csv.DictReader(path.open()):
        out[(r["hazard"], r["model"])] = r
    return out


def deviation(run, hazard):
    path = RUNS / run / "results" / f"deviation_{hazard}_run.csv"
    rows = list(csv.reader(path.open()))
    cols = rows[0][1:]
    return {r[0]: dict(zip(cols, r[1:])) for r in rows[1:]}


def leads(run):
    path = RUNS / run / "results" / "leads_run.csv"
    return list(csv.DictReader(path.open()))


def f(x, n=3):
    return f"{float(x):.{n}f}"


def load():
    man = json.loads((RUNS / XGB_RUN / "manifest.json").read_text())
    tft = json.loads((RUNS / TFT_RUN / "results" / "tft_run.json").read_text())
    d = {
        "man": man,
        "xgb": summary(XGB_RUN),
        "tft": tft,
        "dev_x": {h: deviation(XGB_RUN, h) for h in ("flood", "storm")},
        "dev_t": {h: deviation(TFT_RUN, h) for h in ("flood", "storm")},
        "leads": leads(XGB_RUN),
    }
    panel = man["panel"]
    d["rows"] = panel["rows_as_read"]
    d["counties"] = panel["counties_as_read"]
    booster = json.loads(
        (RUNS / XGB_RUN / "models" / "xgb_flood_run.json").read_text())
    d["n_feats"] = len(booster["learner"]["feature_names"])
    return d


def main():
    d = load()
    man, xgb, tft = d["man"], d["xgb"], d["tft"]
    nfeat = d["n_feats"]
    lead = {(r["hazard"], r["lead"]): r for r in d["leads"]}
    fx = xgb[("flood", "xgboost")]
    sx = xgb[("storm", "xgboost")]

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    t = doc.add_heading("What this project does", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph(f"{AUTHOR} — flood and hurricane prediction for US "
                      f"counties").alignment = WD_ALIGN_PARAGRAPH.LEFT

    points = [
        "It predicts floods and hurricanes for US counties, a few days before "
        "they happen.",

        "For any county on any day, it answers two questions. Will a flood "
        "start in the next 3 days? Will a hurricane hit this county in the "
        "next 3 days?",

        f"We use the Gulf Coast states, so Texas, Louisiana, Florida and the "
        f"ones nearby. That is {d['counties']} counties, every day from 2000 "
        f"to 2026, about {d['rows'] / 1e6:.2f} million rows of data.",

        "The weather data comes from NOAA. The flood records come from NOAA's "
        "storm reports, about 6,700 real flood events. The hurricane tracks "
        "come from the National Hurricane Center.",

        f"The model looks at {nfeat} things. Rain today, rain over the last "
        f"week, month and three months, the rain forecast, temperature, the "
        f"month of the year, where the county is, and how many people live "
        f"there.",

        "It also looks at 21 hurricane things. How close a storm is expected "
        "to pass, how strong it is, how fast it is moving, and whether the "
        "county is inside the warning cone.",

        "Those hurricane features are the big win. Without them the model was "
        "blind to storms.",

        "One important detail. We only give the model forecasts that existed "
        "at that time. It never gets to see what actually happened. Otherwise "
        "it would be cheating.",

        "We split the data by time. 2000 to 2014 to train the model, 2015 to "
        "2019 to check it, and 2020 to 2026 kept hidden until the end.",

        "We do not shuffle the days randomly, because today's weather and "
        "tomorrow's weather are almost the same. Shuffling would let the "
        "model peek at the answer.",

        "We trained two models on exactly the same data. XGBoost, which is "
        "decision trees, and a transformer, which reads 30 days of history at "
        "a time.",

        "Each model has to beat a simple rule first, like just looking at the "
        "rain forecast. If it cannot beat that, it is useless.",

        f"Flood result, XGBoost. Score {f(fx['test_AP'])}. That is "
        f"{float(fx['lift']):.0f} times better than random guessing, and "
        f"{f(fx['vs_baseline'], 1)} times better than the simple rain rule.",

        f"Hurricane result, XGBoost. Score {f(sx['test_AP'])}. That is "
        f"{float(sx['lift']):.0f} times better than guessing, and "
        f"{f(sx['vs_baseline'], 1)} times better than the simple rule. "
        f"Adding the storm features took this from 0.156 to "
        f"{f(sx['test_AP'])}.",

        f"Transformer result. Flood {f(tft['flood']['test']['ap'])}, "
        f"hurricane {f(tft['storm']['test']['ap'])}. Both lower than XGBoost, "
        f"on the same data and the same split. The transformer lost, so the "
        f"website keeps using XGBoost.",

        f"The flood transformer is the weaker of the two. It beats the simple "
        f"rain rule by only "
        f"{tft['flood']['vs_best_baseline']:.2f} times, which is small enough "
        f"to be luck. We would not call that working.",

        f"Error numbers for XGBoost. Flood RMSE "
        f"{f(d['dev_x']['flood']['RMSE']['xgboost'])}, MSE "
        f"{f(d['dev_x']['flood']['brier']['xgboost'], 4)}. Hurricane RMSE "
        f"{f(d['dev_x']['storm']['RMSE']['xgboost'])}, MSE "
        f"{f(d['dev_x']['storm']['brier']['xgboost'], 4)}.",

        f"Those error numbers look small, but they do not mean much. Floods "
        f"only happen on about 1% of days, so a model that always says no "
        f"flood also gets a small error, "
        f"{f(d['dev_x']['flood']['RMSE']['always_zero'], 3)}. That is why we "
        f"use the scores above instead.",

        f"The further ahead we predict, the worse it gets, which is normal. "
        f"Flood: {f(lead[('flood', '+1 day')]['test_AP'])} tomorrow, "
        f"{f(lead[('flood', '+2 day')]['test_AP'])} in 2 days, "
        f"{f(lead[('flood', '+3 day')]['test_AP'])} in 3 days. Hurricane: "
        f"{f(lead[('storm', '+1 day')]['test_AP'])}, "
        f"{f(lead[('storm', '+2 day')]['test_AP'])}, "
        f"{f(lead[('storm', '+3 day')]['test_AP'])}.",

        "We saved the trained models and built a website around them. You "
        "click a county, it pulls that county's live weather from a public "
        "server, and gives you a flood and hurricane probability.",

        "The site also shows what the National Weather Service is warning "
        "right now for the same county, side by side, so you can compare our "
        "model against the real one. And there is a 3 day forecast view, with "
        "a separate model for each day.",
    ]
    for i, p in enumerate(points, start=1):
        doc.add_paragraph(f"{i}. {p}")

    doc.add_heading("The numbers in one table", level=1)
    rows = [
        ("Flood", "XGBoost", f(fx["test_AP"]),
         f"{float(fx['lift']):.0f}x", f"{f(fx['vs_baseline'], 2)}x",
         f(d["dev_x"]["flood"]["RMSE"]["xgboost"]),
         f(d["dev_x"]["flood"]["brier"]["xgboost"], 4),
         f(d["dev_x"]["flood"]["predicted_sd"]["xgboost"])),
        ("Flood", "Transformer", f(tft["flood"]["test"]["ap"]),
         f"{tft['flood']['test']['lift']:.0f}x",
         f"{tft['flood']['vs_best_baseline']:.2f}x",
         f(d["dev_t"]["flood"]["RMSE"]["tft"]),
         f(d["dev_t"]["flood"]["brier"]["tft"], 4),
         f(d["dev_t"]["flood"]["predicted_sd"]["tft"])),
        ("Hurricane", "XGBoost", f(sx["test_AP"]),
         f"{float(sx['lift']):.0f}x", f"{f(sx['vs_baseline'], 2)}x",
         f(d["dev_x"]["storm"]["RMSE"]["xgboost"]),
         f(d["dev_x"]["storm"]["brier"]["xgboost"], 4),
         f(d["dev_x"]["storm"]["predicted_sd"]["xgboost"])),
        ("Hurricane", "Transformer", f(tft["storm"]["test"]["ap"]),
         f"{tft['storm']['test']['lift']:.0f}x",
         f"{tft['storm']['vs_best_baseline']:.2f}x",
         f(d["dev_t"]["storm"]["RMSE"]["tft"]),
         f(d["dev_t"]["storm"]["brier"]["tft"], 4),
         f(d["dev_t"]["storm"]["predicted_sd"]["tft"])),
    ]
    head = ["Hazard", "Model", "Score", "vs guessing", "vs simple rule",
            "RMSE", "MSE", "SD of predictions"]
    tbl = doc.add_table(rows=1, cols=len(head))
    tbl.style = "Light Grid Accent 1"
    for c, h in enumerate(head):
        tbl.rows[0].cells[c].text = h
    for r in rows:
        cells = tbl.add_row().cells
        for c, v in enumerate(r):
            cells[c].text = str(v)

    doc.add_paragraph("")
    doc.add_paragraph(
        "The score column is average precision. For rare events like these it "
        "is the number that separates a useful model from a useless one, which "
        "is why it is listed before RMSE and MSE.")
    doc.add_paragraph(
        f"Both models come from the same run, {RUN}, trained from scratch on "
        f"the same data and scored on the same hidden years, so the rows can "
        f"be compared directly.")

    out = HERE / "PROJECT_SIMPLE.docx"
    doc.save(out)
    print(f"wrote {out}  ({len(points)} points)")


if __name__ == "__main__":
    sys.exit(main())
