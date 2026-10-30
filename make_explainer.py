"""Generate PROJECT_EXPLAINER.docx, the hand-off document for someone who has
to present this project without having built it. Written for a reader with no
statistics background: every technical term is either avoided or explained in
words where it appears, and the glossary at the back covers the rest. Numbers
are pulled from the artifacts rather than retyped, so the document cannot drift
from the results."""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor

HERE = Path(__file__).parent
M = json.loads((HERE / "docs" / "data" / "model.json").read_text())

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


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def numbered(text):
    doc.add_paragraph(text, style="List Number")


def _no_split(row):
    # Without this a row can be torn across a page break, which renders as a
    # stray empty band at the top of the next page.
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:cantSplit"))


def _repeat_header(row):
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:tblHeader"))


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(head))
        run.bold = True
    _no_split(t.rows[0])
    _repeat_header(t.rows[0])
    for row in rows:
        r = t.add_row()
        for i, val in enumerate(row):
            r.cells[i].text = str(val)
        _no_split(r)
    doc.add_paragraph()
    return t


def callout(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x8A, 0x3A, 0x00)


# ====================================================================== title

title = doc.add_heading("Gulf Coast Disaster Declaration Risk", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run(
    "A risk ranking tool and web dashboard for 218 counties on the Gulf Coast\n"
    "Plain-language explainer — what it does, how it was built, and what it "
    "cannot do")
run.italic = True

links = doc.add_paragraph()
links.alignment = WD_ALIGN_PARAGRAPH.CENTER
links.add_run("Dashboard: https://ganeshpinnu5.github.io/county-flood-hurricane-risk/\n"
              "Code: https://github.com/Ganeshpinnu5/county-flood-hurricane-risk")

note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = note.add_run(
    "This document assumes no background in statistics or machine learning. "
    "Technical words are explained where they first appear, and there is a "
    "glossary in section 14.")
run.italic = True
run.font.size = Pt(10)

doc.add_page_break()

# =================================================================== 1 summary

h("1. The whole thing in one paragraph", 1)
p("A huge amount of rain falls on a county somewhere on the Gulf Coast — more "
  "in three days than that county normally sees, the kind of downpour it gets "
  "roughly once in a hundred rainy spells. This project answers one question: "
  "what are the chances the federal government declares a disaster there "
  "within the next two weeks? That is all it does. It does not try to predict "
  "the weather. It waits until the rain has fallen, or until the weather "
  "service says it is coming, and then estimates the chance of an official "
  "government response. It covers 218 counties across Florida, Alabama, "
  "Mississippi, Louisiana and Texas. It learned its patterns from the years "
  f"{M['trained_on']}, was fine-tuned on {M['calibrated_on']}, and was then "
  f"tested on {M['evaluated_on']} — years it had never been shown. On those "
  "unseen years it is about "
  f"{M['test_ap'] / M['test_prevalence']:.1f} times better than guessing. The "
  "whole dashboard runs inside a web browser from a single small file, with "
  "no server behind it.")

h("What to say in the first thirty seconds", 2)
p("“This is a tool that tells you where to look first. It is not a warning "
  "system. Once extreme rain hits a stretch of the Gulf Coast, dozens of "
  "counties are affected and someone has to decide which ones to examine "
  "first. This ranks them. The highest chance it has ever reported for any "
  f"single county is {M['max_observed_probability']:.1%}, so it is never "
  "shouting 'disaster here'. It is roughly twice as good as guessing, and it "
  "tells you honestly where it does not work.”")

callout("The most common misunderstanding: people assume a weather app and "
        "this dashboard should agree with each other. They are measuring "
        "completely different things. A weather app tells you the chance of "
        "rain. This tells you the chance of a federal disaster declaration, "
        "and only for places that have already been hit with extreme rain. "
        "Section 12 has a full answer for this question.")

# ================================================================= 2 not a ...

h("2. What this is not", 1)
p("Being clear about the limits is the strongest part of this project. Say "
  "these four things before anyone has a chance to ask.")

bullet("It is not a weather forecaster. It does not predict rain. It takes "
       "the National Weather Service's rain forecast and uses it as an "
       "ingredient. The weather service does the forecasting; this does "
       "something different with the result.")
bullet("It is not a warning system. The highest chance it has ever produced "
       f"for any county is {M['max_observed_probability']:.1%}. That is "
       "useful for deciding which county to examine first. It is useless for "
       "deciding whether anyone should evacuate, and it should never be used "
       "that way.")
bullet("It only speaks up after extreme rain. A county does not even appear "
       "in the data until it has passed its own extreme-rainfall mark. Below "
       "that line the tool is not saying “low risk” — it is saying nothing "
       "at all, because it was never built to have an opinion there.")
bullet("It finds patterns, not causes. When the tool says heavy rain in the "
       "previous month goes together with a higher chance of a declaration, "
       "that is a pattern it noticed in history. It is not proof that one "
       "thing causes the other.")

# ===================================================================== 3 data

h("3. Where the data comes from", 1)
p("Four sources, all free and public. Nothing was bought, and nothing was "
  "scraped off a website without permission.")

table(
    ["Source", "What it gives us", "What it is used for"],
    [["NOAA weather station records (GHCN-Daily)",
      "How much rain fell each day, measured at 368 real weather stations",
      "The rainfall history, and the recent rainfall shown on the live page"],
     ["FEMA disaster declaration records",
      "Every federal disaster declaration, with its date, county and type",
      "The thing we are trying to predict"],
     ["CDC social vulnerability index",
      "Four scores per county describing poverty, housing, transport and "
      "similar social factors",
      "Tested as possible ingredients, and then dropped — see section 7"],
     ["National Weather Service forecast feed",
      "How much rain is expected, roughly a week ahead",
      "The forward-looking ingredient on the live page"]])

h("Turning station readings into county numbers", 2)
p("Rain is measured at individual weather stations, but declarations happen "
  "to whole counties. So the readings from all the stations inside a county "
  "are averaged together to give that county one rainfall number per day. "
  "368 stations cover the 218 counties. If a county has no station reporting "
  "on a given day, it is recorded as zero rain. That sounds crude, but it is "
  "deliberate: the live system has to make the same assumption, and it is "
  "important that the tool behaves identically when it is being tested and "
  "when it is being used for real.")
p("The tool also needs to know what counts as a normal amount of rain for "
  "each county at each time of year. That “normal” was worked out using only "
  "the years 1990 to 2010, and then locked. This is not a small detail. If "
  "the recent years had been allowed to help define what normal looks like, "
  "the tool would effectively be peeking at the answers before being tested, "
  "and every result in this document would look better than it really is.")
p("Some FEMA records are thrown out on purpose. Declarations tagged "
  "“Biological” are almost all COVID-19, and declarations tagged “Human "
  "Cause” or “Toxic Substances” have nothing to do with weather.")

h("How the years were divided up", 2)
p("The years are split into three blocks that run in order, never shuffled "
  "at random. This matters. Tuesday's weather looks a lot like Monday's, so "
  "if the days were shuffled the tool would see Monday during practice and "
  "then be tested on Tuesday, which is close to letting it see the answers. "
  "Keeping the blocks in chronological order avoids that.")
table(["Block", "Years", "What it is for"],
      [["Practice", M["trained_on"],
        "The tool learns its patterns here"],
       ["Fine-tuning", M["calibrated_on"],
        "Used only to turn the tool's scores into honest percentages"],
       ["Final test", M["evaluated_on"],
        "Never touched until the very end. 5,260 county-days, covering 76 "
        "separate real disaster episodes"]])

# ================================================================== 4 stage 1

h("4. The first attempt, and why it failed", 1)
p("The project started the obvious way: try to predict the disaster itself, "
  "months ahead. A dataset was built with rainfall, how unusual that rainfall "
  "was, and the time of year. Two different prediction methods were trained "
  "on it — XGBoost, a widely used method that builds thousands of small "
  "decision rules and combines them, and a Temporal Fusion Transformer, a "
  "modern neural network designed for data that unfolds over time. Each was "
  "asked to predict floods and storms at zero, one, two and three months "
  "ahead.")
p("Before believing any result, two deliberately dumb benchmarks were run "
  "first. The first simply asks: historically, how often does this county "
  "get a disaster in this month of the year? The second just assumes "
  "tomorrow looks like today. These sound trivial, but the first one already "
  "knows that Louisiana in September is dangerous, which is most of what "
  "anyone expects a county risk tool to tell them. Any real method has to "
  "beat that or it has earned nothing.")

p("The honest result:", bold=True)
table(["What we tried to predict", "How far ahead", "Score",
       "Times better than guessing", "Ranking ability"],
      [["Flood", "Same month", "0.043", "7.41x", "0.759"],
       ["Flood", "2 months ahead", "0.008", "1.55x", "0.582"],
       ["Flood", "3 months ahead", "0.010", "1.91x", "0.459"],
       ["Storm", "Same month", "0.203", "5.78x", "0.828"],
       ["Storm", "3 months ahead", "0.130", "3.67x", "0.793"]])

p("Read the last column. “Ranking ability” runs from 0.5 to 1.0, where 0.5 "
  "means a coin flip and 1.0 means perfect. Predicting floods in the same "
  "month scores 0.759, which is genuinely useful — but that is cheating in a "
  "sense, because the rain has already fallen and the method is largely just "
  "reading it. Three months ahead, the flood score is 0.459. That is below a "
  "coin flip. The method was worse than useless at that range.")
p("This is the result that changed the whole project. Predicting whether "
  "extreme rain will arrive months from now is a weather forecasting problem, "
  "and the National Weather Service has satellites, supercomputers and "
  "decades of science devoted to it. A student project is not going to beat "
  "them, and pretending otherwise would have produced a dishonest paper.")

callout("This is a good story to tell, not something to hide. The project "
        "changed direction because an experiment gave a negative answer and "
        "nobody buried it. That is how research is supposed to work, and an "
        "audience will respect it.")

# ================================================================== 5 pivot

h("5. The change of direction", 1)
p("If forecasting the rain is the wrong job, then the right question is what "
  "happens after the rain. Rewriting the problem that way fixes three things "
  "at once:")
numbered("It stops competing with the National Weather Service. Their "
         "forecast becomes an ingredient instead of a rival.")
numbered("It makes the problem learnable at all. If you ask “will this "
         "county be declared a disaster area tomorrow?” across all counties "
         "and all days, the answer is yes about one time in a thousand. "
         "There is almost nothing to learn from that. If you only ask about "
         "counties that have just been hit with extreme rain, the answer is "
         "yes about one time in twenty — 5.4% during the practice years and "
         "4.6% during the test years. Same reality, a far more answerable "
         "question.")
numbered("It matches a decision somebody actually has to make. Emergency "
         "managers already know it rained; they were watching. What they do "
         "not know is which of the forty affected counties to look at first.")

h("What counts as “extreme rain” is different in every county", 2)
p("Each county gets its own threshold, set at the wettest 1% of three-day "
  "stretches in its own history. So 117 mm of rain over three days is an "
  "unremarkable week in Baldwin County, Alabama, and a once-in-a-lifetime "
  "event in a drier county further west. Using one single rainfall number "
  "for the whole Gulf Coast would have done nothing except pick out the "
  "wettest places over and over again. Each threshold is calculated from the "
  "practice years only and then frozen, so it can never be quietly adjusted "
  "to flatter the results.")
p("Across the full record this gives 22,754 extreme-rain events. Of those, "
  "1,229 were followed by a federal declaration within two weeks — about one "
  "in twenty. Those events cluster into 413 distinct dates, and that number "
  "turns out to matter a great deal later.")

# ================================================================== 6 features

h("6. The model itself", 1)
p("The tool looks at nine pieces of information. Every one of them can be "
  "worked out from rainfall alone, which is exactly why the live version is "
  "possible at all — there is no ingredient that has to be hand-entered or "
  "waited for.")

table(["Ingredient", "In plain words"],
      [["ratio", "How far past its own extreme-rain threshold this county "
        "has gone"],
       ["log_fwd3", "How much rain in total over the three days, in "
        "millimetres"],
       ["max1", "How much fell on the single worst day of those three"],
       ["prev7", "How much rain fell in the week beforehand — was the ground "
        "already wet?"],
       ["prev30", "How much rain fell in the month beforehand — is the soil "
        "already saturated?"],
       ["month_sin, month_cos", "The time of year, written in a way that "
        "makes December and January neighbours instead of opposites"],
       ["other_extent", "How many other counties in the same state crossed "
        "their threshold that same day"],
       ["gulf_extent", "How many counties across the entire Gulf Coast "
        "crossed their threshold that same day"]])

p("Those last two matter far more than they look. FEMA does not declare "
  "disasters one county at a time — a governor requests help for a region, "
  "and the federal government responds to an event. One county with terrible "
  "rainfall is a local flood that the state handles. Forty counties with "
  "terrible rainfall on the same day is a federal emergency. Without these "
  "two ingredients, the tool would be blind to the single thing that most "
  "determines the answer.")

h("Why a simple method was chosen over a complicated one", 2)
p("There are only 413 distinct dates in the entire record on which a "
  "declaration-triggering event happened. That is a small number, and it is "
  "the real constraint on this project. With so few genuine examples, a "
  "complicated method has enough freedom to memorise the particular quirks "
  "of those 413 days rather than learn anything general — and it will look "
  "brilliant right up until it meets a new year.")
p("So the shipped tool is logistic regression, which is about the simplest "
  "method there is: it gives each ingredient a weight, adds them up, and "
  "converts the total into a percentage. The advantage is that you can read "
  "it. It can state, in a sentence, that a wetter preceding month raises the "
  "odds of a declaration by about half again, with everything else held "
  "steady. The complicated methods cannot tell you that.")
p("XGBoost was still tried, as a check on whether something more flexible "
  "would find patterns the simple method missed. It did score slightly "
  "higher on the final test — 0.1045 against 0.0929 on the scoring measure "
  "used throughout this document. But here is the catch. These methods "
  "involve an element of randomness, so running the same XGBoost three times "
  "with different random starting points gives three different scores, and "
  "those three scores vary by about 0.0140 between themselves. The advantage "
  "over the simple method was 0.0116 — smaller than the amount XGBoost "
  "disagrees with itself. So the apparent win is indistinguishable from "
  "luck, and shipping the simpler, readable, repeatable method is the "
  "defensible decision rather than a compromise.")

h("What the model learned", 2)
p("Each ingredient was put on a common scale first, so the numbers below can "
  "be compared directly against each other. The “effect on the odds” column "
  "reads like this: 2.15 means the odds roughly double when that ingredient "
  "goes up by a typical amount. Below 1.0 means the odds go down.")
table(["Ingredient", "Weight", "Effect on the odds", "What it means"],
      [["log_fwd3", "+0.767", "2.15", "More rain overall, higher risk"],
       ["prev30", "+0.412", "1.51", "Wetter preceding month, higher risk"],
       ["gulf_extent", "+0.186", "1.20",
        "More of the Gulf Coast hit at once, higher risk"],
       ["max1", "+0.156", "1.17",
        "Rain concentrated into one day, higher risk"],
       ["ratio", "-0.342", "0.71", "See the explanation below"],
       ["prev7", "-0.349", "0.71", "See the explanation below"],
       ["other_extent", "-0.368", "0.69", "See the explanation below"],
       ["month_sin", "-0.717", "0.49", "Part of the time-of-year effect"],
       ["month_cos", "-0.679", "0.50", "Part of the time-of-year effect"]])

p("Somebody will notice the minus signs and ask how more rain can possibly "
  "lower the risk. It cannot, and that is not what the table says. The "
  "catch is that some of these ingredients overlap heavily, and each number "
  "describes what that ingredient adds once the others are already known.")
p("Take ratio and log_fwd3. Both describe the same rainfall — one as an "
  "absolute amount, one as a multiple of what that county normally gets. "
  "Once you already know a county got 200 mm, the only extra information in "
  "the ratio is whether 200 mm is unusual there. A high ratio therefore "
  "points at a normally dry county, where the same 200 mm does less damage "
  "because there is less built on the floodplain. The minus sign is picking "
  "up that, not saying rain is protective. The same thing happens with "
  "other_extent and gulf_extent: between them the model separates a storm "
  "confined to one state from one spanning the whole coast.")
p("The practical rule for presenting this: never quote one of these numbers "
  "on its own. They only make sense as a set.")

# ================================================================== 7 ablation

h("7. The real experiment, and a result that came back negative", 1)
p("The scientific question was not “does this work” but “which ingredients "
  "actually earn their place”. So the ingredients were added in stages, and "
  "at each stage the question was whether the tool genuinely got better, or "
  "whether it just looked better by chance.")
p("Telling those two apart needs care. With only 413 real events, a small "
  "improvement can easily be an accident of which particular years ended up "
  "in the test block. So each comparison was repeated 2,000 times on "
  "randomly resampled versions of the data, which produces a range rather "
  "than a single number. If that range includes zero, the improvement cannot "
  "be told apart from luck.")

table(["What was added", "Change in score", "Plausible range", "Verdict"],
      [["Rainfall alone → plus how widespread the event is",
        "+0.0079", "+0.0020 to +0.0164",
        "Real improvement — the range stays above zero"],
       ["plus population and land area",
        "-0.0056", "-0.0157 to +0.0011",
        "Cannot be told apart from luck"],
       ["plus social vulnerability scores",
        "-0.0001", "-0.0047 to +0.0032",
        "Cannot be told apart from luck"]])

p("So how widespread the event is earns its place. Population, land area and "
  "the four social vulnerability scores do not — their effects are tiny and "
  "their ranges sit across zero.")
callout("This is a genuine negative result, and it needs to be stated "
        "carefully. The correct wording is: once you already know how severe "
        "the rainfall was and how widely it was spread, social vulnerability "
        "adds no measurable predictive power here, at this sample size. The "
        "wrong wording is “social vulnerability does not matter in "
        "disasters”. That is a much bigger claim and this study does not "
        "support it. Vulnerable places are often also places that get hit "
        "harder, so by the time rainfall and spread are accounted for there "
        "may be nothing distinct left to detect — especially with only 413 "
        "events to learn from.")
p("The shipped tool is therefore the nine-ingredient version: rainfall plus "
  "how widespread the event is, and nothing beyond that, because that is the "
  "last point where the evidence supported adding anything.")

# ================================================================== 8 stage 3

h("8. Checking that it actually works", 1)

h("Making the percentages honest", 2)
p("There is a difference between a tool that ranks counties correctly and a "
  "tool whose numbers mean something. The first version was good at putting "
  "the right counties at the top, but its actual percentages were badly "
  "wrong — it was far too confident. So a small correction step was fitted "
  f"afterwards, using only the {M['calibrated_on']} block, which was set "
  "aside for exactly this and never used for anything else.")
table(["Measure of honesty", "Before correction", "After correction"],
      [["How far off the percentages were overall (lower is better)",
        "0.2036", "0.0429"],
       ["How far off a stated percentage is in practice", "0.3442", "0.0121"]])
p("Read the second row as follows. Before the correction, when the tool said "
  "8% it was off by around 34 percentage points — meaningless. After, it is "
  "off by around 1 percentage point. So when the dashboard now says 8%, "
  "roughly 8 out of every 100 similar county-days really did end in a "
  "declaration. That is what makes the number on screen worth reading rather "
  "than just a position in a list.")

h("Why we never report accuracy", 2)
p("Accuracy is the obvious measure and it is completely misleading here. "
  "Declarations happen on 4.6% of these county-days, so a tool that simply "
  "says “no declaration” every single time, for every county, forever, is "
  "95.4% accurate. It is also worthless. Any project reporting high accuracy "
  "on a rare event is either confused or hoping you are.")
p("Instead this project reports two things. The first is a score of how well "
  "the tool ranks the county-days that really did end in a declaration "
  "towards the top of the list — formally called average precision. The "
  "second, and the more intuitive one, is how many times better than "
  "guessing that is. Both can be made to look bad by a bad tool, which is "
  "the entire point of choosing them.")

h("The headline numbers", 2)
table(["Measure", "Value"],
      [["Ranking score on the unseen years", f"{M['test_ap']:.4f}"],
       ["How often a declaration actually happens",
        f"{M['test_prevalence']:.2%}"],
       ["Times better than guessing",
        f"{M['test_ap'] / M['test_prevalence']:.2f}x"],
       ["Highest chance ever reported for any county",
        f"{M['max_observed_probability']:.1%}"],
       ["County-days in the unseen test years", "5,260"],
       ["Separate real disaster episodes in those years", "76"]])

h("Choosing how sensitive to make it", 2)
p("Somebody has to decide what counts as “flag this county”. Set the bar low "
  "and you catch almost every real disaster but drown an analyst in work. "
  "Set it high and the workload is tiny but you miss most of them. There is "
  "no clever answer that avoids this trade-off; there is only choosing where "
  "to sit on it.")
p("In the table below, “of those flagged, how many were real” is the share "
  "of flagged counties that really were declared. “Of the real ones, how "
  "many were caught” is the share of actual declarations the tool picked up.")
table(["Flag a county above", "Counties flagged per year",
       "Of those flagged, how many were real",
       "Of the real ones, how many were caught", "Times better than guessing"],
      [[f"{op['threshold']:.0%}", f"{op['flagged_per_year']:.0f}",
        f"{op['precision']:.1%}", f"{op['recall']:.1%}", f"{op['lift']:.2f}x"]
       for op in M["operating_points"]] +
      [["15%", "3", "5.6%", "0.4%", "1.21x"]])
p("The 15% row is included deliberately, because an audience always asks why "
  "not just raise the bar higher and get better precision. At 15% the tool "
  "fires about three times a year in total, and performance actually gets "
  "worse rather than better. There are simply too few examples that far up "
  "for anything measured there to be trustworthy.")

h("The hardest test: a state it has never seen", 2)
p("The toughest question you can ask is whether the tool learned something "
  "general about rainfall and disasters, or just memorised the habits of "
  "particular places. So each state was deleted from the practice data "
  "entirely, the tool was rebuilt from the remaining four, and then it was "
  "pointed at the state it had never seen.")
table(["State", "Extreme-rain days", "Declarations", "How often declared",
       "Times better than guessing when unseen"],
      [["Florida", "775", "51", "6.6%", "1.98x"],
       ["Louisiana", "579", "76", "13.1%", "1.56x"],
       ["Mississippi", "228", "21", "9.2%", "1.44x"],
       ["Alabama", "746", "60", "8.0%", "1.23x"],
       ["Texas", "2,932", "34", "1.2%", "1.11x"]])

callout("Texas does not work. At 1.11x the tool is barely better than "
        "guessing there, and the dashboard says so on every screen that can "
        "show a Texas county. Never present a Texas number as a finding. If "
        "someone asks about Houston, the correct answer is that this tool "
        "has not been shown to work in Texas.")
p("The reason is sitting right there in the table. Texas produces more "
  "extreme-rain days than any other state — 2,932 of them — but only 34 "
  "declarations. That is 1.2%, against 13.1% in Louisiana. Texas counties "
  "blow past their rainfall thresholds constantly and nothing official "
  "happens. Whatever decides a Texas declaration is mostly not in the "
  "rainfall. It might be the sheer size of Texas counties, or how the state "
  "handles things itself before asking Washington for help. The tool cannot "
  "see any of that, so it fails there, and the honest move is to say so "
  "rather than quietly average it into the headline number.")

h("Letting the tool say “I don't know”", 2)
p("One extra check was run. Instead of always producing an answer, the tool "
  "was allowed to return a range of possible answers, or occasionally "
  "nothing at all when the case was too unclear. The target was to be right "
  "90% of the time; it managed 91.0%. In 95.0% of cases it committed to a "
  "single answer, and in 5.0% it declined to commit. A tool that knows when "
  "to stay quiet is more useful than one that always has an opinion.")

# ================================================================== 9 stage 4

h("9. The dashboard", 1)
p("Once the tool was finished, everything it had learned was written out "
  "into one small file: the nine weights, the correction step, and each "
  "county's own rainfall threshold. From there, producing a prediction is "
  "just arithmetic — multiply, add, convert to a percentage. No special "
  "software is needed, which is why it can run inside a web browser.")
p("That rewritten arithmetic was checked against the original research code "
  "and the two agree to nine decimal places. This matters more than it "
  "sounds. It means the website and the paper can never disagree with each "
  "other, because there is no second version of the tool to drift out of "
  "step.")
p("The county shapes on the map were converted once, in advance, into plain "
  "drawing instructions, so the browser does not have to download a mapping "
  "library. The entire site is three files and four small data files, hosted "
  "free on GitHub Pages. There is no server, no database, no tracking, and "
  "nothing to keep running.")

table(["File", "Size", "What is in it"],
      [["model.json", "21 KB",
        "The weights, the correction step, and 218 county thresholds"],
       ["replay.json", "177 KB",
        "Every extreme-rain day in the test years, with what really happened"],
       ["geo.json", "168 KB", "The county outlines for the map"],
       ["live.json", "13 KB", "The current rolling forecast window"]])

h("The three pages", 2)
bullet("Looking back — every extreme-rain day in the test years, showing the "
       "tool's ranking next to what actually happened. Because the real "
       "outcome is printed in the table, anyone watching can check the tool "
       "themselves rather than take your word for it. The best single day is "
       "17 May 2021, during the Louisiana floods, where 5 of the top 6 "
       "counties it ranked were genuinely declared.")
bullet("Right now — a rolling window running from recent measured rainfall "
       "out to the end of the weather service forecast. Nothing on this page "
       "can be checked, because the future has not happened yet. Say that "
       "out loud when demonstrating it; it is the difference between a "
       "demonstration and a claim.")
bullet("Sensitivity slider — recalculates all the numbers from section 8 "
       "live in the browser, across all 5,260 test county-days, as you drag "
       "the threshold. It reproduces the published table exactly, and that "
       "is the whole point: an audience can verify the paper's numbers in "
       "front of you.")

# ================================================================== 10 live

h("10. How the live page gets its data", 1)
p("The tricky part is not the forecast. It is the past. Two of the nine "
  "ingredients describe how much rain fell in the previous week and the "
  "previous month, and that is measured rainfall, which comes from a "
  "completely different place than the forecast and arrives on a different "
  "schedule.")

table(["Source", "What it provides", "How current it is"],
      [["NOAA weather station records", "Rain that has actually been "
        "measured, from the same stations used to build the tool",
        "About two days behind"],
       ["National Weather Service forecast", "Rain expected over roughly the "
        "next week", "Up to the minute"]])

p("A built-in check compares the live data against the stored research data "
  "for days where both exist. They match to within 0.05 mm 98.76% of the "
  "time. They are not identical, and never will be, because NOAA keeps "
  "quality-checking and revising recent measurements for weeks after first "
  "publishing them.")

h("The gap in the middle", 2)
p("Here is the awkward consequence. Measured rain stops about two days ago. "
  "The forecast starts today. In between sits a short stretch — usually one "
  "day, sometimes three across a weekend — that neither source covers. Since "
  "every prediction needs a full three days of rainfall, any day whose "
  "three-day window touches that stretch cannot be calculated at all.")
p("The dashboard is upfront about this. Those dates are named on screen as "
  "unavailable, and if you pick one it jumps to the nearest usable day and "
  "tells you why it moved. A day that cannot be calculated is never drawn as "
  "an empty map, because “nothing happened” and “we cannot know” are two "
  "very different statements and showing them identically would be a lie by "
  "omission.")

h("Quiet days", 2)
p("On most days, no county anywhere on the Gulf Coast crosses its threshold, "
  "so there is genuinely nothing to report. Rather than show a blank screen "
  "that looks broken, the dashboard says so plainly and lists the counties "
  "that came closest, with how far they got — for example Tallapoosa, "
  "Alabama reaching 59 mm of the 82 mm it would need, which is 72% of the "
  "way. Those counties are shaded flat grey, never a colour from the risk "
  "scale, because below the threshold there is no percentage to report. "
  "There is only a distance.")

h("Keeping it current", 2)
p("An automated job rebuilds the live window every day at 10:20 UTC. Without "
  "it the live page would silently freeze on whatever day it was last "
  "generated — and a stale forecast still looks perfectly current, which is "
  "worse than having no live page at all. The browser also re-checks the "
  "file on every visit instead of trusting whatever it downloaded last time.")

# ================================================================== 11 how built

h("11. How it was built, in order", 1)
p("If someone asks “what did you actually do”, this is the sequence.")
numbered("Collected rainfall from 368 weather stations, every federal "
         "disaster declaration, and the social vulnerability scores. "
         "Combined them into one table with a row for every county and every "
         "day.")
numbered("Worked out what normal rainfall looks like for each county at each "
         "time of year, using only 1990 to 2010, then locked it.")
numbered("Tried to predict disasters months ahead using XGBoost and a neural "
         "network. Compared both against deliberately dumb benchmarks. Found "
         "that beyond the current month the predictions were no better than "
         "a coin flip.")
numbered("Changed the question. Instead of predicting the rain, waited for "
         "it — kept only the days where a county passed its own "
         "extreme-rainfall mark, and asked what happens next.")
numbered("Built nine ingredients from rainfall alone, so that the whole "
         "thing could eventually run live.")
numbered("Added the ingredients in stages and tested each addition properly. "
         "Rainfall spread earned its place. Population, area and social "
         "vulnerability did not.")
numbered("Fitted the simple readable method on 2000 to 2014, corrected its "
         "percentages on 2015 to 2019, and then tested it once on 2020 to "
         "2026 — years it had never been shown.")
numbered("Deleted each state in turn and retested, which is how the Texas "
         "problem was found.")
numbered("Exported everything to a small file, rewrote the arithmetic to run "
         "in a browser, checked the two versions agree to nine decimal "
         "places, and built the dashboard around it.")
numbered("Connected the live feeds, handled the gap between measured and "
         "forecast rain, and set up a job that refreshes it daily.")

# ================================================================== 12 FAQ

doc.add_page_break()
h("12. Questions you will be asked", 1)

qa = [
    ("My weather app says 60% chance of rain in Houston tomorrow. Why does "
     "your dashboard show nothing for Houston?",
     "Because three different things are being measured and they are easy to "
     "confuse. Your weather app shows the chance that any rain falls at all "
     "— one drop counts. This project uses a different forecast product "
     "entirely, which says how many millimetres are expected. Harris County "
     "needs 97.9 mm over three days before it even enters the data, and a "
     "60% chance of a light shower is nowhere near that. Then the thing this "
     "project outputs is a third quantity again: the chance of a federal "
     "disaster declaration. So both screens are telling the truth. They are "
     "answering different questions."),
    ("Why is the highest number only 15.7%? That seems low.",
     "Because that is genuinely what the history supports. Even after "
     "extreme rainfall, a federal declaration follows only 4.6% of the time. "
     "If the tool displayed 80% it would simply be wrong, and the dashboard "
     "was deliberately built so that its percentages mean what they say. The "
     "value here is the ordering and the roughly 2x improvement over "
     "guessing, not a big dramatic number."),
    ("Is 2x better than guessing actually good?",
     "It is modest, and it is real. It was measured on years the tool had "
     "never seen, against a benchmark that is harder to beat than it sounds, "
     "with proper uncertainty ranges. The common tricks that would have "
     "inflated it — shuffling the years, reporting accuracy instead, tuning "
     "against the test data — were all specifically avoided. A believable 2x "
     "is worth more than an unbelievable 10x."),
    ("Why not use AI or deep learning?",
     "A neural network was built and tested; it is in the repository. The "
     "limit is not how clever the method is, it is that there are only 413 "
     "genuine events in the whole record. Even XGBoost's small apparent "
     "advantage turned out to be smaller than how much it disagrees with "
     "itself between runs. No amount of model sophistication can create "
     "information that is not in the data."),
    ("Could someone actually use this?",
     "Not as a warning system, and not in Texas. As a way of ordering a "
     "to-do list for an analyst who already has to review every county that "
     "got hit: at the 10% setting it flags about 86 counties a year, and "
     "about 12 in every 100 of those were really declared, against a "
     "background rate of under 5 in 100. Whether that is worth someone's "
     "time is a judgement for an emergency management office, not for the "
     "people who built the tool."),
    ("What would improve it most?",
     "Not a better method. Better information. How well-resourced a county's "
     "emergency management office is, whether it has been declared before, "
     "river gauge levels, damage assessments. A disaster declaration is a "
     "political and administrative decision as much as a physical event, and "
     "right now the tool sees only rainfall."),
    ("How do I know the numbers on the website are real and not made up?",
     "The looking-back page prints what actually happened next to every "
     "prediction, and the sensitivity slider recalculates the published "
     "results from the raw data inside your own browser. Both can be checked "
     "live, in front of an audience, without having to trust anybody."),
    ("Who is this for?",
     "It was built as a research project, so the honest answer is that its "
     "first audience is the paper. The realistic practical audience would be "
     "an analyst at a state emergency management agency who already monitors "
     "the whole coast and needs to triage attention in the hours after a "
     "major rainfall event."),
]
for q, a in qa:
    p(q, bold=True)
    p(a)

# ================================================================== 13 limits

h("13. Known limits — say these before someone else finds them", 1)
bullet("Texas does not work. At 1.11x it is barely better than guessing "
       "there. Treat any Texas row as meaningless.")
bullet("The forecast and the training data do not measure rain in quite the "
       "same way. The forecast estimates rain at a single point in the "
       "middle of the county; the training data averaged real stations "
       "scattered across it. How much this matters has not been measured.")
bullet("A disaster declaration is a political and administrative decision as "
       "well as a physical event. The tool sees rainfall. It does not see "
       "whether a county is prepared, or whether its governor tends to "
       "request federal help.")
bullet("The weights in section 6 describe patterns among overlapping "
       "ingredients. They are not causes, and no single one of them should "
       "be quoted on its own.")
bullet("The live page will never exactly match the historical pages, because "
       "NOAA keeps revising recent rainfall measurements for weeks after "
       "first publishing them.")
bullet("The tool only speaks after extreme rain. For almost every county on "
       "almost every day, it has nothing to say at all — and that is by "
       "design, not a bug.")

# ================================================================== 14 running

h("14. Running it yourself", 1)
p("Nothing needs to be installed to look at the dashboard — it is just a "
  "public web page. To rebuild the analysis from scratch:")
code = doc.add_paragraph()
run = code.add_run(
    "git clone https://github.com/Ganeshpinnu5/county-flood-hurricane-risk\n"
    "cd county-flood-hurricane-risk\n"
    "pip install -r requirements.txt\n\n"
    "python stage2.py          # build the data and fit the model\n"
    "python stage3.py          # correct the percentages and run the checks\n"
    "python stage4_export.py   # write the model out for the browser\n"
    "python stage4_geo.py      # convert county shapes into map drawings\n"
    "python stage4_live.py     # fetch live rainfall and predict\n\n"
    "python serve.py           # then open http://localhost:8787")
run.font.name = "Consolas"
run.font.size = Pt(9.5)

p("The full rainfall dataset is 25 MB and is not stored in the repository, "
  "since it is large and belongs to NOAA rather than to us. Everything the "
  "dashboard and the daily refresh need is already there, so a fresh copy "
  "works immediately without any private files.")

h("What is in the repository", 2)
table(["File or folder", "What it is"],
      [["config.py, build_panel.py",
        "Settings, and the code that assembles the raw data"],
       ["baselines.py", "The deliberately dumb benchmarks"],
       ["train_xgb.py, tft_colab.py",
        "The first attempt — XGBoost and the neural network"],
       ["stage2.py", "The change of direction, the staged experiment, and "
        "fitting the final model"],
       ["stage3.py",
        "Correcting the percentages and all the validation checks"],
       ["stage4_export.py, stage4_geo.py",
        "Writing the model and the map out for the browser"],
       ["stage4_live.py", "Fetching live rainfall and predicting"],
       ["docs/", "The dashboard website itself"],
       ["artifacts/", "Every results table quoted in this document"]])

# ================================================================== 15 glossary

doc.add_page_break()
h("15. Glossary", 1)
p("Words that come up if somebody in the audience is technical, or if you "
  "read the code or the paper.")

table(["Term", "What it means"],
      [["Average precision",
        "A single score for how well a tool pushes the real cases towards "
        "the top of a ranked list. Higher is better. Used here instead of "
        "accuracy because the thing being predicted is rare."],
       ["Base rate / prevalence",
        "How often the thing actually happens. Here, 4.6% of extreme-rain "
        "county-days ended in a declaration."],
       ["Lift",
        "How many times better than guessing. A lift of 2x means the tool is "
        "twice as good as picking counties at random."],
       ["Accuracy",
        "The share of predictions that were correct. Deliberately avoided "
        "here: always saying 'no' scores 95.4% and is useless."],
       ["Precision",
        "Of the counties the tool flagged, what share really were declared."],
       ["Recall",
        "Of the counties that really were declared, what share the tool "
        "flagged."],
       ["ROC-AUC",
        "Another ranking score, running from 0.5 (coin flip) to 1.0 "
        "(perfect). Flattering when the event is rare, so it is reported "
        "only for the first attempt in section 4."],
       ["Calibration",
        "Making the stated percentages honest, so that things the tool calls "
        "8% really do happen about 8% of the time."],
       ["Platt scaling",
        "The specific small correction used to do that calibration."],
       ["Brier score",
        "A measure of how far off the stated percentages are. Lower is "
        "better."],
       ["Training / practice set",
        "The years the tool learned from. Here, 2000 to 2014."],
       ["Held-out / test set",
        "Years deliberately kept hidden until the very end, so the final "
        "score is honest. Here, 2020 to 2026."],
       ["Overfitting",
        "When a tool memorises the particular examples it practised on "
        "instead of learning a general pattern. It looks excellent in "
        "practice and fails on anything new."],
       ["Leakage",
        "When information from the test years sneaks into the practice "
        "stage. It inflates every result and is often invisible unless you "
        "look for it."],
       ["Bootstrap",
        "Repeating a measurement thousands of times on randomly resampled "
        "versions of the data, to see how much the answer wobbles."],
       ["Confidence interval",
        "The range the true answer plausibly sits in. If the range includes "
        "zero, the effect cannot be told apart from luck."],
       ["Logistic regression",
        "The simple method used here. Gives each ingredient a weight, adds "
        "them up, converts the total to a percentage."],
       ["XGBoost",
        "A more powerful method that builds thousands of small decision "
        "rules and combines them. Tried here; its advantage was within its "
        "own run-to-run noise."],
       ["Temporal Fusion Transformer",
        "A neural network designed for data that unfolds over time. Built "
        "and tested in the first attempt."],
       ["Feature",
        "One piece of input information. This project uses nine."],
       ["Standardised",
        "Rescaled so that different ingredients can be compared fairly "
        "against each other."],
       ["Odds ratio",
        "How much the odds change when one ingredient goes up by a typical "
        "amount. Above 1 means higher risk, below 1 means lower."],
       ["Conformal prediction",
        "A method that lets a tool return 'I am not sure' instead of forcing "
        "an answer."],
       ["Percentile",
        "A ranking position out of 100. A county's threshold is its 99th "
        "percentile, meaning only the wettest 1% of its three-day stretches "
        "are above it."],
       ["Climatology",
        "The long-run normal for a place and time of year. Also the name of "
        "the simplest benchmark used in section 4."],
       ["Antecedent rainfall",
        "Rain that fell before the event being studied. Matters because "
        "already-saturated ground floods faster."],
       ["QPF",
        "Quantitative Precipitation Forecast — how many millimetres of rain "
        "are expected. This is what the project uses."],
       ["PoP",
        "Probability of Precipitation — the chance of any rain at all. This "
        "is what a phone weather app shows, and it is not what the project "
        "uses."],
       ["GHCN-Daily",
        "NOAA's global archive of daily weather station readings. The source "
        "of all measured rainfall here."],
       ["JSON",
        "A plain text file format. The whole trained model is stored in one, "
        "at 21 KB."],
       ["GitHub Pages",
        "A free service that serves a website straight from a code "
        "repository. No server is being paid for or maintained."]])

p("Every number in this document is read straight from the result files when "
  "the document is generated, so it cannot drift away from the actual "
  "results.", italic=True)

doc.core_properties.author = "Ganesh Pinnu"
doc.core_properties.last_modified_by = "Ganesh Pinnu"

out = HERE / "PROJECT_EXPLAINER.docx"
doc.save(out)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
