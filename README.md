# County Flood & Hurricane Risk

> **This repository has moved.** Development and the live dashboard are now at
> [Sudarshan006/gulf-county-flood-hurricane-risk](https://github.com/Sudarshan006/gulf-county-flood-hurricane-risk).
> Nothing here is refreshed any more; the daily forecast job runs there. What
> remains is kept as the published history up to 18 November 2026.

**[Open the dashboard →](https://sudarshan006.github.io/gulf-county-flood-hurricane-risk/)**

Pick any county in the lower 48 and see the live 4-day weather forecast, any
National Weather Service warning in force right now, and — across the Gulf
Coast, where the models are trained and validated — modelled flood and
hurricane risk for a given day alongside the flood episodes and cyclone passes
actually on record for it.

The full training write-up is
[V2_TRAINING_RESULTS.docx](V2_TRAINING_RESULTS.docx): target, features, splits,
baselines, both architectures, the lead sweep and the deviation scoring, every
number read out of the sealed run directory. Regenerate it with
`python make_report.py`.

The dashboard is a static page. No server, no build step, no mapping library,
no API keys.

## The question

Given a county and a day, will a **NOAA-recorded flood episode or a HURDAT2
cyclone pass begin there within the next 3 days?**

Two things about that target are deliberate.

**Onset, not occurrence.** An "is a flood active today" label is trivially
predictable from yesterday's label and would score well while being worthless.

**Observations, not declarations.** An earlier version of this project
predicted FEMA disaster declarations. A declaration is an administrative act:
it arrives weeks after the water does, it requires a governor to request it,
and plenty of real flooding never receives one. That model was partly
predicting paperwork. Flood days now come from NOAA StormEvents and cyclone
passes from the HURDAT2 best track, both of which record the hazard itself. The
change multiplied the flood training signal roughly fourteen-fold — 1,517
positives became 20,734.

## What this is, and what it is not

It is **not a weather forecaster.** It does not predict rain and it does not
run a cyclone track model. It consumes rainfall and the official NHC forecast
advisories as issued, and maps them to hazard-onset risk.

It is **not a warning system.** Onset is rare: 1.05% of county-days for floods,
0.38% for hurricanes. Raw probabilities are correspondingly small, so the
dashboard shows a band ("Elevated", "High") derived from quantiles of the
scored panel. "High" means this county-day ranks in the top 1% of county-days,
not that a hazard is likely.

## Results

Panel of **3,252,826 county-days** across 356 Gulf Coast counties, 2000–2026.
Splits are contiguous blocks, never random — adjacent days share weather state,
so a shuffled split leaks across the boundary — and this run adds a **3-day
embargo**, so a forward-looking label can never straddle a split boundary and
the panel's last 3 days, whose labels cannot yet be verified, are dropped
before splitting.

| | Train | Validate | Test |
|---|---|---|---|
| Years | 2000–2014 | 2015–2019 | 2020–2026 |
| Rows | 1,866,781 | 618,985 | 764,238 |
| Flood onsets | 20,734 | 7,129 | 8,020 |
| Hurricane onsets | 4,303 | 1,599 | 2,246 |

Hurricane metrics are scored on 2020–2024 (2,223 onsets over 579,263 rows):
HURDAT2 is finalised a season behind, so later hurricane labels do not exist
yet. Flood metrics use the full test window.

Average precision and lift over prevalence are reported rather than accuracy or
ROC-AUC. At a 1% base rate a model that always answers "no" scores 99% accuracy
while being useless.

| Hazard | Model | Test AP | Base rate | Lift | vs best baseline |
|---|---|---|---|---|---|
| Flood | XGBoost, monotone | **0.26500** | 1.05% | 25.3× | **1.33×** |
| Flood | TFT, calibrated | 0.20051 | 1.05% | 19.0× | 1.01× |
| Hurricane | XGBoost, monotone | **0.54115** | 0.38% | 141.0× | **1.71×** |
| Hurricane | TFT, calibrated | 0.39966 | 0.38% | 104.1× | 1.26× |

The baselines are deliberately crude — one raw feature, or the historical rate
for the calendar month. On a rare-event target a crude predictor is much harder
to beat than it sounds: the best flood baseline is a single rainfall ratio
(`ratio_fwd3`) scoring 0.19893 on its own, and the best hurricane baseline is a
single forecast-track proximity feature (`cyc_dist2`) scoring 0.31699.

**Read the hurricane number with care.** The hurricane model's inputs include
a block of features built from the official NHC forecast as it was issued —
forecast track proximity, the forecast 34-kt wind radius, distance to the cone
edge. The label comes from HURDAT2's best track. So part of the hurricane
model's skill is NHC's own forecast skill re-expressed, not skill this model
added; the honest claim is the 1.71× over the best single forecast feature,
not the 141× over climatological prevalence. The same block is deliberately
shared with the flood model — much Gulf Coast flooding is tropical in origin —
and an ablation puts its contribution there at essentially nil.

The dashboard serves the XGBoost models, because they win both hazards. The
transformer's numbers are read from the run's results JSON rather than
re-scored in the browser, so the page and the document cannot diverge.

### Gradient boosting against the transformer

Both were trained cold in the same sealed run, on the same panel, the same
splits and the same label. The transformer sees a 30-day encoder window per
county, is trained with a positive-class weight of 50, and is then calibrated
with isotonic regression fitted on validation so its outputs can be read as
probabilities (post-calibration ECE: 0.0013 flood, 0.0003 hurricane). It is
the heavier model by a wide margin — the whole run took 23.1 hours on an H100,
nearly all of it transformer training — and it loses pooled on both hazards
(0.20051 vs 0.26500 flood, 0.39966 vs 0.54115 hurricane) and five of the six
exact leads below.

### How much warning does it actually buy?

Reporting one number for a "1 to 3 day forecast" hides the thing a user needs
most. "Something starts in the next three days" is a much easier question than
"something starts three days from now", so a single pooled score quietly
flatters the longer leads. Every lead below therefore has its own model,
trained and scored on its own target.

| Hazard | Lead | Base rate | Best single feature | XGBoost AP | TFT AP |
|---|---|---|---|---|---|
| Flood | +1 day | 0.374% | 0.0904 | **0.1780** (1.97×) | 0.1178 |
| Flood | +2 days | 0.373% | 0.1082 | **0.1300** (1.20×) | 0.1489 |
| Flood | +3 days | 0.373% | 0.0533 | **0.0777** (1.46×) | 0.0668 |
| Hurricane | +1 day | 0.128% | 0.1738 | **0.4829** (2.78×) | 0.3096 |
| Hurricane | +2 days | 0.128% | 0.1758 | **0.3915** (2.23×) | 0.2418 |
| Hurricane | +3 days | 0.128% | 0.1055 | **0.2169** (2.06×) | 0.1326 |

Boosting decays monotonically with lead on both hazards, which is the expected
shape, and beats its best single-feature baseline at every lead. The
transformer is monotone on hurricanes but **not** on floods (0.1178, 0.1489,
0.0668) — and its one win anywhere, flood +2 days, sits inside that inversion.
With roughly 60 flood episodes in the test period, a single-lead gap of this
size is within episode-level noise; the inversion is reported rather than
smoothed, and it is a reason to treat the transformer's per-lead numbers as
noisier than the boosted ones, not a finding about lead time.

The honest reading of this table: at +3 days the flood model scores 0.0777,
which is 21× its base rate but far below the pooled 0.26500. The pooled number
is the one a less careful write-up would have published as a three-day
forecast.

### What survives contact with a real forecast

The training features include forward-looking rain over the prediction window
— rain that, on the morning the model runs, a forecaster would only have as a
forecast. The published numbers above are therefore an upper bound, and
instead of leaving that as a caveat, it is measured: the test window was
re-scored with the five forward-rain features replaced by the **NOAA GEFS v12
ensemble-mean forecast as it was issued** (0.5°, 00Z run), nothing retrained.
The audit covers 2020-09-24 onward, where the GEFS archive begins, so its
"observed" column differs slightly from the headline table. Confidence
intervals are a 2,000-draw episode-block bootstrap (7-day gap), because
county-days within one weather episode are not independent.

| Flood (60 episodes) | AP | 95% CI |
|---|---|---|
| Observed forward rain (as published) | 0.27433 | [0.24228, 0.29651] |
| GEFS forecast rain (deployable) | 0.09301 | [0.07590, 0.11041] |
| Best forecast-fed single feature | 0.06393 | [0.05229, 0.07895] |

Fed real forecasts, the flood model keeps **33.9%** [30.5%, 38.4%] of its
published skill. It still beats the best single feature given the same
forecast rain by **1.45×** [1.19×, 1.73×], so the model adds value at fixed
information — but it reaches only 0.47× of the observation-fed baseline it is
published against. The 3-day county rain forecast itself (correlation 0.58
with what fell) is the binding constraint, not the model.

Hurricane is the opposite story: **0.55567** [0.44122, 0.66051] on forecast
rain against 0.53974 [0.42015, 0.64883] on observed — forecast rain is the
*better* input, by +0.016 [-0.018, +0.052]. The hurricane model's skill rides
on the cyclone block, which is already forecast-sourced, so it loses nothing
by deployment. At 19 test episodes this is reported, not claimed: the interval
on the difference straddles zero.

Regenerate with `python audit_forecast_rain.py`; the numbers live in
`artifacts/audit_forecast_rain_gefs.json`.

### Deviation from observation

Average precision ranks; it does not say what a user would experience at a
threshold. Scored at each hazard's 99th-percentile probability over the test
period:

| Hazard | Hits | Misses | False alarms | Detection rate | Precision | CSI | Brier skill |
|---|---|---|---|---|---|---|---|
| Flood | 2,562 | 5,409 | 5,003 | 32.1% | 33.9% | 0.197 | 0.146 |
| Hurricane | 1,638 | 585 | 4,155 | 73.7% | 28.3% | 0.257 | 0.306 |

Roughly one flood alarm in three is real, and a little over one hurricane
alarm in four — but the hurricane model now catches roughly three storm-onset
days in four. Both beat the base rate substantially and neither is a siren.

## Coverage

| | Counties |
|---|---|
| Drawn on the map | 3,109 (all of the lower 48) |
| Risk modelled | 356 |
| Live weather + NWS warnings | 3,109 |

Risk modelling is restricted to the five Gulf states, and within them to
counties with a rainfall station in the source data. Every other county is
drawn unshaded and says so when clicked. Weather and official warnings are
queried per-county at the county's area-weighted centroid and work nationwide.

## Provenance

Every published number comes from one sealed training run,
`20261115T174654Z-hpc-full`, finished after 23.1 hours on an NVIDIA H100 NVL
(Slurm, crest-g002; python 3.11.18, xgboost 3.2.0, torch 2.5.1+cu121). The run
is self-contained — no file outside its own directory contributed to a
reported number — and it was made from a clean working tree at commit
`b5a23ed3`, so the code that ran is exactly the code at that commit. It was
trained on a panel with sha256 `b7455bfd…`, and each of the eight boosters was
reloaded and confirmed to reproduce its recorded test average precision to
five decimals before export. XGBoost and the transformer were trained in the
same run, so the comparison between them is never a comparison across panels
or labels.

Runs are sealed for a reason. Earlier work wrote into a shared `artifacts/`
directory, and three times a result could not be traced back to the panel that
produced it. A run directory is hash-verified against its panel; a loose model
directory is whatever was written last.

Scoring a saved booster has one trap worth naming. Early stopping keeps
training for `early` more rounds after the best one, and `save_model` writes all
of them, so the file holds `best_iteration + early` trees — here the flood file
has 245 trees for a best iteration of 194, and the hurricane file 71 trees for
a best iteration of 20, so scoring that file whole would use more than three
times the trees that were selected. `v2_models.py` therefore passes
`iteration_range` explicitly, with `best_iteration` read from the run's results
JSON rather than from the file.

## Repository layout

### The pipeline

| File | What it is | Why it exists |
|---|---|---|
| `config.py` | Shared paths, year splits, hazard type lists, climatology window. | One place to change a split or a data directory. The climatology baseline is frozen to 1990–2010 so anomaly features for later years never see their own distribution — widening it leaks test information into training features. |
| `labels.py` | Builds flood onsets from NOAA StormEvents and cyclone passes from the HURDAT2 best track. | The observational label set. Keeping it separate from the panel build means the label definition can be audited on its own. |
| `build_daily.py` | Builds the county-**day** panel: rainfall features, hazard onset labels, population density, social vulnerability. Writes `artifacts/daily_panel.parquet`. | The dataset everything downstream trains and scores on. |
| `cyclone_features.py` | Builds the cyclone feature block from the NHC a-deck — the forecast as it was issued, never the best track. | Using the best track as a feature would hand the model the answer; using the advisory is what a forecaster actually had that morning. |
| `pipeline.py` | The whole training run: baselines, XGBoost, the TFT, the lead sweep, the deviation scoring, plots. Writes one sealed run directory. | A run is stamped with the panel hash, library versions and GPU, so a result can always be traced to its inputs. |
| `gen_pipeline_nb.py` | Generates `pipeline.ipynb` from `pipeline.py`. | The notebook is the Colab entry point. Generating it means the two cannot drift. |
| `archive_colab_run.py` | Pulls a finished Colab run directory back down intact. | A run that only exists on a disposable VM is not provenance. |
| `audit_forecast_rain.py` | Re-scores the test window with forward rain replaced by the archived GEFS forecast. | The forward-rain features are the project's largest overstatement; this measures it instead of footnoting it. |
| `v2_models.py` | The single place that knows how to *load and score* the trained models. | Both consumers import from here, so the dashboard can never show risk from a model other than the one the paper reports. |
| `export_v2.py` | Builds the JSON payloads the dashboard loads: `meta.json`, `events.json`, `history.json`. | Scoring every county-day would be a 3M-entry payload. This ships only the windows a user actually opens. |
| `fetch_live.py` | Fetches Open-Meteo weather per county and scores today plus the next 3 days. Writes `docs/data/live.json`. | Open-Meteo returns up to 92 past days alongside the forecast, so one request per county supplies both the antecedent windows and the forward-looking rain. No API key. |
| `stage4_geo.py` | Downloads public county/state GeoJSON and projects it to SVG path strings. | Projecting in Python means the dashboard needs no mapping library. Albers equal-area conic, so county areas stay comparable by eye. |
| `make_report.py` | Generates `V2_TRAINING_RESULTS.docx` from the sealed run. | Keeping it generated is what makes "the document cannot drift from the results" true. |
| `make_ledger.py` | Generates `EXPERIMENTS.md` from every run manifest on disk. | A hand-kept ledger is a ledger that disagrees with the runs it describes. |
| `hpc/` | Slurm batch scripts for the same pipeline on a cluster. | The GPU path that does not depend on a Colab session staying alive. The published run is one of these jobs. |
| `serve.py` | Local dev server on port 8787. | Identical to `http.server` except it tells the browser not to cache. |

### The dashboard

| File | What it is |
|---|---|
| `docs/index.html` | Page structure: controls, map, county detail panel, model card. GitHub Pages serves from `/docs`, so this directory *is* the site. |
| `docs/app.js` | All behaviour: map drawing, county selection, risk bands, live weather and NWS alert fetches, event history, model card. Plain JavaScript, no framework. |
| `docs/styles.css` | Styling and both colour themes. Every map colour is a CSS variable, so switching theme repaints without rebuilding geometry. |
| `docs/data/geo.json` | 3,109 pre-projected county paths, 49 state outlines, names and centroids. Centroids are area-weighted on each county's largest ring, so the weather query point lands *inside* the county. |
| `docs/data/meta.json` | Model metrics, risk bands, coverage, and the full county list flagged modelled / not modelled. |
| `docs/data/history.json` | Model risk by date for the shipped windows. What the map shades in Past-events mode. |
| `docs/data/events.json` | Recorded flood episodes and cyclone passes per county, each paired with what the models said on the three days before onset. |
| `docs/data/live.json` | Today + 3 days, scored from the live forecast. Regenerated daily by GitHub Actions. |
| `.github/workflows/refresh-live.yml` | Daily job that re-runs `fetch_live.py` and commits the result. Without it the live view silently freezes on the day it was last run. |

### The sealed run

`artifacts/runs/20261115T174654Z-hpc-full/` holds the manifest, per-model
results, plots, console log, per-epoch TFT metrics and the eight XGBoost
boosters. The boosters ship because the daily refresh job scores with them
from a clean checkout. The transformer checkpoints do not — nothing the site
serves reads them, and the dashboard takes the TFT's numbers from the results
JSON.

## Reproducing

```bash
pip install -r requirements.txt

python build_daily.py                   # county-day panel with onset labels
python pipeline.py                      # the full training run -> artifacts/runs/<id>/
python stage4_geo.py                    # project county geometry to SVG paths
python export_v2.py                     # write the dashboard payloads
python fetch_live.py                    # score today + next 3 days from live weather
python make_report.py                   # regenerate the results document
python audit_forecast_rain.py           # re-score on archived GEFS forecasts

python serve.py                         # http://localhost:8787
```

`python v2_models.py` is a smoke check: it reloads every booster and confirms
it reproduces the test average precision the run recorded, which catches a
truncated download or a drifted feature list before either reaches the
dashboard. (It applies the same 3-day embargo trim the pipeline does — scoring
the panel's unverifiable tail days as negatives shifts the flood APs at the
fourth decimal.)

The raw rainfall panel and the FEMA/SVI source tables are not committed — they
are large and not ours to republish. `build_daily.py` expects them in the
directory named by `CLIMATE_DATA_DIR`; everything after that step runs from
committed files.

After a panel rebuild, regenerate the live feature cache with
`python fetch_live.py --rebuild-static`. `refdata/county_static.csv` holds
per-county rainfall climatology that forms the denominators of the ratio
features, and the top flood feature is one of them — a cache from an older
panel would quietly score every county against the wrong normal.

## Known limits

- **The label is an observational record, not hazard physics.** A StormEvents
  flood episode exists because a local Weather Service office wrote one up, so
  a real flood nobody reported counts as a negative. Hurricane positives are
  purely geometric — the county intersected the best track — so a county grazed
  by a weak system and a county destroyed by a major landfall are the same
  positive.
- **Training rainfall is observed, not forecast.** Features use station
  rainfall over the prediction window, which a real forecaster would not have
  on the morning the model runs. This is the project's largest overstatement,
  and it is now measured rather than footnoted: fed archived GEFS forecasts,
  the flood model keeps 33.9% of its published skill (see *What survives
  contact with a real forecast*). Treat the flood headline as an upper bound;
  the hurricane numbers survive deployment because their driving features are
  already forecast-sourced.
- **Part of the hurricane skill is NHC's, re-expressed.** The cyclone features
  are the official forecast; the label is the best track the forecast was
  chasing. The 1.71× margin over the best single forecast feature is the
  model's own contribution; the raw AP is not.
- **No held-out-state validation in this run.** Earlier work measured what the
  models give back when a whole state is removed from training. This run did
  not repeat it, so nothing here supports a claim about counties outside the
  Gulf Coast. This is the clearest gap.
- **356 of 3,109 counties are modelled.** Training is restricted to the five
  Gulf states and, within them, to counties with a rainfall station in the
  source data.
- **The rainfall record has holes.** The panel spans 9,645 calendar days and
  the median county has 9,595, but the worst has 3,702 and 28 counties fall
  below 80%. Events whose three-day lead-up falls entirely in a gap have no
  model reading at all.
- **Hurricane labels stop at 2024.** HURDAT2 is finalised a season behind, so
  hurricane metrics are scored on 2020–2024 while flood metrics use the full
  test window. The two hazards' test sets are not the same size.
- **No soil moisture, no river gauges.** Antecedent wetness is proxied by 30-,
  60- and 90-day rainfall, a weak substitute for catchment state.
- **No storm surge, no intensity beyond the advisory.** The cyclone block
  carries forecast track, 34-kt wind radius and cone geometry; surge and
  rapid-intensification risk are absent entirely.
- **Episodes, not county-days, are the real sample size.** The test period
  holds 8,020 flood and 2,223 scored hurricane onset days, but only ~60 flood
  and ~19 hurricane episodes, which is why the forecast-rain audit reports
  episode-bootstrap intervals and declines to claim its hurricane result. The
  headline run metrics are point estimates.

## Data sources

NOAA NCEI GHCN-Daily · NOAA NCEI StormEvents · NHC HURDAT2 Atlantic best track ·
NHC a-deck forecast advisories · NOAA GEFS v12 archive (audit) ·
FEMA Disaster Declarations Summaries (secondary target only) ·
CDC/ATSDR Social Vulnerability Index · Open-Meteo (forecast and ERA5 archive) ·
NWS api.weather.gov (active alerts)
