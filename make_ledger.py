"""Generate EXPERIMENTS.md from every run manifest under artifacts/runs/.

A hand-kept ledger is a ledger that disagrees with the runs it describes, which
is the failure this project already had once: the README carried numbers from a
discarded run for two days because it was the only surface a human maintained.
So every figure here is read out of a manifest. The only typed text is NOTES
below -- why a run was made and what was wrong with it, which no manifest
records -- and it is keyed by run_id so it cannot drift onto the wrong run.

    python make_ledger.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "artifacts" / "runs"
DEST = ROOT / "EXPERIMENTS.md"

# Why the run happened and what it turned out to be worth. Keyed by run_id.
NOTES = {
    "20261103T040430Z-run-colab": (
        "First full-panel run, on a Colab T4. Produced the V1 dashboard "
        "numbers. Superseded: its label window counted an event as positive on "
        "the day it was recorded rather than the day it began."),
    "20261104T015759Z-run": (
        "First run on the corrected label. Discarded, not published: it was "
        "warm-started from the previous feature set, so its boosters inherited "
        "rounds fitted against the old label and the run cannot be reproduced "
        "from its own directory."),
    "20261106T012030Z-run": (
        "Cold run on the corrected label. This is the run every published "
        "figure comes from. Known defects, all disclosed: five forward-rain "
        "features describe weather after the prediction date, the splits carry "
        "no embargo so the 2014 tail holds 2015 answers, and the tree is "
        "dirty at fe68bcc9.\n\n"
        "Its lead-sweep APs match the discarded run's to five decimals, which "
        "looks like a copied file and is not one. The pooled target comes from "
        "the panel and did change between the two; the lead targets are derived "
        "inside `pipeline.py` by shifting `{hazard}_onset`, which did not. No "
        "per-lead checkpoint existed yet either, so the lead models were cold "
        "in both runs. Same seed, same labels, and test-period features that "
        "differ by about 1e-7 put them in the same place. The fits are real: "
        "`fit_seconds` is 148 against 263. Checked separately, the pooled "
        "`y_flood` column and the OR of its three derived lead targets agree "
        "on 99.998% of rows, the exceptions being county and panel edges where "
        "the shift has nothing to read."),
    "20261107T202150Z-run": (
        "Abandoned mid-run, kept only as the record that it happened. It was "
        "splitting the storm label into 50-kt and 64-kt tiers, which its "
        "`deviation_storm50` and `deviation_storm64` files are the remains of. "
        "The tiers were reverted: they cut the positive count per tier to the "
        "point where the test AP was driven by a handful of episodes. No "
        "headline, no models, nothing published from it."),
    "20261108T044034Z-cyc": (
        "Killed by the OOM reaper during the flood fit, which is why it holds "
        "only `params_cyc.json` and the split plot. The panel read is 70 "
        "columns wide and the hist cache on 1.87M rows by 44 features did not "
        "fit alongside a desktop session. Relaunched as the run below once "
        "memory was free; nothing here was ever scored."),
    "20261108T044448Z-cyc": (
        "The first complete cyclone-feature run, and numerically identical to "
        "the published run below -- same seed, same panel, same features. It is "
        "superseded for a mechanical reason only: it was written under "
        "CRISK_TAG=cyc, and `v2_models`, `export_v2` and `make_report` hardcode "
        "the `_run` filename suffix in about twenty places, so no downstream "
        "tool can read it. Kept because two independent invocations agreeing to "
        "five decimals is worth more than the disk it costs."),
    "20261108T050140Z-run": (
        "The published run. Cold on 44 features: the 21-column cyclone block "
        "built from the NHC a-deck forecast as it was issued, and the three CDC "
        "SVI themes removed. Hurricane AP goes from 0.15570 to 0.53294, which "
        "is 1.63x to 5.59x over the best single feature; flood barely moves, "
        "0.26136 to 0.26190.\n\n"
        "Read the hurricane gain with care. `cyc_in34` is NHC's *forecast* "
        "34-kt wind radius and `y_storm` comes from HURDAT2's *best-track* "
        "34-kt field, so part of that 5.59x is NHC's own forecast skill "
        "re-expressed rather than skill this model added. The ablation puts the "
        "cyclone block at +242% on hurricane and essentially nil on flood, so "
        "what the block buys is almost entirely the hazard it describes.\n\n"
        "Inherited defects, unchanged: the five forward-rain features still "
        "describe rain falling after the prediction date, and the tree is dirty "
        "at 5c356632. No transformer -- both pooled TFTs still need a GPU rerun "
        "against this panel, so every transformer figure is marked withdrawn "
        "rather than carried across two feature sets.\n\n"
        "One provenance repair came out of this run. `refdata/cone_radii.json` "
        "had never been written to disk, so `cyclone_features.cone_radii` "
        "re-measured the track-error table on every panel build and the cone "
        "margins were reproducible only because that measurement happens to be "
        "deterministic. The table is now frozen, and it was verified first: five "
        "of its seven knots appear verbatim in the panel, and the panel's "
        "minimum cone radius of 149.043949 km is exactly the interpolation of "
        "the other two at tau 30h, the shortest lead a 1-3 day window reaches."),
    "20261108T225052Z-run": (
        "The Colab run that restored the transformer to the comparison: XGBoost "
        "boosters byte-identical to the run above, plus pooled and per-lead "
        "TFTs trained cold on the same panel (flood 0.21066 vs 0.26190, storm "
        "0.40705 vs 0.53294, both to XGBoost). Published as the dashboard pin "
        "because, unlike its parent, it is committed and fetchable.\n\n"
        "Its manifest records an empty git commit -- the Colab session had no "
        "clone to read -- which is what made `self_contained` fail on empty "
        "commits thereafter. Superseded by the HPC run below."),
    "20261115T174654Z-hpc-full": (
        "The first HPC run, and the current published pin. Slurm job 1568242 "
        "on CREST (crest-g002, one H100 NVL), 23h05m wall, cold on the "
        "sha-verified Nov 7 panel with the full 44-feature set, 3-day embargo, "
        "storm labels capped at 2024. XGBoost and TFT trained in the same "
        "sealed run, honest git commit b5a23ed3, dirty=false.\n\n"
        "XGBoost wins pooled on both hazards (flood 0.26500 vs 0.20050, storm "
        "0.54115 vs 0.39970) and five of six exact leads; the TFT is ahead "
        "only at flood lead 2 (0.14895 vs 0.13004), a gap within episode-level "
        "noise at n~65 flood episodes. Every lead beats its best "
        "single-feature baseline by 1.20x to 2.78x.\n\n"
        "Numbers differ from the Colab parent at the third decimal (xgboost "
        "3.2.0 vs 3.4.1, different hardware); every booster was re-verified to "
        "reproduce its recorded AP from the run directory alone."),
}


def start_of(j):
    """Whether a run actually continued a checkpoint, not whether it was allowed to.

    `params.warm_start` is only the permission, and it defaults to on, so it
    reads true for any run not launched with CRISK_COLD=1. `warm_parents`
    records the event: xgb_fit writes an entry only after a checkpoint loads and
    its feature count matches. The two disagree for every run that changes
    FEATS, because a checkpoint with a different layout is skipped -- which is
    how this column came to call the cyclone run warm when all eight of its
    boosters logged "cold start" and its manifest records self_contained.
    """
    if "warm_parents" in j:
        return "warm" if j["warm_parents"] else "cold"
    # Older runs carry no such record, so the permission flag is all there is.
    # The question mark marks it as an intent rather than an outcome.
    return "warm?" if (j.get("params") or {}).get("warm_start") else "cold"


def rows():
    for d in sorted(RUNS.iterdir()):
        man = d / "manifest.json"
        if not d.is_dir() or not man.exists():
            continue
        j = json.loads(man.read_text())
        if j.get("smoke"):
            continue
        p, g = j.get("params", {}), j.get("git", {})
        pan, hl = j.get("panel", {}), j.get("headline", {})
        yield {
            "run_id": j["run_id"],
            # A run killed before CELL 11 never writes finished_utc. Say so
            # rather than printing its start time as if it completed.
            "finished": (j.get("finished_utc") or "did not finish")[:19],
            "env": j.get("env", "?"),
            "host": j.get("host", "?"),
            "minutes": round((j.get("wall_seconds") or 0) / 60, 1),
            "commit": (g.get("commit") or "")[:8] + ("*" if g.get("dirty") else ""),
            "panel_sha": (pan.get("sha256") or "")[:10],
            "rows": pan.get("rows_as_read"),
            "counties": pan.get("counties_as_read"),
            "start": start_of(j),
            "embargo": p.get("embargo_days", 0),
            "seed": p.get("seed"),
            "flood": (hl.get("xgboost") or {}).get("flood"),
            "storm": (hl.get("xgboost") or {}).get("storm"),
            "leads": hl.get("xgb_leads") or {},
            "tft": bool((hl.get("tft") or {})),
        }


def num(v, digits=5):
    return "n/a" if v is None else f"{v:.{digits}f}"


def cnt(v):
    return "n/a" if v is None else f"{v:,}"


R = list(rows())
if not R:
    raise SystemExit(f"no non-smoke runs found under {RUNS}")

L = ["# Run ledger", "",
     "Generated by `make_ledger.py` from the manifests under `artifacts/runs/`.",
     "Do not edit by hand -- re-run the script. Smoke runs are omitted.", "",
     f"{len(R)} full runs. The published run is the last one with a `cold` "
     "start and no `*` on its commit, unless the table says otherwise.", "",
     "## Runs", "",
     "| run | finished (UTC) | where | min | commit | start | embargo | "
     "flood AP | hurricane AP |",
     "|---|---|---|---|---|---|---|---|---|"]
for r in R:
    L.append(f"| `{r['run_id']}` | {r['finished']} | {r['env']} / {r['host']} "
             f"| {r['minutes']} | `{r['commit']}` | {r['start']} "
             f"| {r['embargo'] or 'none'} d | {num(r['flood'])} "
             f"| {num(r['storm'])} |")
L += ["", "`*` on a commit means the working tree was dirty, so the code that "
      "ran is not exactly the code at that commit.", "",
      "## Lead sweep", "",
      "| run | " + " | ".join(f"flood +{k}" for k in (1, 2, 3))
      + " | " + " | ".join(f"storm +{k}" for k in (1, 2, 3)) + " |",
      "|---|" + "---|" * 6]
for r in R:
    cells = [num(r["leads"].get(f"{h}_l{k}")) for h in ("flood", "storm")
             for k in (1, 2, 3)]
    L.append(f"| `{r['run_id']}` | " + " | ".join(cells) + " |")

L += ["", "## Panel each run read", "",
      "| run | panel sha256 | rows | counties | transformer |", "|---|---|---|---|---|"]
for r in R:
    L.append(f"| `{r['run_id']}` | `{r['panel_sha']}` | {cnt(r['rows'])} "
             f"| {cnt(r['counties'])} | {'yes' if r['tft'] else 'no'} |")

L += ["", "## What each run was for", ""]
for r in R:
    L.append(f"### `{r['run_id']}`")
    L.append("")
    L.append(NOTES.get(r["run_id"], "_No note recorded. Add one to "
                                    "`make_ledger.py:NOTES`._"))
    L.append("")

missing = [r["run_id"] for r in R if r["run_id"] not in NOTES]
DEST.write_text("\n".join(L).rstrip() + "\n")
print(f"wrote {DEST}  ({len(R)} runs)")
if missing:
    print("runs with no note in make_ledger.py:NOTES: " + ", ".join(missing))
