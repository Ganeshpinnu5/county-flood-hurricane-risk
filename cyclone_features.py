"""County-day cyclone features from the official forecast that was on the wire.

labels.py builds the storm *label* from the post-season best track, which is
correct: the label should be what happened. This builds the *features* from the
a-deck forecasts instead, because a feature may only contain what was knowable
at the origin date. The two files are deliberately fed from different sources.

The advisory a row gets
----------------------
Local day t runs 06Z to 06Z, midnight Central, matching the panel's labels and
the rainfall features. The advisory chosen for origin date t is the most recent
one with a synoptic time at or before 00Z on day t -- the same cut the GEFS
rainfall arm uses, and for the same reason: 00Z products are on the wire by
about 03Z, which is before local day t begins. The 06Z advisory of day t is
issued at 3am local and is deliberately *not* used, so no feature here can
contain information that postdates the start of the day it describes.

That makes the forecast up to 24 hours stale, which is the honest cost of a
clean cut and is what a system running once a morning would actually have.

Features
--------
One set per origin date, describing the single most threatening storm -- the one
whose forecast track comes closest to the county across the three-day window.
Picking per county rather than per storm matters in a double-storm season: the
county cares about whichever one is coming for it.

Distances are great-circle kilometres from the county centroid to the forecast
track, minimised over the hours of each local day, with intensity, wind radii
and motion read at that closest hour. Rows with no active storm carry a
`cyc_active` of 0 and nulls elsewhere -- not a large sentinel distance, because
"no storm exists" and "a storm is 4,000 km away" are different statements and a
sentinel would quietly merge them.

The cone
--------
`cyc_cone` asks whether the county falls inside the forecast track cone, which
needs a track-error radius per lead. It is measured here rather than taken from
NHC's published table: the OFCL forecasts in the a-deck are verified against the
best track at their valid time, and the radius is the 67th percentile error at
that lead. Computed over CONE_YEARS only and then frozen, so no test-period
information reaches a training row -- the same treatment the rainfall
climatology gets. NHC's own table is a 5-year rolling average that would differ
per row and partly encode errors measured after the training window closed.
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd

import labels as L

ADECK = pathlib.Path(os.environ.get("CRISK_ADECK_PARQUET",
                                    "artifacts/nhc_adeck.parquet"))
LEADS = (1, 2, 3)
# Forecast hours worth interpolating. The furthest hour any lead-3 local day can
# reach is 102, so carrying 120 costs nothing and tolerates a stale advisory.
MAX_TAU = 120
# Local day boundary: 06Z is midnight Central Standard.
DAY_OFFSET = pd.Timedelta(hours=6)
# Oldest advisory still usable at an origin date. One day, so a storm that
# stopped getting advisories yesterday is gone by tomorrow.
MAX_STALE = pd.Timedelta(days=1)
# Error percentile defining the cone. NHC's cone is drawn to contain the track
# about two thirds of the time, so 67 reproduces its meaning.
CONE_PCT = 67
CONE_TAUS = (12, 24, 36, 48, 72, 96, 120)
# Training years only; must match pipeline.PARAMS["years"]["train"], since a
# cone measured over 2015 onward would carry validation and test errors into a
# training row.
CONE_YEARS = tuple(int(y) for y in
                   os.environ.get("CRISK_CONE_YEARS", "2000,2014").split(","))
# A forecast track that never comes within this of the panel cannot produce a
# feature that matters, and dropping those advisories is most of the work saved.
FAR_KM = 2500.0
# The measured cone table, frozen to disk. The live scorer has no best track to
# verify forecasts against, and a radius re-measured per environment would make
# two county-days scored by the same booster incomparable.
CONE_JSON = pathlib.Path(os.environ.get(
    "CRISK_CONE_JSON", pathlib.Path(__file__).parent / "refdata/cone_radii.json"))
KM_PER_NM = L.NM_TO_KM
QUADS = L.QUADS
# The wind speeds an a-deck carries a forecast radius for. These describe the
# forecast wind field and are features, not label thresholds: the label is the
# single 34-kt one in labels.py.
TIERS = (34, 50, 64)


def load_adeck():
    if not ADECK.exists():
        raise SystemExit(f"{ADECK} not found -- run fetch_nhc_adeck.py first")
    a = pd.read_parquet(ADECK)
    return a[a["tau"].between(0, MAX_TAU)].reset_index(drop=True)


def cone_radii(adeck=None, best_track=None):
    """67th-percentile official track error by lead, from training years only.

    Read from CONE_JSON when it exists and measured once when it does not, so
    the radii a row was built with are the radii every later row is built with.
    Verified against the best track at the forecast's own valid time, which is
    the standard NHC verifies against. Interpolated rather than taken per tau at
    use time, because the deck carries taus the cone table has no entry for.
    """
    if CONE_JSON.exists():
        d = json.loads(CONE_JSON.read_text())
        print(f"cone radius: frozen table from {CONE_JSON} "
              f"({d['pct']}th pct, {d['years'][0]}-{d['years'][1]})")
        return {int(k): float(v) for k, v in d["km_by_tau"].items()}
    if adeck is None or best_track is None:
        raise SystemExit(
            f"{CONE_JSON} not found and no a-deck or best track given to "
            f"measure one from -- build the panel first, which freezes it")
    bt = best_track[["storm_id", "ts", "lat", "lon"]].rename(
        columns={"ts": "valid", "lat": "bt_lat", "lon": "bt_lon"})
    f = adeck[adeck["tau"].isin(CONE_TAUS)].copy()
    f["valid"] = f["issued"] + pd.to_timedelta(f["tau"], unit="h")
    f = f[f["issued"].dt.year.between(*CONE_YEARS)]
    m = f.merge(bt, on=["storm_id", "valid"], how="inner")
    if m.empty:
        raise SystemExit("no OFCL forecast verified against the best track -- "
                         "check storm_id formatting between the two sources")
    m["err"] = L._haversine(m["bt_lat"].to_numpy(), m["bt_lon"].to_numpy(),
                            m["lat"].to_numpy(), m["lon"].to_numpy())
    g = m.groupby("tau")["err"]
    table = {int(t): float(v) for t, v in
             g.quantile(CONE_PCT / 100.0).items() if g.size()[t] >= 30}
    if len(table) < 3:
        raise SystemExit(f"only {len(table)} leads have enough verified "
                         f"forecasts to measure a cone radius")
    print(f"cone radius, {CONE_PCT}th pct OFCL error, "
          f"{CONE_YEARS[0]}-{CONE_YEARS[1]} ({len(m):,} verified forecasts):")
    for t in sorted(table):
        print(f"  tau {t:>3}h  {table[t]:>6.0f} km  "
              f"n={int(g.size()[t]):,}")
    CONE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONE_JSON.write_text(json.dumps({
        "pct": CONE_PCT, "years": list(CONE_YEARS),
        "n_verified": int(len(m)),
        # Full precision, not rounded for looks: a rebuilt panel has to
        # reproduce the cone margins of the one the booster was fitted on.
        "km_by_tau": {str(t): table[t] for t in sorted(table)},
    }, indent=1) + "\n")
    print(f"  froze {CONE_JSON}")
    return table


def _advisory_window(times, cut):
    """Index range of advisories a row with this cut may use.

    One function because two callers have to agree on it exactly: `build` reads
    the window to pick a track, and `live` reads it to decide whether a county
    with no track is quiet or merely beyond the forecast.
    """
    hi = np.searchsorted(times, np.datetime64(cut), side="right")
    lo = np.searchsorted(times, np.datetime64(cut - MAX_STALE), side="left")
    return lo, hi


def _hourly(g):
    """One advisory's forecast positions resampled to hourly.

    The deck gives 12-hourly steps past day one, and a storm moving at 15 kt
    covers 330 km between them, so testing only the archived taus would step
    straight over counties the forecast track passes through.
    """
    g = g.set_index(g["tau"].astype(int)).drop(columns="tau").sort_index()
    full = pd.RangeIndex(int(g.index.min()), int(g.index.max()) + 1, name="tau")
    return g.reindex(full).interpolate(method="index", limit_direction="both")


def build(counties, adeck=None, best_track=None, cone=None, origins=None,
          now=None):
    """Per county-day cyclone features. `counties` is county_fips / lat / lon.

    Returns one row per county and origin date on which some storm had an
    official forecast. Days with no active cyclone are absent and are filled by
    the caller, because the panel knows its own date range and this does not.

    `origins` names the origin dates explicitly instead of deriving them from
    the span of the deck. The training path leaves it unset, since the panel's
    dates and the deck's cover each other. A live scorer cannot: it has to score
    dates the deck does not reach yet, and deriving the range would silently
    return nothing for them.

    `now` caps the advisory cut at the present instant, and only a live caller
    passes it. For an origin date in the future the rule "the latest advisory at
    or before 00Z on the origin date" selects an advisory that has not been
    issued, so the cut falls back to the freshest one that exists. Note which way
    that leans: scoring tomorrow uses today's 18Z advisory if it is out, which is
    both fresher than the 00Z one and still before tomorrow's cut, so it stays
    train-equivalent. The staleness gate keeps measuring from the cut, so a storm
    that stopped getting advisories still drops out rather than being carried
    forward forever.
    """
    a = load_adeck() if adeck is None else adeck
    if cone is None:
        cone = cone_radii(a, best_track)
    ctaus = np.array(sorted(cone))
    cvals = np.array([cone[t] for t in ctaus])

    clat = counties["lat"].to_numpy()
    clon = counties["lon"].to_numpy()
    cfips = counties["county_fips"].to_numpy()
    nc = len(counties)

    rcols = [f"r{t}_{q}" for t in TIERS for q in QUADS]
    cols = ["tau", "lat", "lon", "vmax", "bearing", "fwd_kt"] + rcols

    # Every advisory's hourly track, once. The same advisory is the chosen one
    # for at most one origin date, but assembling it inside the date loop would
    # re-interpolate the same storm for every county.
    tracks = {}
    for (sid, issued), g in a.groupby(["storm_id", "issued"], sort=False):
        if len(g) < 2:
            continue
        tracks[(sid, issued)] = _hourly(g[cols].astype(float))

    # Which advisory each origin date gets: per storm, the latest issued at or
    # before 00Z that morning. The window is left-bounded at MAX_STALE so a
    # storm that has dissipated drops out instead of being carried forward on an
    # advisory nobody would still have been looking at.
    if not tracks:
        # No storm has an official forecast in range. Real, and the common case
        # out of season: the caller fills every county-day as quiet.
        if origins is None:
            raise SystemExit("no advisory carried an official forecast track")
        return pd.DataFrame(columns=["county_fips", "DATE"] + FEATURES)

    adv = (pd.DataFrame([{"storm_id": k[0], "issued": k[1]} for k in tracks])
           .sort_values("issued").reset_index(drop=True))
    times = adv["issued"].to_numpy()
    derived = origins is None
    if derived:
        origins = pd.date_range(adv["issued"].min().normalize(),
                                adv["issued"].max().normalize(), freq="D")
    out = []
    for origin in origins:
        cut = origin if now is None else min(origin, pd.Timestamp(now))
        lo, hi = _advisory_window(times, cut)
        if hi <= lo:
            continue
        # adv is sorted by issue time, so last() per storm is the freshest.
        pick = adv.iloc[lo:hi].groupby("storm_id").last()
        rows = []
        for sid, r in pick.iterrows():
            h = tracks[(sid, r["issued"])]
            # Forecast hour -> local day index relative to the origin date.
            valid = r["issued"] + pd.to_timedelta(h.index.to_numpy(), unit="h")
            lead = ((valid - DAY_OFFSET).normalize() - origin).days
            sel = np.isin(lead, LEADS)
            if not sel.any():
                continue
            hh = h[sel]
            lead = lead[sel]
            plat = hh["lat"].to_numpy()
            plon = hh["lon"].to_numpy()
            # nc x nh distances. Trimmed first on the track's closest approach
            # to any county, which drops the Cape Verde half of every season.
            dist = L._haversine(clat[:, None], clon[:, None],
                                plat[None, :], plon[None, :])
            if dist.min() > FAR_KM:
                continue
            rows.append((sid, hh, lead, dist))
        if not rows:
            continue

        # Per county, the storm with the single closest forecast approach.
        best = np.full(nc, np.inf)
        which = np.full(nc, -1)
        for i, (_, _, _, dist) in enumerate(rows):
            d = dist.min(axis=1)
            take = d < best
            best[take] = d[take]
            which[take] = i

        rec = {c: np.full(nc, np.nan) for c in FEATURES}
        rec["cyc_active"] = np.ones(nc)
        rec["cyc_n_active"] = np.full(nc, float(len(rows)))
        for i, (sid, hh, lead, dist) in enumerate(rows):
            sl = which == i
            if not sl.any():
                continue
            _fill(rec, sl, hh, lead, dist[sl], clat[sl], clon[sl], ctaus, cvals)
        f = pd.DataFrame(rec)
        f.insert(0, "DATE", origin)
        f.insert(0, "county_fips", cfips)
        out.append(f)

    if not out:
        # Over a derived range this is a contradiction -- the range came from the
        # advisories themselves, so at least one date had to produce a row.
        if derived:
            raise SystemExit("no origin date produced a cyclone feature")
        return pd.DataFrame(columns=["county_fips", "DATE"] + FEATURES)
    return pd.concat(out, ignore_index=True)


def _fill(rec, sl, hh, lead, dist, lat, lon, ctaus, cvals):
    """Write one storm's features for the counties it is closest to."""
    tau = hh.index.to_numpy().astype(float)
    vmax = hh["vmax"].to_numpy()
    brg = hh["bearing"].to_numpy()
    fwd = hh["fwd_kt"].to_numpy()
    plat = hh["lat"].to_numpy()
    plon = hh["lon"].to_numpy()

    # Closest approach across the whole window, and within each lead day.
    jmin = np.argmin(dist, axis=1)
    rec["cyc_dist_min"][sl] = dist[np.arange(len(jmin)), jmin]
    rec["cyc_lead_closest"][sl] = lead[jmin]
    rec["cyc_vmax_close"][sl] = vmax[jmin]
    rec["cyc_vmax_max"][sl] = np.nanmax(vmax) if len(vmax) else np.nan
    rec["cyc_fwd_kt"][sl] = fwd[jmin]
    # Motion as a direction pair. A single bearing in degrees puts 359 and 1 at
    # opposite ends of every split the tree can make.
    rad = np.deg2rad(brg[jmin])
    rec["cyc_bearing_sin"][sl] = np.sin(rad)
    rec["cyc_bearing_cos"][sl] = np.cos(rad)

    for k in LEADS:
        m = lead == k
        if m.any():
            rec[f"cyc_dist{k}"][sl] = dist[:, m].min(axis=1)

    # The wind radius facing the county, which is the one that decides exposure.
    # Quadrant is taken from the county's bearing as seen from the storm, the
    # same convention labels.py uses so the feature and the label agree on which
    # side of the storm the county is on.
    cb = L._bearing(plat[jmin], plon[jmin], lat, lon)
    qi = np.select([cb < 90, cb < 180, cb < 270], [0, 1, 2], default=3)
    for t in TIERS:
        r = np.stack([hh[f"r{t}_{q}"].to_numpy()[jmin] for q in QUADS])
        facing = r[qi, np.arange(len(qi))] * KM_PER_NM
        rec[f"cyc_r{t}"][sl] = facing
        # Inside the forecast wind field at the closest approach. Null radius
        # means the forecast gave the storm no such wind field, which is a
        # negative rather than an unknown, so it lands as 0.
        rec[f"cyc_in{t}"][sl] = np.where(
            np.isnan(facing), 0.0,
            (rec["cyc_dist_min"][sl] <= facing).astype(float))

    # Cone at the lead of closest approach, interpolated from the frozen table.
    cone = np.interp(tau[jmin], ctaus, cvals)
    rec["cyc_cone_km"][sl] = cone
    rec["cyc_cone_margin"][sl] = rec["cyc_dist_min"][sl] - cone
    rec["cyc_in_cone"][sl] = (rec["cyc_dist_min"][sl] <= cone).astype(float)


FEATURES = [
    "cyc_active", "cyc_n_active",
    "cyc_dist_min", "cyc_dist1", "cyc_dist2", "cyc_dist3",
    "cyc_lead_closest", "cyc_vmax_close", "cyc_vmax_max", "cyc_fwd_kt",
    "cyc_bearing_sin", "cyc_bearing_cos",
    "cyc_r34", "cyc_r50", "cyc_r64",
    "cyc_in34", "cyc_in50", "cyc_in64",
    "cyc_cone_km", "cyc_cone_margin", "cyc_in_cone",
]
# Features whose absence means "no", not "unknown". A county-day with no active
# cyclone is not a county-day whose exposure is in doubt, and leaving these null
# would let the booster route 99% of rows down a missing-value branch that
# carries no information.
ZERO_WHEN_QUIET = ("cyc_active", "cyc_n_active",
                   "cyc_in34", "cyc_in50", "cyc_in64", "cyc_in_cone")

# The other fifteen are null on 85-95% of county-days, and a dense model has no
# missing-value branch to route them down. XGBoost does, which is why it never
# needed this. The TFT filled every feature with 0.0, and 0.0 on a distance
# says a hurricane is directly overhead -- the value 85.6% of all county-days
# would have carried into the encoder.
#
# Each sentinel is chosen to be wrong only in the safe direction: an absent
# storm reads as far away and windless, never as near. FAR_KM sits beyond the
# largest separation the panel contains (6,720 km) so it cannot be mistaken for
# a real distance, and `cyc_active` is already 0 on exactly these rows, so a
# model has an explicit gate rather than having to infer the sentinel.
FAR_KM = 10000.0
NEURAL_FILL = {
    "cyc_dist_min": FAR_KM, "cyc_dist1": FAR_KM, "cyc_dist2": FAR_KM,
    "cyc_dist3": FAR_KM,
    # Negative margin means inside the cone, so the quiet sentinel has to be
    # large and positive. Zero here would read as "on the cone boundary".
    "cyc_cone_margin": FAR_KM,
    # No storm means no cone was drawn, which is the one quantity in this block
    # that zero describes correctly.
    "cyc_cone_km": 0.0,
    "cyc_lead_closest": 0.0,
    "cyc_vmax_close": 0.0, "cyc_vmax_max": 0.0, "cyc_fwd_kt": 0.0,
    # (0, 0) is off the unit circle every real bearing lies on, so it reads as
    # "no bearing" rather than as due east.
    "cyc_bearing_sin": 0.0, "cyc_bearing_cos": 0.0,
    "cyc_r34": 0.0, "cyc_r50": 0.0, "cyc_r64": 0.0,
}


def fill_dense(d):
    """Impute the cyclone block for a model that cannot represent a null.

    Call this *before* any blanket fillna, so the cyclone columns are already
    numbers by the time the general-purpose fill runs and cannot be swept to a
    value that means something else.
    """
    missed = set(FEATURES) - set(ZERO_WHEN_QUIET) - set(NEURAL_FILL)
    if missed:
        # A feature added to FEATURES without a sentinel would otherwise fall
        # through to the caller's fillna and silently acquire 0.0.
        raise SystemExit(f"no dense sentinel declared for {sorted(missed)}")
    for c in ZERO_WHEN_QUIET:
        if c in d.columns:
            d[c] = d[c].fillna(0.0)
    for c, v in NEURAL_FILL.items():
        if c in d.columns:
            d[c] = d[c].fillna(v)
    return d


def attach(panel, counties, best_track):
    """Merge cyclone features onto a panel, filling the quiet days."""
    f = build(counties, best_track=best_track)
    out = panel.merge(f, on=["county_fips", "DATE"], how="left")
    for c in ZERO_WHEN_QUIET:
        out[c] = out[c].fillna(0.0)
    return out


def live(counties, origins, adeck=None):
    """Cyclone features for the origin dates a live scorer is about to score.

    Returns one row per county per origin date, so the caller merges a complete
    frame instead of having to know which dates a storm reached.

    An origin with no track gets one of two different answers, and conflating
    them is the trap here. If no storm was under advisory at the cut, the day is
    quiet and the ZERO_WHEN_QUIET features are 0, exactly as in training. If a
    storm *was* under advisory but no forecast hour reached the lead window, the
    day is unknown and every cyclone feature stays null. OFCL stops at 120
    hours, so the furthest date a live scorer wants can sit past the end of the
    forecast -- and filling that with zero would state "no cyclone" on the days
    a hurricane is approaching. The caller drops those county-days rather than
    scoring them, because `cyc_active` was never null in training and the
    booster has no learned default for it.

    The cone radii come off the frozen table rather than being re-measured: a
    live scorer has no best track to verify forecasts against, and a radius
    measured per environment would make two county-days scored by the same
    booster incomparable.
    """
    import fetch_nhc_adeck

    a = fetch_nhc_adeck.live_adeck() if adeck is None else adeck
    now = pd.Timestamp.now("UTC").tz_localize(None)
    origins = pd.to_datetime(list(origins))
    f = build(counties, adeck=a, cone=cone_radii(), origins=origins, now=now)
    grid = counties[["county_fips"]].merge(pd.DataFrame({"DATE": origins}),
                                           how="cross")
    if f.empty:
        # Out of season every county-day is quiet, and merging an empty frame
        # would only argue with itself about column dtypes.
        for c in FEATURES:
            grid[c] = np.nan
        out = grid
    else:
        out = grid.merge(f, on=["county_fips", "DATE"], how="left")

    times = np.sort(a["issued"].to_numpy()) if len(a) else np.empty(
        0, dtype="datetime64[ns]")
    quiet = []
    for o in origins:
        lo, hi = _advisory_window(times, min(o, now))
        if hi <= lo:
            quiet.append(o)
    q = out["DATE"].isin(quiet)
    for c in ZERO_WHEN_QUIET:
        out.loc[q, c] = out.loc[q, c].fillna(0.0)
    return out
