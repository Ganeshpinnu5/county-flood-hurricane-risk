"""Episode-block bootstrap, shared by the audits that report an interval.

The test set holds hundreds of thousands of county-days but nowhere near that
many observations: one cyclone writes a positive row into every county it
crosses on every day it is near. Resampling rows would treat those as
independent and return an interval far tighter than the data supports, so the
period is cut into contiguous blocks -- one per weather episode -- and whole
blocks are resampled.

This lives in its own module because more than one audit quotes an interval,
and two copies of this logic that drifted apart would produce intervals that
look comparable and are not.
"""
import numpy as np
from sklearn.metrics import average_precision_score

# Onset dates further apart than this belong to different weather systems, so a
# cluster below it is one event seen from several counties on several days.
# Same constant as export_v2.EPISODE_GAP_DAYS.
GAP_DAYS = 7


def blocks(test, onset_col, gap_days=GAP_DAYS):
    """Block id per test row: one block per episode, cut in the quiet gaps.

    Cutting at the midpoint of each quiet gap rather than at the episode's own
    edges keeps every county-day in exactly one block, so a bootstrap draw has
    roughly the prevalence of the test set instead of a prevalence set by how
    many quiet days happened to be excluded.
    """
    dates = np.sort(test.loc[test[onset_col] == 1, "DATE"].unique())
    if len(dates) == 0:
        return None, 0
    gap = np.diff(dates) / np.timedelta64(1, "D")
    breaks = np.where(gap > gap_days)[0]
    cuts = dates[breaks] + (dates[breaks + 1] - dates[breaks]) / 2
    return np.searchsorted(cuts, test["DATE"].to_numpy()), len(cuts) + 1


def boot(y, p, blk, n_blocks, rng, draws):
    """AP over `draws` resamples of whole blocks."""
    idx = [np.flatnonzero(blk == b) for b in range(n_blocks)]
    ys = [y[i] for i in idx]
    ps = [p[i] for i in idx]
    out = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, n_blocks, n_blocks)
        yd = np.concatenate([ys[b] for b in pick])
        if yd.sum() == 0:          # a draw that caught no event has no AP
            out[d] = np.nan
            continue
        out[d] = average_precision_score(yd, np.concatenate([ps[b] for b in pick]))
    return out


def pct(a):
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [2.5, 97.5])
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                lo95=float(lo), hi95=float(hi), n_usable=int(len(a)))
