"""Off-policy contrast with the selection of rated slots taken into account.

The estimand of s05 is the like rate among rated slots: a slot enters only if
its item received a rating. This script checks whether being rated depends on
the item chosen, and recomputes the contrast between the deployed policy and
uniform random selection with each rated slot weighted by the inverse of its
probability of being rated.

1) Rating of favoured items. A selection is "favoured" when the deployed policy
   chose its item with a probability above uniform (p_log > 1/K). Within each
   participant and position in the plan, the share of selections that received a
   rating is compared between favoured and other selections (Mantel-Haenszel
   weights), with a 95% cluster bootstrap over participants.
2) Weighted contrast. P(rated) is estimated for each participant, position in
   the plan and favoured flag, shrunk towards the rate of the position and flag
   and that towards the rate of the position (pseudo-count 10). Both the
   on-policy mean and the SNIPS value of uniform selection are weighted by
   1 / P(rated). The model of P(rated) is refitted in every bootstrap draw.

Sources, read only: intervention_mab_samples.csv of the pre-migration snapshot
and feature_audit.csv of the main snapshot, loaded as in s05.
Output: results/offpolicy_rated_selection.json and the macros listed at the end.
"""
import json

import numpy as np
import pandas as pd

from common import SNAPSHOT, RES_DIR, reg

BOOT = 5000
SEED = 20261015
ALPHA = 10                                   # pseudo-count of the shrinkage of P(rated)
POSITION_BINS = [-1, 0, 1, 2, 4, 7, 14, 1000]

# ---------------------------------------------------------------- data, as in s05, without restricting to rated slots
src = pd.read_csv(SNAPSHOT.parent / "2026_06_23_13_40" / "intervention_mab_samples.csv")
sm = src["sample"].apply(json.loads)
src["p_log"] = sm.apply(lambda d: float(d["p_log"]))
src["K"] = sm.apply(lambda d: len(dict.fromkeys(c["rec_id"] for c in d.get("candidates", []))))
src["rec"] = src["selected_rec_ids"].str.strip("{}")
fa = pd.read_csv(SNAPSHOT / "feature_audit.csv").rename(columns={"rec_id": "rec"})
a = src.merge(fa[["user_id", "slot_idx", "rec", "reward"]], on=["user_id", "slot_idx", "rec"], how="left")
a = a[(a.K >= 2) & (a.p_log > 0)].copy()
a["rated"] = a.reward.notna().astype(int)
n_selections, rated_share = len(a), a.rated.mean()
a = a[a.user_id.isin(a[a.rated == 1].user_id.unique())].reset_index(drop=True)
a["fav"] = (a.p_log > 1 / a.K).astype(int)
a["pos"] = pd.cut(a.slot_idx, POSITION_BINS, labels=False)

users = a.user_id.unique()
U, P = len(users), len(POSITION_BINS) - 1
UI = a.user_id.map({u: i for i, u in enumerate(users)}).values
PI, FAV, RATED = a.pos.values, a.fav.values, a.rated.values
LIKE = (a.reward > 0).astype(float).values
W = ((1 / a.K) / a.p_log).values             # importance weight of uniform selection
assert RATED.sum() == 819 and U == 63        # the analysis set of s05


def favoured_difference(m):
    """Within participant and position: rated share of favoured minus other selections."""
    num = den = 0.0
    for u in np.flatnonzero(m):
        for p in range(P):
            cell = (UI == u) & (PI == p)
            f1, f0 = cell & (FAV == 1), cell & (FAV == 0)
            n1, n0 = f1.sum(), f0.sum()
            if n1 and n0:
                wgt = m[u] * n1 * n0 / (n1 + n0)
                num += wgt * (RATED[f1].mean() - RATED[f0].mean())
                den += wgt
    return num / den


def contrasts(m):
    """Deployed minus uniform (SNIPS), unweighted and weighted by 1 / P(rated); m = multiplicity of each user."""
    mult = m[UI]
    rate_pos = np.bincount(PI, mult * RATED, P) / np.maximum(np.bincount(PI, mult, P), 1e-12)
    k = PI * 2 + FAV
    s1, n1 = np.bincount(k, mult * RATED, 2 * P), np.bincount(k, mult, 2 * P)
    rate_pos_fav = (s1 + ALPHA * np.repeat(rate_pos, 2)) / (n1 + ALPHA)
    k2 = UI * 2 * P + k
    s2, n2 = np.bincount(k2, RATED, U * 2 * P), np.bincount(k2, None, U * 2 * P)
    p_rated = (s2[k2] + ALPHA * rate_pos_fav[k]) / (n2[k2] + ALPHA)
    s = (RATED == 1) & (mult > 0)
    mm, v, w, r = mult[s], 1 / p_rated[s], W[s], LIKE[s]
    plain = (mm * r).sum() / mm.sum() - (mm * w * r).sum() / (mm * w).sum()
    weighted = (mm * v * r).sum() / (mm * v).sum() - (mm * w * v * r).sum() / (mm * w * v).sum()
    return plain, weighted


ones = np.ones(U)
fav_est = favoured_difference(ones)
plain_est, weighted_est = contrasts(ones)
rng = np.random.default_rng(SEED)
draws = [np.bincount(rng.integers(0, U, U), minlength=U).astype(float) for _ in range(BOOT)]
fav_boot = np.array([favoured_difference(m) for m in draws[:1000]])     # slower loop: 1,000 draws
con_boot = np.array([contrasts(m) for m in draws])
fav_ci = np.percentile(fav_boot, [2.5, 97.5])
plain_ci, weighted_ci = (np.percentile(con_boot[:, c], [2.5, 97.5]) for c in (0, 1))
weighted_le_zero = float(np.mean(con_boot[:, 1] <= 0))

out = dict(n_selections=int(n_selections), rated_share=float(rated_share),
           favoured_minus_other_rated=dict(est=fav_est, ci95=fav_ci.tolist(), boot_draws=1000),
           contrast_plain=dict(est=plain_est, ci95=plain_ci.tolist()),
           contrast_rating_weighted=dict(est=weighted_est, ci95=weighted_ci.tolist(),
                                         share_of_draws_le_zero=weighted_le_zero),
           boot_draws=BOOT, seed=SEED, alpha=ALPHA, position_bins=POSITION_BINS)
(RES_DIR / "offpolicy_rated_selection.json").write_text(json.dumps(out, indent=1))
reg(opSelN=f"{n_selections:,}", opSelRated=f"{100 * rated_share:.1f}\\%",
    opFavRatedDiff=f"{100 * fav_est:+.1f}", opFavRatedLo=f"{100 * fav_ci[0]:+.1f}", opFavRatedHi=f"{100 * fav_ci[1]:+.1f}",
    opRwGain=f"{weighted_est:+.3f}", opRwLo=f"{weighted_ci[0]:+.3f}", opRwHi=f"{weighted_ci[1]:+.3f}",
    opRwLeZero=f"{100 * weighted_le_zero:.1f}\\%")
print(json.dumps(out, indent=1))
