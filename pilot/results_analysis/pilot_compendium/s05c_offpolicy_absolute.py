"""Absolute off-policy values with bootstrap intervals (panel a of fig_offpolicy).

s05 stores the point estimates of each policy's value and the intervals of the
contrasts, but not intervals for the absolute values. Until 2 October 2026
fig_offpolicy.py computed them inside the figure script; the computation is
moved here unchanged, so that the intervals are stored and registered as
macros, and the figure reads them.

Quantity: estimated acceptance per rated slot of each policy. The deployed
policy is its on-policy mean; uniform random, the exclusion rule and greedy
popularity have a SNIPS and a doubly robust (DR) estimate. Intervals are 95%
percentile cluster-bootstrap intervals over participants (5,000 draws, seed
20261002).

Sources, read only: offpolicy.json (s05), intervention_mab_samples.csv of the
pre-migration snapshot and feature_audit.csv of the main snapshot, loaded
exactly as s05 loads them.
Output: results/offpolicy_absolute.json and the macros listed at the end.
"""
import json

import numpy as np
import pandas as pd

from common import SNAPSHOT, RES_DIR, sigmoid, reg, eb_kappa

BOOT = 5000
SEED_FOLDS = 20260814    # s05's seed; its first draw assigns the user folds
SEED_BOOT = 20261002     # bootstrap of the absolute values

op = pd.read_json(RES_DIR / "offpolicy.json").set_index("policy")

# ---------------------------------------------------------------- data, as in s05
PRE_MIGRATION = SNAPSHOT.parent / "2026_06_23_13_40"
src = pd.read_csv(PRE_MIGRATION / "intervention_mab_samples.csv")
sm = src["sample"].apply(json.loads)
src["p_log"] = sm.apply(lambda d: float(d["p_log"]))
src["cands"] = sm.apply(lambda d: [c["rec_id"] for c in d.get("candidates", [])])
src["p_by_item"] = sm.apply(lambda d: d.get("mc_propensity", {}).get("p_by_item", {}))
src["rec"] = src["selected_rec_ids"].str.strip("{}")

fa = pd.read_csv(SNAPSHOT / "feature_audit.csv")
fa["rts"] = pd.to_datetime(fa["rating_ts"], format="ISO8601", utc=True)
j = src.merge(fa[["user_id", "slot_idx", "rec_id", "reward", "rts"]].rename(columns={"rec_id": "rec"}),
              on=["user_id", "slot_idx", "rec"], how="inner")
j["cands"] = j["cands"].apply(lambda cs: list(dict.fromkeys(cs)))
j["K"] = j["cands"].apply(len)
j = j[(j.K >= 2) & (j.p_log > 0)].sort_values("rts").reset_index(drop=True)
j["r"] = (j.reward > 0).astype(float)

# user folds for the cross-fitted popularity and reward model, as in s05
rng = np.random.default_rng(SEED_FOLDS)
users = j.user_id.unique()
fold = {u: i % 5 for i, u in enumerate(rng.permutation(users))}
j["fold"] = j.user_id.map(fold)

# participant term of the reward model: the participant's own ratings given before the
# slot (j is sorted by rating time), so no slot informs its own prediction. A user-fold
# estimate cannot be used here, because the participant of a slot is never in the
# training folds of that slot and the term would always equal the fold mean.
j["past_sum"] = j.groupby("user_id")["r"].cumsum() - j["r"]
j["past_n"] = j.groupby("user_id").cumcount()
pop_by_fold, model_by_fold = {}, {}
for f in range(5):
    tr = j[j.fold != f]
    gm = tr.r.mean()
    g = tr.groupby("rec")["r"].agg(["sum", "size"])
    # weights estimated as in s05 (common.eb_kappa; 20 until 5 November 2026)
    k_item = eb_kappa(tr.rec.values, tr.r.values)
    k_user = eb_kappa(tr.user_id.values, tr.r.values)
    pop_by_fold[f] = ((g["sum"] + k_item * gm) / (g["size"] + k_item)).to_dict()
    model_by_fold[f] = (gm, pop_by_fold[f], k_user)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def q_hat(row, item):
    gm, pop, k_user = model_by_fold[row.fold]
    umean = (row.past_sum + k_user * gm) / (row.past_n + k_user)
    return float(sigmoid(logit(pop.get(item, gm)) + logit(umean) - logit(gm)))


def pi_uniform(row, disliked):
    return {c: 1.0 / row.K for c in row.cands}


def pi_exclusion(row, disliked):
    bad = disliked.get(row.user_id, set())
    allowed = [c for c in row.cands if c not in bad] or row.cands
    p = {c: float(row.p_by_item.get(c, 0.0)) for c in allowed}
    tot = sum(p.values())
    return {c: v / tot for c, v in p.items()} if tot > 0 else {c: 1.0 / len(allowed) for c in allowed}


def pi_greedy(row, disliked):
    pop = pop_by_fold[row.fold]
    best = max(row.cands, key=lambda c: pop[c] if c in pop else -np.inf)
    return {best: 1.0}


def weights(target):
    """Importance weight, q(chosen item) and q under the target, per slot."""
    w, qc, qt = [], [], []
    disliked = {}
    for _, row in j.iterrows():
        pi = target(row, disliked)
        w.append(pi.get(row.rec, 0.0) / row.p_log)
        qc.append(q_hat(row, row.rec))
        qt.append(sum(p * q_hat(row, c) for c, p in pi.items()))
        if row.r == 0:
            disliked.setdefault(row.user_id, set()).add(row.rec)
    return np.array(w), np.array(qc), np.array(qt)


r = j.r.values
targets = {"uniform random": pi_uniform, "avoid disliked (V4)": pi_exclusion,
           "greedy on popularity": pi_greedy}
W = {name: weights(fn) for name, fn in targets.items()}

# point estimates must reproduce offpolicy.json
for name, (w, qc, qt) in W.items():
    assert np.isclose((w * r).sum() / w.sum(), op.loc[name, "snips"]), name
    assert np.isclose((qt + w * (r - qc)).mean(), op.loc[name, "dr"]), name

# cluster bootstrap over participants for the absolute values
idx = {u: j.index[j.user_id == u].values for u in users}
rng = np.random.default_rng(SEED_BOOT)
boot = {"on": []} | {(n, e): [] for n in W for e in ("snips", "dr")}
for _ in range(BOOT):
    take = np.concatenate([idx[u] for u in rng.choice(users, len(users), replace=True)])
    rb = r[take]
    boot["on"].append(rb.mean())
    for name, (w, qc, qt) in W.items():
        wb = w[take]
        boot[(name, "snips")].append((wb * rb).sum() / wb.sum() if wb.sum() > 0 else np.nan)
        boot[(name, "dr")].append((qt[take] + wb * (rb - qc[take])).mean())
ci = {k: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] for k, v in boot.items()}

# ---------------------------------------------------------------- output
rows = [dict(policy="deployed", estimator="on-policy", est=float(r.mean()), ci95=ci["on"])]
for name in W:
    for e in ("snips", "dr"):
        rows.append(dict(policy=name, estimator=e, est=float(op.loc[name, e]),
                         ci95=ci[(name, e)]))
out = dict(quantity="acceptance per rated slot", interval="95% percentile cluster "
           "bootstrap over participants", boot_draws=BOOT, boot_seed=SEED_BOOT,
           n_slots=int(len(j)), n_participants=int(j.user_id.nunique()), rows=rows)
(RES_DIR / "offpolicy_absolute.json").write_text(json.dumps(out, indent=1))

# macros: the point estimates of s05 keep their names (opOnPolicy, opUniform,
# opVfourValue); the new ones follow them
STEM = {("deployed", "on-policy"): "opOnPolicy",
        ("uniform random", "snips"): "opUniform",
        ("uniform random", "dr"): "opUniformDR",
        ("avoid disliked (V4)", "snips"): "opVfourValue",
        ("avoid disliked (V4)", "dr"): "opVfourValueDR",
        ("greedy on popularity", "snips"): "opGreedyValue",
        ("greedy on popularity", "dr"): "opGreedyValueDR"}
NEW_ESTIMATES = {"opUniformDR", "opVfourValueDR", "opGreedyValue", "opGreedyValueDR"}
macros = {"opAbsBootDraws": f"{BOOT:,}"}
for row in rows:
    stem = STEM[(row["policy"], row["estimator"])]
    if stem in NEW_ESTIMATES:
        macros[stem] = f"{row['est']:.3f}"
    macros[stem + "Lo"] = f"{row['ci95'][0]:.3f}"
    macros[stem + "Hi"] = f"{row['ci95'][1]:.3f}"
reg(**macros)
for row in rows:
    print(f"  {row['policy']:22s} {row['estimator']:9s} {row['est']:.4f} "
          f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]")
print(f"wrote {RES_DIR / 'offpolicy_absolute.json'}; registered {len(macros)} macros")
