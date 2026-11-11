"""S05 — Off-policy evaluation of alternative content policies on the pilot logs.

This is the data-grounded counterpart to the simulator experiments: it needs no
generative model of users, only the logged propensities. The deployed policy
recorded them from the first day of the pilot (`mc_propensity_n=1000` in
selector.py), but the dual-rating migration of 24 June 2026 truncated the table
in the live database. The history is intact in the last pre-migration snapshot,
which is where this script reads the propensities from.

Estimand: expected P(like) per rated slot under a target policy, on the
subpopulation of slots that received a rating. Rating is itself a behaviour, so
this is a conditional estimand, not the unconditional value of the policy; the
same conditioning applies to the on-policy reference, so the contrast is
between policies within the same subpopulation.

Estimators: IPS, self-normalised IPS (primary), and doubly robust with a
user-fold cross-fitted reward model. Uncertainty by cluster bootstrap over
users, matching the rest of the compendium.
"""
import json

import numpy as np
import pandas as pd

from common import SNAPSHOT, RES_DIR, sigmoid, reg, savefig, C, pct, eb_kappa
import matplotlib.pyplot as plt

PRE_MIGRATION = SNAPSHOT.parent / "2026_06_23_13_40"
BOOT = 5000
RNG = np.random.default_rng(20260814)

# ---------------------------------------------------------------- load
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

print(f"slot con propensity nello snapshot pre-migrazione: {len(src):,}")
print(f"di cui con reward osservato e menu >= 2: {len(j)}  ({j.user_id.nunique()} utenti)")
print(f"K mediano {j.K.median():.0f}; p_log mediana {j.p_log.median():.3f}, min {j.p_log.min():.3f}")

# ---------------------------------------------------------------- target policies
def pi_uniform(row):
    """Uniform over the menu."""
    return {c: 1.0 / row.K for c in row.cands}


def pi_avoid_disliked(row, disliked):
    """Deployed-like but never re-serve an item this user disliked (E3's V4).
    Modelled as: the deployed propensities renormalised over the allowed set."""
    bad = disliked.get(row.user_id, set())
    allowed = [c for c in row.cands if c not in bad] or row.cands
    p = {c: float(row.p_by_item.get(c, 0.0)) for c in allowed}
    tot = sum(p.values())
    return {c: v / tot for c, v in p.items()} if tot > 0 else {c: 1.0 / len(allowed) for c in allowed}


def pi_popularity(row, pop):
    """Deterministic: the menu item with the highest cross-fitted popularity."""
    best = max(row.cands, key=lambda c: pop.get(c, np.nan) if c in pop else -np.inf)
    return {best: 1.0}


# cross-fitted popularity and reward model, by user folds (no leakage)
users_ = j.user_id.unique()
fold = {u: i % 5 for i, u in enumerate(RNG.permutation(users_))}
j["fold"] = j.user_id.map(fold)

# participant term of the reward model: the participant's own ratings given before the
# slot (j is sorted by rating time), so no slot informs its own prediction. A user-fold
# estimate cannot be used here, because the participant of a slot is never in the
# training folds of that slot and the term would always equal the fold mean.
j["past_sum"] = j.groupby("user_id")["r"].cumsum() - j["r"]
j["past_n"] = j.groupby("user_id").cumcount()
pop_by_fold, rmodel_by_fold = {}, {}
for f in range(5):
    tr = j[j.fold != f]
    gm = tr.r.mean()
    # shrinkage weights estimated by empirical Bayes on the training folds
    # (common.eb_kappa), separately for items and participants; until
    # 5 November 2026 both were fixed at 20 ratings
    k_item = eb_kappa(tr.rec.values, tr.r.values)
    k_user = eb_kappa(tr.user_id.values, tr.r.values)
    g = tr.groupby("rec")["r"].agg(["sum", "size"])
    pop_by_fold[f] = ((g["sum"] + k_item * gm) / (g["size"] + k_item)).to_dict()
    rmodel_by_fold[f] = (gm, pop_by_fold[f], k_user)
    print(f"  fold {f}: kappa item {k_item:.1f}, kappa participant {k_user:.1f}")


def q_hat(row, item):
    """Cross-fitted reward model: additive on the logit of shrunk item/user rates."""
    gm, pop, k_user = rmodel_by_fold[row.fold]
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    umean = (row.past_sum + k_user * gm) / (row.past_n + k_user)
    return float(sigmoid(lg(pop.get(item, gm)) + lg(umean) - lg(gm)))


# ---------------------------------------------------------------- estimators
# Weights are computed once, on the original chronological ordering, because
# the V4 target depends on each user's dislike history. The cluster bootstrap
# then resamples the (weight, reward, q) triples by user, holding that history
# fixed; it therefore reflects sampling uncertainty over users, not uncertainty
# in the reconstructed history.
def weights_for(target_fn):
    w, q_chosen, q_target = [], [], []
    disliked = {}
    for _, row in j.iterrows():
        pi = (target_fn(row, disliked) if target_fn.__code__.co_argcount == 2
              else target_fn(row))
        w.append(pi.get(row.rec, 0.0) / row.p_log)
        q_chosen.append(q_hat(row, row.rec))
        q_target.append(sum(p * q_hat(row, c) for c, p in pi.items()))
        if row.r == 0:
            disliked.setdefault(row.user_id, set()).add(row.rec)
    return np.array(w), np.array(q_chosen), np.array(q_target)


def point(w, r, qc, qt):
    return dict(ips=float((w * r).mean()),
                snips=float((w * r).sum() / w.sum()) if w.sum() > 0 else np.nan,
                dr=float((qt + w * (r - qc)).mean()),
                ess=float(w.sum() ** 2 / (w ** 2).sum()),
                wmax=float(w.max()), coverage=float((w > 0).mean()))


IDX = {u: j.index[j.user_id == u].values for u in j.user_id.unique()}
USERS = np.array(list(IDX.keys()))


def cluster_boot_diff(w, r, qc, qt):
    """Paired CI for (on-policy mean - SNIPS) over user resamples."""
    d_snips, d_dr = [], []
    for _ in range(BOOT):
        take = np.concatenate([IDX[u] for u in RNG.choice(USERS, len(USERS), replace=True)])
        wb, rb, qcb, qtb = w[take], r[take], qc[take], qt[take]
        if wb.sum() <= 0:
            continue
        d_snips.append(rb.mean() - (wb * rb).sum() / wb.sum())
        d_dr.append(rb.mean() - (qtb + wb * (rb - qcb)).mean())
    # Bonferroni over the alternatives compared with the deployed policy (8 Nov 2026)
    a = 100 * 0.05 / len(targets) / 2
    ADJ.append((np.percentile(d_snips, [a, 100 - a]), np.percentile(d_dr, [a, 100 - a])))
    return (np.percentile(d_snips, [2.5, 97.5]), np.percentile(d_dr, [2.5, 97.5]))


ON = j.r.mean()
print(f"\nvalore ON-POLICY osservato (bandit deployato): {ON:.4f}"
      f"   [{len(j)} slot, {j.user_id.nunique()} utenti]")

targets = [("uniform random", pi_uniform),
           ("avoid disliked (V4)", pi_avoid_disliked),
           ("greedy on popularity", lambda row: pi_popularity(row, pop_by_fold[row.fold]))]
R = j.r.values
rows = []
ADJ = []
for name, fn in targets:
    w, qc, qt = weights_for(fn)
    e = point(w, R, qc, qt)
    (slo, shi), (dlo, dhi) = cluster_boot_diff(w, R, qc, qt)
    rows.append(dict(policy=name, **e, diff=ON - e["snips"],
                     diff_lo=slo, diff_hi=shi, diff_dr=ON - e["dr"],
                     diff_dr_lo=dlo, diff_dr_hi=dhi))
    print(f"\n{name}")
    print(f"  IPS {e['ips']:.4f} | SNIPS {e['snips']:.4f} | DR {e['dr']:.4f}")
    print(f"  ESS {e['ess']:.0f}/{len(j)}, peso max {e['wmax']:.1f}, slot con peso>0 {e['coverage']:.0%}")
    print(f"  deployato - target: SNIPS {ON - e['snips']:+.4f} IC95 [{slo:+.4f}, {shi:+.4f}]"
          f" | DR {ON - e['dr']:+.4f} IC95 [{dlo:+.4f}, {dhi:+.4f}]")

res = pd.DataFrame(rows)
res["adj_lo"], res["adj_hi"] = [x[0][0] for x in ADJ], [x[0][1] for x in ADJ]
res["adj_dr_lo"], res["adj_dr_hi"] = [x[1][0] for x in ADJ], [x[1][1] for x in ADJ]
res.to_json(RES_DIR / "offpolicy.json", orient="records", indent=1)

u = res[res.policy == "uniform random"].iloc[0]
reg(opN=str(len(j)), opUsers=str(int(j.user_id.nunique())),
    opSlotsLogged=f"{len(src):,}",
    opOnPolicy=f"{ON:.3f}", opUniform=f"{u.snips:.3f}",
    opGain=f"{u['diff']:+.3f}", opGainLo=f"{u.diff_lo:+.3f}", opGainHi=f"{u.diff_hi:+.3f}",
    opESS=f"{u.ess:.0f}", opWmax=f"{u.wmax:.1f}")
v4 = res[res.policy == "avoid disliked (V4)"].iloc[0]
reg(opVfourValue=f"{v4.snips:.3f}", opVfourGain=f"{-v4['diff']:+.3f}",
    opVfourLo=f"{-v4.diff_hi:+.3f}", opVfourHi=f"{-v4.diff_lo:+.3f}", opVfourESS=f"{v4.ess:.0f}")
g = res[res.policy == "greedy on popularity"].iloc[0]
# deployed minus greedy popularity, same estimators and bootstrap as the others
reg(opGreedyCoverage=pct(g.coverage, 1), opGreedyIPS=f"{g.ips:.2f}", opVfourCoverage=pct(v4.coverage, 1),
    opGreedyESS=f"{g.ess:.0f}",
    opGreedyGain=f"{g['diff']:+.3f}", opGreedyLo=f"{g.diff_lo:+.3f}", opGreedyHi=f"{g.diff_hi:+.3f}",
    opGreedyGainDR=f"{g.diff_dr:+.3f}", opGreedyDRLo=f"{g.diff_dr_lo:+.3f}",
    opGreedyDRHi=f"{g.diff_dr_hi:+.3f}")
# doubly robust counterparts, quoted by the framework paper as a second estimator
reg(opGainDR=f"{u.diff_dr:+.3f}", opGainDRLo=f"{u.diff_dr_lo:+.3f}", opGainDRHi=f"{u.diff_dr_hi:+.3f}",
    opVfourGainDR=f"{-v4.diff_dr:+.3f}", opVfourDRLo=f"{-v4.diff_dr_hi:+.3f}",
    opVfourDRHi=f"{-v4.diff_dr_lo:+.3f}")
reg(opNContrasts=str(len(targets)), opVfourAdjLo=f"{-v4.adj_hi:+.3f}", opVfourAdjHi=f"{-v4.adj_lo:+.3f}",
    opVfourAdjDRLo=f"{-v4.adj_dr_hi:+.3f}", opVfourAdjDRHi=f"{-v4.adj_dr_lo:+.3f}",
    opUniformAdjLo=f"{u.adj_lo:+.3f}", opUniformAdjHi=f"{u.adj_hi:+.3f}")
print(f"\nscritto offpolicy.json; macro op* registrate")
