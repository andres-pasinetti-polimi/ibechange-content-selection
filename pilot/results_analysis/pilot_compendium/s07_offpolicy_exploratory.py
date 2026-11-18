"""S07 - Exploratory analyses of the content policy on the logged selections (18 Nov 2026).

Four analyses, all on the rated slots of s05 (pairing of common.rating_selection_pairs),
all exploratory: they were specified after the main contrasts of s05 were seen.

1. Learning diagnostic. For every recommendation with at least one dislike, the
   logged probability that the deployed policy selects it, divided by the
   probability under uniform selection (1 / number of admissible items), over the
   decisions in which it was a candidate, before and after its first dislike
   (the model was pooled, so the first dislike of any participant counts),
   compared with the same ratio for the recommendations never disliked, before
   and after the same date; sign test over the disliked recommendations.
2. Contrasts by period: deployed minus uniform (SNIPS) in the first and second
   half of the rated slots, split at the median selection time.
3. Contrasts on the slots of a participant after that participant's first
   dislike, a condition that uses only the past of the participant.
4. Participant-term rule: the deployed policy, except that beyond the third
   recommendation of a plan the item is not sent when the probability of a like
   with the participant term of a25 (item and participant shares from earlier
   ratings only) is below 0.5. The rule only removes slots, so its like rate and
   its sum of rewards (+1 like, -1 dislike) are computed on the slots it keeps,
   one decision at a time.
Intervals: percentile bootstrap over participants.
"""
import json

import numpy as np
import pandas as pd

from common import RES_DIR, eb_kappa, feature_audit, pct, rating_selection_pairs, reg, selection_log

BOOT = 2000
RNG = np.random.default_rng(20261118)

src = selection_log()
sm = src["sample"].apply(json.loads)
src["cands"] = sm.apply(lambda d: list(dict.fromkeys(c["rec_id"] for c in d.get("candidates", []))))
src["K"] = src["cands"].apply(len)
src["p_log"] = sm.apply(lambda d: float(d["p_log"]))
src["pbi"] = sm.apply(lambda d: d.get("mc_propensity", {}).get("p_by_item", {}))
src["pos"] = src.groupby("plan_id").cumcount() + 1

pairs = rating_selection_pairs()
j = src.merge(pairs[["id", "reward", "rating_time", "fa_row"]], on="id")
j = j[(j.K >= 2) & (j.p_log > 0)].sort_values("rating_time").reset_index(drop=True)
j["like"] = (j.reward > 0).astype(float)
j["r"] = np.where(j.reward > 0, 1, -1)
j["w_uni"] = (1 / j.K) / j.p_log

# exclusion rule weights and the participant's dislike history, on the full sequence
dis, w_v4, after = {}, [], []
for row in j.itertuples():
    bad = dis.get(row.user_id, set())
    after.append(len(bad) > 0)
    allowed = [c for c in row.cands if c not in bad] or row.cands
    pr = {c: float(row.pbi.get(c, 0.0)) for c in allowed}
    tot = sum(pr.values())
    pi = {c: v / tot for c, v in pr.items()} if tot > 0 else {c: 1 / len(allowed) for c in allowed}
    w_v4.append(pi.get(row.rec, 0.0) / row.p_log)
    if row.like == 0:
        dis.setdefault(row.user_id, set()).add(row.rec)
j["w_v4"], j["after"] = w_v4, after


def contrasts(q):
    on = q.like.mean()
    return on - (q.w_uni * q.like).sum() / q.w_uni.sum(), (q.w_v4 * q.like).sum() / q.w_v4.sum() - on


def boot(q, fn):
    users = q.user_id.unique()
    g = {u: x for u, x in q.groupby("user_id")}
    return np.array([fn(pd.concat([g[u] for u in RNG.choice(users, len(users))])) for _ in range(BOOT)])


out = {}
half = j.sel_time.median()
for tag, q in (("First", j[j.sel_time <= half]), ("Second", j[j.sel_time > half]), ("After", j[j.after])):
    d = contrasts(q)
    b = boot(q, contrasts)
    out[tag] = dict(n=len(q), users=int(q.user_id.nunique()), dislikes=int((q.like == 0).sum()),
                    uni=d[0], uni_lo=np.percentile(b[:, 0], 2.5), uni_hi=np.percentile(b[:, 0], 97.5),
                    v4=d[1], v4_lo=np.percentile(b[:, 1], 2.5), v4_hi=np.percentile(b[:, 1], 97.5))
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[tag].items()})
f = lambda x: f"{x:+.3f}"
reg(exSplitDate=half.strftime("%-d %B %Y"))
for tag in ("First", "Second", "After"):
    o = out[tag]
    reg(**{f"ex{tag}N": str(o["n"]), f"ex{tag}Users": str(o["users"]), f"ex{tag}Dislikes": str(o["dislikes"]),
           f"ex{tag}Uni": f(o["uni"]), f"ex{tag}UniLo": f(o["uni_lo"]), f"ex{tag}UniHi": f(o["uni_hi"]),
           f"ex{tag}Vfour": f(o["v4"]), f"ex{tag}VfourLo": f(o["v4_lo"]), f"ex{tag}VfourHi": f(o["v4_hi"])})

# ---------------------------------------------------------------- 1. learning diagnostic
fa_all = feature_audit().copy()
fa_all["rts"] = pd.to_datetime(fa_all["rating_ts"], format="ISO8601", utc=True)
first_dis = fa_all[fa_all.reward == 0].groupby("rec_id").rts.min()
# ratio of the logged probability of a candidate to 1 / K, for every candidate of every decision
L = pd.DataFrame([(c, row.sel_time, row.pbi[c] * row.K) for row in src.itertuples() if row.K >= 2
                  for c in row.cands if c in row.pbi], columns=["rec", "t", "ratio"])
ctrl = L[~L.rec.isin(set(first_dis.index))]
change, ctrl_change = [], []
for item, d in first_dis.items():
    x = L[L.rec == item]
    b_, a_ = x[x.t <= d].ratio.mean(), x[x.t > d].ratio.mean()
    if np.isnan(b_) or np.isnan(a_):
        continue
    # control: items never disliked, before and after the same date
    cb, ca = ctrl[ctrl.t <= d].groupby("rec").ratio.mean(), ctrl[ctrl.t > d].groupby("rec").ratio.mean()
    cc = pd.concat([cb, ca], axis=1).dropna()
    change.append(np.log(a_ / b_))
    ctrl_change.append(np.median(np.log(cc.iloc[:, 1] / cc.iloc[:, 0])))
change, ctrl_change = np.array(change), np.array(ctrl_change)
n_lower = int((change < ctrl_change).sum())
from scipy.stats import binomtest
p_sign = binomtest(n_lower, len(change), 0.5).pvalue
reg(exLearnItems=str(len(first_dis)), exLearnItemsBoth=str(len(change)),
    exLearnFactor=f"{np.exp(np.median(change)):.2f}", exLearnCtrlFactor=f"{np.exp(np.median(ctrl_change)):.2f}",
    exLearnLower=str(n_lower),
    exLearnP="{:.0e}".format(p_sign).replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}")
print(f"items with a dislike {len(first_dis)}, candidates before and after the first dislike {len(change)}: "
      f"median factor {np.exp(np.median(change)):.2f} vs never-disliked items {np.exp(np.median(ctrl_change)):.2f}; "
      f"below control {n_lower}/{len(change)}, sign test p = {p_sign:.1e}")

# ---------------------------------------------------------------- 4. participant-term rule
fa = fa_all.sort_values("rts", kind="mergesort").reset_index(drop=True)
y, uid, rec = fa.reward.astype(int).to_numpy(), fa.user_id.to_numpy(), fa.rec_id.to_numpy()
lg = lambda p_: np.log(p_ / (1 - p_))
score = np.zeros(len(y))
item_s, item_n, user_s, user_n = {}, {}, {}, {}
likes = count = 0
for t in range(len(y)):          # as in a25: shares from the earlier ratings only
    m = (likes + 1) / (count + 2)
    k_i, k_u = (eb_kappa(rec[:t], y[:t], m=m), eb_kappa(uid[:t], y[:t], m=m)) if t >= 2 else (1000.0, 1000.0)
    pi_ = (item_s.get(rec[t], 0) + k_i * m) / (item_n.get(rec[t], 0) + k_i)
    pu_ = (user_s.get(uid[t], 0) + k_u * m) / (user_n.get(uid[t], 0) + k_u)
    score[t] = lg(pi_) + lg(pu_) - lg(m)
    item_s[rec[t]] = item_s.get(rec[t], 0) + y[t]; item_n[rec[t]] = item_n.get(rec[t], 0) + 1
    user_s[uid[t]] = user_s.get(uid[t], 0) + y[t]; user_n[uid[t]] = user_n.get(uid[t], 0) + 1
    likes += y[t]; count += 1
j["score"] = j.fa_row.map(dict(zip(fa["id"], score)))
j["skip"] = (j.score < 0) & (j.pos >= 4)


def rule(q):
    keep = q[~q.skip]
    return (keep.like.mean() - q.like.mean(), (keep.r.sum() - q.r.sum()) / len(q))


for tag, q in (("All", j), ("After", j[j.after])):
    d = rule(q)
    b = boot(q, rule)
    reg(**{f"exRule{tag}Skip": str(int(q.skip.sum())), f"exRule{tag}SkipDis": str(int((q.skip & (q.like == 0)).sum())),
           f"exRule{tag}Like": f"{q.like.mean():.3f}", f"exRule{tag}LikeRule": f"{q[~q.skip].like.mean():.3f}",
           f"exRule{tag}LikeLo": f(np.percentile(b[:, 0], 2.5)), f"exRule{tag}LikeHi": f(np.percentile(b[:, 0], 97.5)),
           f"exRule{tag}Sum": str(int(q.r.sum())), f"exRule{tag}SumRule": str(int(q[~q.skip].r.sum()))})
    print(f"rule {tag}: skipped {int(q.skip.sum())} ({int((q.skip & (q.like == 0)).sum())} dislikes); "
          f"like {q.like.mean():.3f} -> {q[~q.skip].like.mean():.3f}; sum {q.r.sum()} -> {q[~q.skip].r.sum()}")
print("s07 done")
