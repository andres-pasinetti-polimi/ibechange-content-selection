"""A31 - Check of the rule that pairs each rating with its selection (17 Nov 2026).

common.rating_selection_pairs() finds the copy of the rated item through the
dispatched notification of the rating (process_id), the scheduled copy of the
item in the same plan and the rank of that copy (k_chain). This script reports
how often that copy differs from the one given by counting the earlier
deliveries of the item in the Mission (k_delivered, the first version of the
rule), and whether the difference comes from an earlier copy that was
scheduled and not dispatched.
"""
from common import delivery, pct, rating_selection_pairs, reg

p = rating_selection_pairs()
chain = p["k_chain"].notna()
diff = chain & (p["k_chain"] != p["k_delivered"])
d = delivery()
d = d[d["plan_id"].isin(set(p["plan_id"]))].sort_values(["plan_id", "sched"]).copy()
d["sched_copy"] = d.groupby(["plan_id", "content_id"]).cumcount() + 1
skipped = sum(bool((~d[(d.plan_id == r.plan_id) & (d.content_id == r.rec) & (d.sched_copy < r.k_chain)].sent_ok).any())
              for r in p[diff & (p["k_chain"] > p["k_delivered"])].itertuples())
later = int((diff & (p["k_chain"] > p["k_delivered"])).sum())
print(f"pairs {len(p)}; copy found through the dispatched notification {int(chain.sum())}")
print(f"  copy differs from the delivery count: {int(diff.sum())} "
      f"(push {int((diff & ~p.prompted).sum())}, review {int((diff & p.prompted).sum())})")
print(f"  later copy {later}, of which with an earlier copy scheduled and not dispatched {skipped}; "
      f"earlier copy {int((diff & (p['k_chain'] < p['k_delivered'])).sum())}")
reg(pairAll=f"{len(p):,}", pairChain=f"{int(chain.sum()):,}", pairDiff=str(int(diff.sum())),
    pairDiffLater=str(later), pairDiffSkipped=str(skipped),
    pairDiffEarlier=str(int((diff & (p['k_chain'] < p['k_delivered'])).sum())))

# Sensitivity of the main off-policy contrast (deployed minus uniform, SNIPS, no
# bootstrap) to the choice of the copy: the rule above, the delivery count, and a
# random copy of the item in the plan (200 draws).
import json
import numpy as np
from common import selection_log

src = selection_log()
_sm = src["sample"].apply(json.loads)
src["p_log"] = _sm.apply(lambda d_: float(d_["p_log"]))
src["K"] = _sm.apply(lambda d_: len(dict.fromkeys(c["rec_id"] for c in d_.get("candidates", []))))
ncop = src.groupby(["plan_id", "rec"]).size()


def contrast(q):
    jj = src.merge(q[["plan_id", "rec", "k", "reward"]], left_on=["plan_id", "rec", "copy"],
                   right_on=["plan_id", "rec", "k"])
    jj = jj[(jj.K >= 2) & (jj.p_log > 0)]
    r_, w_ = (jj.reward > 0).astype(float), (1 / jj.K) / jj.p_log
    return r_.mean() - (w_ * r_).sum() / w_.sum()


c_chain = contrast(p)
alt = p[["plan_id", "rec", "k_delivered", "reward"]].rename(columns={"k_delivered": "k"})
c_deliv = contrast(alt[~alt.duplicated(["plan_id", "rec", "k"], keep=False)])
rng = np.random.default_rng(20261117)
draws = []
for _ in range(200):
    q = p[["plan_id", "rec", "reward"]].copy()
    q["k"] = [rng.integers(1, ncop[(a_, b_)] + 1) for a_, b_ in zip(q.plan_id, q.rec)]
    draws.append(contrast(q.drop_duplicates(["plan_id", "rec", "k"])))
print(f"deployed - uniform: rule {c_chain:+.4f}, delivery count {c_deliv:+.4f}, "
      f"random copy {min(draws):+.4f} to {max(draws):+.4f}")
reg(pairSensRule=f"{c_chain:+.3f}", pairSensDelivered=f"{c_deliv:+.3f}",
    pairSensRandomLo=f"{min(draws):+.3f}", pairSensRandomHi=f"{max(draws):+.3f}")
print("a31 done")
