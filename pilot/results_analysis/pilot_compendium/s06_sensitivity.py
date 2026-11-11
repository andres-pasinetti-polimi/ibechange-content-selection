"""S06 — Sensitivity of the off-policy conclusions to the two instrument changes
that split the pilot's rating stream.

  (a) Event flow. 30.2% of ratings are collected at mission close rather than
      from a push notification. The compendium's decision to pool the two is
      justified by a TOST bound on the like RATE (within +/-2.89 pp); that bound
      says nothing about item-level structure, and only 9 items have >=5 ratings
      in both flows, so the question cannot be settled from these data.
  (b) Rating interface. Reworded response options and an explicit no-rating
      dismissal shipped on 17 June 2026; the dual-model deployment started
      24 June.

Both are handled here by re-estimating the headline off-policy contrasts on
restricted subsets, rather than by assuming the pooling is harmless.
"""
import json
import warnings

import numpy as np
import pandas as pd

from common import SNAPSHOT, RES_DIR, sigmoid, reg

warnings.filterwarnings("ignore", message=".*encountered in matmul")
BOOT = 5000
RNG = np.random.default_rng(20260814)
PRE = SNAPSHOT.parent / "2026_06_23_13_40"

src = pd.read_csv(PRE / "intervention_mab_samples.csv")
sm = src["sample"].apply(json.loads)
src["p_log"] = sm.apply(lambda d: float(d["p_log"]))
src["cands"] = sm.apply(lambda d: list(dict.fromkeys(c["rec_id"] for c in d.get("candidates", []))))
src["p_by_item"] = sm.apply(lambda d: d.get("mc_propensity", {}).get("p_by_item", {}))
src["rec"] = src["selected_rec_ids"].str.strip("{}")

fa = pd.read_csv(SNAPSHOT / "feature_audit.csv")
fa["prompted"] = fa["prompted"].astype(str).str.lower().isin(["t", "true", "1"])
fa["rts"] = pd.to_datetime(fa["rating_ts"], format="ISO8601", utc=True)
J = src.merge(fa[["user_id", "slot_idx", "rec_id", "reward", "prompted", "rts"]]
              .rename(columns={"rec_id": "rec"}), on=["user_id", "slot_idx", "rec"], how="inner")
J["K"] = J["cands"].apply(len)
J = J[(J.K >= 2) & (J.p_log > 0)].sort_values("rts").reset_index(drop=True)
J["r"] = (J.reward > 0).astype(float)


# The V4 target depends on each user's dislike history. That history is built
# ONCE on the full chronological sequence, so restricting a subset restricts
# the evaluation set only, never the definition of the policy being evaluated.
W_UNI, W_V4 = {}, {}
_disliked = {}
for i, row in J.iterrows():
    W_UNI[i] = (1.0 / row.K) / row.p_log
    bad = _disliked.get(row.user_id, set())
    allowed = [c for c in row.cands if c not in bad] or row.cands
    pb = {c: float(row.p_by_item.get(c, 0.0)) for c in allowed}
    tot = sum(pb.values())
    pi = ({c: v / tot for c, v in pb.items()} if tot > 0
          else {c: 1.0 / len(allowed) for c in allowed})
    W_V4[i] = pi.get(row.rec, 0.0) / row.p_log
    if row.r == 0:
        _disliked.setdefault(row.user_id, set()).add(row.rec)


def contrasts(d):
    """(on-policy - uniform, on-policy - avoid-disliked) with paired cluster CIs."""
    w_uni = np.array([W_UNI[i] for i in d.index])
    w_v4 = np.array([W_V4[i] for i in d.index])
    d = d.reset_index(drop=True)
    r = d.r.values
    idx = {u: d.index[d.user_id == u].values for u in d.user_id.unique()}
    us = np.array(list(idx.keys()))
    out = {}
    for name, w in [("uniform", w_uni), ("avoid_disliked", w_v4)]:
        pt = r.mean() - (w * r).sum() / w.sum()
        bs = []
        for _ in range(BOOT):
            take = np.concatenate([idx[u] for u in RNG.choice(us, len(us), replace=True)])
            wb, rb = w[take], r[take]
            if wb.sum() > 0:
                bs.append(rb.mean() - (wb * rb).sum() / wb.sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = (pt, lo, hi, float(w.sum() ** 2 / (w ** 2).sum()))
    return out, len(d), d.user_id.nunique()


subsets = [
    ("tutti gli slot", J),
    ("solo in-flow (push)", J[~J.prompted]),
    ("solo end-of-mission", J[J.prompted]),
    ("solo interfaccia originale (< 17 giu)", J[J.rts < "2026-06-17"]),
]
print(f"{'sottoinsieme':40s} {'n':>5s} {'ut':>4s} | {'dep - uniforme':>26s} | {'dep - evita disliked':>26s}")
rows = []
for name, d in subsets:
    if len(d) < 50:
        print(f"{name:40s} {len(d):5d}  -- troppo piccolo")
        continue
    res, n, nu = contrasts(d)
    u, v = res["uniform"], res["avoid_disliked"]
    print(f"{name:40s} {n:5d} {nu:4d} | {u[0]:+.4f} [{u[1]:+.4f},{u[2]:+.4f}] "
          f"| {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]")
    rows.append(dict(subset=name, n=n, users=nu,
                     uni=u[0], uni_lo=u[1], uni_hi=u[2], uni_ess=u[3],
                     v4=v[0], v4_lo=v[1], v4_hi=v[2], v4_ess=v[3]))

pd.DataFrame(rows).to_json(RES_DIR / "offpolicy_sensitivity.json", orient="records", indent=1)
inflow = [r for r in rows if r["subset"].startswith("solo in-flow")][0]
reg(opInflowN=str(inflow["n"]),
    opInflowGain=f"{inflow['uni']:+.3f}", opInflowLo=f"{inflow['uni_lo']:+.3f}",
    opInflowHi=f"{inflow['uni_hi']:+.3f}",
    opInflowVfour=f"{-inflow['v4']:+.3f}",
    opInflowVfourLo=f"{-inflow['v4_hi']:+.3f}", opInflowVfourHi=f"{-inflow['v4_lo']:+.3f}")
eom = [r for r in rows if r["subset"].startswith("solo end-of-mission")][0]
def _f3(x):
    s = f"{x:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s
reg(opEomGain=_f3(eom["uni"]), opEomGainLo=_f3(eom["uni_lo"]), opEomGainHi=_f3(eom["uni_hi"]),
    opEomUniESS=f"{eom['uni_ess']:.0f}", opSubsetBootDraws=f"{BOOT:,}")
reg(opEomN=str(eom["n"]), opEomVfour=_f3(-eom["v4"]),
    opEomVfourLo=_f3(-eom["v4_hi"]), opEomVfourHi=_f3(-eom["v4_lo"]))
print("\nscritto offpolicy_sensitivity.json")
