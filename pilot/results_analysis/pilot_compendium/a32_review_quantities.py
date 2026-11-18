"""A32 - Weights of the off-policy estimates, plan value under (A6), data of the timing networks (18 Nov 2026).

1. Weights of the off-policy estimates (s05 pairing): distribution of the
   importance weights of uniform selection and of the exclusion rule, effective
   sample size, slots per participant, share of the largest participant, the
   contrast without the slot of largest weight, the range of the contrast when
   one participant is left out, and the contrast with the weights clipped at 5,
   the one clipping level used.
2. Plan value under condition (A6): for the 449 plans built up to 23 June 2026,
   the expected content reward of each selected recommendation, mu = 2 p - 1,
   with p the like share of the item among the OTHER participants (shrunk by
   empirical Bayes), is summed over the plan (V, every item dispatched) and
   weighted by the share dispatched at its position in the plan (J). The range of
   J over all orders of the items of each plan follows from the rearrangement
   inequality (best: largest mu at the position with the largest share). These
   are model quantities, not observed rewards.
3. Data behind the timing networks, from the platform records: notifications
   scheduled by the timing module, dispatched and opened within W per
   participant, and the share of scheduled notifications not dispatched before
   the dispatch fix of 7 July 2026.
"""
import json

import numpy as np
import pandas as pd

from common import (EUT_EXCLUDED, delivery, dispatched_notifications, eb_kappa, feature_audit, pct, rating_selection_pairs, reg,
                    selection_log)

DISPATCH_FIX = pd.Timestamp("2026-07-07", tz="UTC")
CLIP = 5.0

# ---------------------------------------------------------------- 1. weights
src = selection_log()
sm = src["sample"].apply(json.loads)
src["cands"] = sm.apply(lambda d: list(dict.fromkeys(c["rec_id"] for c in d.get("candidates", []))))
src["K"] = src["cands"].apply(len)
src["p_log"] = sm.apply(lambda d: float(d["p_log"]))
src["pbi"] = sm.apply(lambda d: d.get("mc_propensity", {}).get("p_by_item", {}))
pairs = rating_selection_pairs()
j = src.merge(pairs[["id", "reward", "rating_time"]], on="id")
j = j[(j.K >= 2) & (j.p_log > 0)].sort_values("rating_time").reset_index(drop=True)
j["like"] = (j.reward > 0).astype(float)
j["w_uni"] = (1 / j.K) / j.p_log
dis, wv = {}, []
for row in j.itertuples():
    bad = dis.get(row.user_id, set())
    allowed = [c for c in row.cands if c not in bad] or row.cands
    pr = {c: float(row.pbi.get(c, 0.0)) for c in allowed}
    tot = sum(pr.values())
    pi = {c: v / tot for c, v in pr.items()} if tot > 0 else {c: 1 / len(allowed) for c in allowed}
    wv.append(pi.get(row.rec, 0.0) / row.p_log)
    if row.like == 0:
        dis.setdefault(row.user_id, set()).add(row.rec)
j["w_v4"] = wv


def snips(w, r):
    return (w * r).sum() / w.sum()


on = j.like.mean()
per_user = j.groupby("user_id").size()
top = per_user.idxmax()
out = dict(wdUsersSlotsMedian=f"{per_user.median():.0f}", wdUsersSlotsQone=f"{per_user.quantile(.25):.0f}",
           wdUsersSlotsQthree=f"{per_user.quantile(.75):.0f}", wdUsersSlotsMax=str(int(per_user.max())),
           wdTopSlotsShare=pct(per_user.max() / len(j)),
           wdTopDislikesShare=pct((j[j.user_id == top].like == 0).sum() / max((j.like == 0).sum(), 1)))
for tag, col, sign in (("Uni", "w_uni", 1), ("Vfour", "w_v4", -1)):
    w = j[col].to_numpy(); r = j.like.to_numpy()
    est = sign * (on - snips(w, r))       # Uni: deployed - uniform; Vfour: exclusion - deployed
    k = np.argmax(w)
    m = np.ones(len(w), bool); m[k] = False
    no_max = sign * (j.like[m].mean() - snips(w[m], r[m]))
    loo = [sign * (j.like[j.user_id != u].mean() - snips(w[j.user_id != u], r[j.user_id != u]))
           for u in j.user_id.unique()]
    clipped = sign * (on - snips(np.minimum(w, CLIP), r))
    out.update({f"wd{tag}Max": f"{w.max():.1f}", f"wd{tag}Pninety": f"{np.quantile(w, .9):.2f}",
                f"wd{tag}Pnn": f"{np.quantile(w, .99):.2f}", f"wd{tag}OverTwo": str(int((w > 2).sum())),
                f"wd{tag}OverFive": str(int((w > 5).sum())), f"wd{tag}ESS": f"{w.sum() ** 2 / (w ** 2).sum():.0f}",
                f"wd{tag}Est": f"{est:+.3f}", f"wd{tag}NoMax": f"{no_max:+.3f}",
                f"wd{tag}LooLo": f"{min(loo):+.3f}", f"wd{tag}LooHi": f"{max(loo):+.3f}",
                f"wd{tag}Clip": f"{clipped:+.3f}"})
    print(tag, {k_: v for k_, v in out.items() if k_.startswith(f"wd{tag}")})
reg(wdClip=f"{CLIP:.0f}", wdN=f"{len(j):,}", **out)
print({k_: v for k_, v in out.items() if k_.startswith("wdUsers") or k_.startswith("wdTop")})

# ---------------------------------------------------------------- 2. plan value under (A6)
sel = src[~src.user_id.astype(str).isin(EUT_EXCLUDED)].copy()
sel = sel.sort_values(["plan_id", "slot_idx"])
sel["pos"] = sel.groupby("plan_id").cumcount() + 1
dl = delivery()
dl = dl[dl.plan_id.isin(set(sel.plan_id)) & dl.content_id.astype(str).str.contains("Rc")].sort_values(["plan_id", "sched"])
seq = sel.groupby("plan_id").rec.apply(list)
rows = []
for pid, g in dl.groupby("plan_id"):            # as in a28: earliest free position of the same item
    s_, free = seq[pid], [True] * len(seq[pid])
    for r in g.itertuples():
        p_ = next((i for i, x in enumerate(s_) if free[i] and x == r.content_id), None)
        if p_ is not None:
            free[p_] = False
            rows.append((pid, p_ + 1, bool(r.sent_ok)))
m_ = pd.DataFrame(rows, columns=["plan_id", "pos", "sent"])
sel = sel.merge(m_, on=["plan_id", "pos"], how="left")
sel["dispatched"] = sel.sent.eq(True)
q_pos = sel.groupby("pos").dispatched.mean()

fa = feature_audit()
y, it, us = fa.reward.astype(int).to_numpy(), fa.rec_id.to_numpy(), fa.user_id.to_numpy()
gm = y.mean()
k_item = eb_kappa(it, y)
tab = pd.DataFrame({"it": it, "us": us, "y": y})
item_tot = tab.groupby("it").y.agg(["sum", "size"])
item_user = tab.groupby(["it", "us"]).y.agg(["sum", "size"])


def p_other(item, user):
    s_, n_ = (item_tot.loc[item, "sum"], item_tot.loc[item, "size"]) if item in item_tot.index else (0, 0)
    if (item, user) in item_user.index:
        s_ -= item_user.loc[(item, user), "sum"]; n_ -= item_user.loc[(item, user), "size"]
    return (s_ + k_item * gm) / (n_ + k_item)


sel["mu"] = [2 * p_other(a, b) - 1 for a, b in zip(sel.rec, sel.user_id)]
sel["q"] = sel.pos.map(q_pos)
V = sel.mu.sum(); J = (sel.q * sel.mu).sum()
best = worst = 0.0
for _, g in sel.groupby("plan_id"):
    mu_ = np.sort(g.mu.to_numpy()); q_ = np.sort(g.q.to_numpy())
    best += (mu_ * q_).sum(); worst += (mu_ * q_[::-1]).sum()
n = len(sel)
reg(avSelected=f"{n:,}", avDispatched=f"{int(sel.dispatched.sum()):,}", avDispatchedShare=pct(sel.dispatched.mean()),
    avV=f"{V / n:.3f}", avJ=f"{J / n:.3f}", avJBest=f"{best / n:.3f}", avJWorst=f"{worst / n:.3f}",
    avMuLo=f"{sel.mu.min():.2f}", avMuHi=f"{sel.mu.max():.2f}")
print(f"plan value per selected recommendation: V {V / n:.3f}, J {J / n:.3f}, range over orders {worst / n:.3f} to "
      f"{best / n:.3f}; dispatched {sel.dispatched.mean():.1%} of {n}")

# ---------------------------------------------------------------- 3. data behind each timing network
d = delivery()
s = dispatched_notifications()
per = pd.DataFrame({"sched": d.groupby("user_id").size(), "disp": s.groupby("user_id").size(),
                    "open_w": s.groupby("user_id").open_w.sum()}).fillna(0)
per = per[per.sched > 0]
pre = d[d.sched < DISPATCH_FIX]
q3 = lambda x: (f"{x.median():.0f}", f"{x.quantile(.25):.0f}", f"{x.quantile(.75):.0f}", f"{x.min():.0f}", f"{x.max():.0f}")
for tag, col in (("Sched", "sched"), ("Disp", "disp"), ("Open", "open_w")):
    md, lo, hi, mn, mx = q3(per[col])
    reg(**{f"tn{tag}Median": md, f"tn{tag}Qone": lo, f"tn{tag}Qthree": hi, f"tn{tag}Min": mn, f"tn{tag}Max": mx})
reg(tnUsers=str(len(per)), tnNoOpen=str(int((per.open_w == 0).sum())), tnDispTotal=f"{int(per.disp.sum()):,}",
    tnPreFixLost=pct(1 - pre.sent_ok.mean()), tnPreFixShare=pct(len(pre) / len(d)))
print(f"participants with scheduled notifications {len(per)}: scheduled median {per.sched.median():.0f}, dispatched "
      f"median {per.disp.median():.0f}, opened within W median {per.open_w.median():.0f}, none opened "
      f"{int((per.open_w == 0).sum())}; dispatched total {int(per.disp.sum())}; pre-fix not dispatched "
      f"{1 - pre.sent_ok.mean():.1%} ({len(pre) / len(d):.1%} of the scheduled are pre-fix)")
print("a32 done")
