"""Capacity of the RCT recommendation catalogue against the stays observed in the pilot.

Prints, as Markdown tables:
  1. pilot: active days per Mission selection (capped at 7) by pillar, and consecutive stays in one pillar;
  2. RCT catalogue v1 (ICO, 23 October 2026): unique recommendations and texts eligible per Mission, per
     pillar (shared items) and per Nutrition Objective, with the day on which new content is exhausted
     (variants start) and the day on which every text has been sent once, at 1 and at 2 recommendations
     a day.

Usage (from ibechange/rct):
  python3 -I system/scripts/catalogue_capacity.py \
      intervention_structure/RCT_Recommendations_v1_23Oct2026_parsed.csv system/catalogue/missions.csv \
      ../pilot/results_analysis/pilot_compendium ../pilot/intervention_structure/derived/mission_state_map.csv

The parsed CSV is written by system/scripts/parse_ico_recommendations_v1.py. The 111 Physical activity
recommendations without a Mission are counted as "All" (provisional, to be confirmed by ICO); the
column "without the block" gives the count without them.
"""
import sys, re, csv, collections
import pandas as pd

parsed_csv, missions_csv, compendium_dir, state_map_csv = sys.argv[1:5]

# ---- 1. Pilot stays -------------------------------------------------------------------------------
sys.path.insert(0, compendium_dir)
from common import missions  # noqa: E402  (pilot loader: selections with closure events)
nm = missions()
mp = pd.read_csv(state_map_csv)[["mission", "pillar"]]
nm = nm.merge(mp, left_on="mission_id", right_on="mission", how="left").sort_values(["user_id", "sel"]).reset_index(drop=True)
nm["next_sel"] = nm.groupby("user_id")["sel"].shift(-1)
nm["next_pil"] = nm.groupby("user_id")["pillar"].shift(-1)
# changes of mind: a selection replaced within a day by another in the same pillar
nm["replaced_1d"] = (nm["next_sel"] - nm["sel"] <= pd.Timedelta(days=1)) & (nm["next_pil"] == nm["pillar"])
k = nm[~nm.replaced_1d].copy()
k["next_sel"] = k.groupby("user_id")["sel"].shift(-1)
end_c = pd.concat([k["sel"] + pd.Timedelta(days=7), k["fin"], k["next_sel"]], axis=1).min(axis=1)
end_n = pd.concat([k["sel"] + pd.Timedelta(days=7), k["next_sel"]], axis=1).min(axis=1)
k["days"] = ((end_c - k["sel"]).dt.total_seconds() / 86400).clip(0, 7)      # to closure with the button
k["days_nc"] = ((end_n - k["sel"]).dt.total_seconds() / 86400).clip(0, 7)   # closure ignored
print("### Pilot: active days per Mission selection (cap 7 days), after removing %d of %d selections replaced within a day\n" % (int(nm.replaced_1d.sum()), len(nm)))
print("| Pillar | Selections | To closure: median | q3 | Share reaching 7 | Closure ignored: median | q3 |")
print("|---|---|---|---|---|---|---|")
for p, g in list(k.groupby("pillar")) + [("All", k)]:
    print(f"| {p} | {len(g)} | {g.days.median():.1f} | {g.days.quantile(.75):.1f} | {(g.days >= 6.99).mean():.2f} | {g.days_nc.median():.1f} | {g.days_nc.quantile(.75):.1f} |")
k["new_run"] = (k["pillar"] != k.groupby("user_id")["pillar"].shift(1)).astype(int)
k["run"] = k.groupby("user_id")["new_run"].cumsum()
runs = k.groupby(["user_id", "run", "pillar"]).agg(days=("days", "sum"), days_nc=("days_nc", "sum")).reset_index()
print("\n### Pilot: consecutive stay in one pillar (sum of capped days over consecutive selections)\n")
print("| Pillar | Runs | To closure: median | q3 | p90 | Closure ignored: median | q3 | p90 |")
print("|---|---|---|---|---|---|---|---|")
for p, g in runs.groupby("pillar"):
    print(f"| {p} | {len(g)} | {g.days.median():.1f} | {g.days.quantile(.75):.1f} | {g.days.quantile(.9):.1f} | {g.days_nc.median():.1f} | {g.days_nc.quantile(.75):.1f} | {g.days_nc.quantile(.9):.1f} |")

# ---- 2. RCT catalogue capacity ---------------------------------------------------------------------
df = pd.read_csv(parsed_csv)
ms = list(csv.DictReader(open(missions_csv)))
pillar_missions = collections.defaultdict(list); objs_of = {}
for m in ms:
    pillar_missions[m["code"][0]].append(m["code"])
    objs_of[m["code"]] = [o for o in re.split(r"[;,\s]+", m["objectives"].strip()) if o]
U, T, U0, T0 = collections.defaultdict(set), collections.Counter(), collections.defaultdict(set), collections.Counter()
for _, r in df.iterrows():
    s = "" if pd.isna(r.missions) else str(r.missions)
    mset = set(re.findall(r"[SANPE]M\d+", s))
    for m in mset:
        U0[m].add(r.code); T0[m] += 1
    if not mset and r.pillar == "Physical_activity":
        mset = set(pillar_missions["P"])
    for m in mset:
        U[m].add(r.code); T[m] += 1
names = {"S": "Smoking", "A": "Alcohol", "N": "Nutrition", "P": "Physical activity", "E": "Emotional wellbeing"}
print("\n### RCT catalogue v1: eligible pool per Mission (median over the pillar's Missions) and exhaustion days\n")
print("| Pillar | Missions | Unique per Mission | Texts per Mission | 1/day: variants from day | all texts used by day | 2/day: variants from day | all texts used by day | Missions with fewer than 7 unique |")
print("|---|---|---|---|---|---|---|---|---|")
for L, name in names.items():
    mm = pillar_missions[L]
    u = pd.Series([len(U[m]) for m in mm]); t = pd.Series([T[m] for m in mm])
    label = name if L != "P" else f"{name} (unassigned block as All; {pd.Series([len(U0[m]) for m in mm]).median():.0f} unique without it)"
    print(f"| {label} | {len(mm)} | {u.median():.0f} | {t.median():.0f} | {u.median():.0f} | {t.median():.0f} | {u.median()/2:.1f} | {t.median()/2:.1f} | {int((u < 7).sum())} |")
ew = pd.Series([T[m] for m in pillar_missions["E"]]); ewu = pd.Series([len(U[m]) for m in pillar_missions["E"]])
print(f"\nEmotional wellbeing per Mission: unique {ewu.min()} to {ewu.max()} (q1 {ewu.quantile(.25):.0f}, q3 {ewu.quantile(.75):.0f}); texts {ew.min()} to {ew.max()} (q1 {ew.quantile(.25):.0f}, q3 {ew.quantile(.75):.0f}).")
print("\n### RCT catalogue v1: shared pools (pillar level) and Nutrition per Objective\n")
print("| Pool | Unique | Texts | 1/day: variants from day | all texts used by day | 2/day: variants from day | all texts used by day |")
print("|---|---|---|---|---|---|---|")
pill = df.groupby("pillar").agg(n_unique=("code", "nunique"), n_texts=("code", "size"))
for p in ["Smoking", "Alcohol", "Physical_activity"]:
    r = pill.loc[p]; print(f"| {p.replace('_', ' ')} (pillar) | {r.n_unique} | {r.n_texts} | {r.n_unique} | {r.n_texts} | {r.n_unique/2:.1f} | {r.n_texts/2:.1f} |")
objU, objT = collections.defaultdict(set), collections.Counter()
for _, r in df.iterrows():
    s = "" if pd.isna(r.missions) else str(r.missions)
    for o in {o for m in set(re.findall(r"[SANPE]M\d+", s)) for o in objs_of.get(m, [])}:
        objU[o].add(r.code); objT[o] += 1
for o in sorted(x for x in objU if x.startswith("N")):
    print(f"| {o} (Objective) | {len(objU[o])} | {objT[o]} | {len(objU[o])} | {objT[o]} | {len(objU[o])/2:.1f} | {objT[o]/2:.1f} |")
