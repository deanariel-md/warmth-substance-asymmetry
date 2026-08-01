#!/usr/bin/env python3
"""Reproduces every main-text statistic in

  "Empathy did not distinguish safe from unsafe consumer chatbot health advice
   in six languages"

Reads data/ratings.csv, data/checklist_items.csv and data/prompts.csv.
Deterministic (fixed seeds).
Dependencies: numpy. Run:  python code/analysis.py
"""
import csv, collections, math, os
import numpy as np
from collections import Counter

SEED, BOOT, PERM = 20260710, 2000, 3000
HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(HERE, "..", "data", f)

R = list(csv.DictReader(open(D("ratings.csv"), encoding="utf-8-sig")))
for x in R:
    for c in ["clinical_accuracy", "safety", "referral", "cultural", "empathy"]:
        x[c] = float(x[c])
    x["sub"] = (x["clinical_accuracy"] + x["safety"] + x["referral"]) / 3.0
    x["D"] = x["sub"] - x["empathy"]

CI = list(csv.DictReader(open(D("checklist_items.csv"), encoding="utf-8-sig")))
chk = collections.defaultdict(list)
for i in CI:
    chk[(i["rater_id"], i["response_id"])].append(i)


def must_frac(key, dims):
    t = c = 0
    for i in chk[key]:
        if i["item_type"] == "must" and i["dimension"] in dims:
            t += 1; c += int(i["checked"])
    return c / t if t else float("nan")


def auc(score, pos):
    score = np.asarray(score, float); pos = np.asarray(pos, bool)
    p, n = score[pos], score[~pos]
    return sum((1 if a > b else .5 if a == b else 0) for a in p for b in n) / (len(p) * len(n))


def within_rater_slope(rows, yk):
    xs, ys = [], []
    by = collections.defaultdict(list)
    for x in rows:
        by[x["rater_id"]].append(x)
    for g in by.values():
        e = np.array([r["empathy"] for r in g]); y = np.array([r[yk] for r in g])
        if e.std() == 0:
            continue
        xs += list(e - e.mean()); ys += list(y - y.mean())
    xs, ys = np.array(xs), np.array(ys)
    return (xs @ ys) / (xs @ xs)


byscen = collections.defaultdict(list)
for x in R:
    byscen[x["scenario_id"]].append(x)
scens = list(byscen)

print("# N = %d ratings of %d responses" % (len(R), len({x["response_id"] for x in R})))

# 1 ---- empathy-quality null ----
print("\n[EMPATHY-QUALITY NULL]")
for yk, lab in [("safety", "safety"), ("sub", "substance")]:
    pt = within_rater_slope(R, yk)
    rng = np.random.default_rng(SEED + (0 if yk == "safety" else 1))
    bs = []
    for _ in range(BOOT):
        s = []
        for sc in rng.choice(scens, len(scens), replace=True):
            s += byscen[sc]
        bs.append(within_rater_slope(s, yk))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  within-rater slope empathy->%-9s %+.3f  95%% CI [%+.2f, %+.2f]" % (lab, pt, lo, hi))

resp = collections.defaultdict(list)
for x in R:
    resp[x["response_id"]].append(x)
ids = list(resp)
emp = {i: np.mean([z["empathy"] for z in resp[i]]) for i in ids}
safe = {i: np.mean([z["safety"] for z in resp[i]]) > 2 for i in ids}
a_safe = auc([emp[i] for i in ids], [safe[i] for i in ids])
bysc = collections.defaultdict(list)
for i in ids:
    bysc[resp[i][0]["scenario_id"]].append(i)
rngA = np.random.default_rng(SEED + 3); bo = []
for _ in range(BOOT):
    e, s = [], []
    for sc in rngA.choice(list(bysc), len(bysc), replace=True):
        for i in bysc[sc]:
            e.append(emp[i]); s.append(safe[i])
    s = np.array(s, bool)
    if s.all() or not s.any():
        continue
    bo.append(auc(e, s))
lo, hi = np.percentile(bo, [2.5, 97.5])
print("  empathy AUC, safe-vs-unsafe separation %.3f  95%% CI [%.2f, %.2f]  (n unsafe = %d)"
      % (a_safe, lo, hi, sum(1 for i in ids if not safe[i])))

# 2 ---- checklist 2x2 ----
recs, nclin = [], 0
for x in R:
    k = (x["rater_id"], x["response_id"])
    nclin += sum(1 for i in chk[k] if i["item_type"] == "must"
                 and i["dimension"] in ("clinical_accuracy", "safety", "referral"))
    recs.append((x["rater_id"], x["sub"], x["empathy"],
                 must_frac(k, ("clinical_accuracy", "safety", "referral")),
                 must_frac(k, ("empathy",))))


def wr(ix, iy):
    by = collections.defaultdict(list); xs, ys = [], []
    for r in recs:
        if r[ix] == r[ix] and r[iy] == r[iy]:
            by[r[0]].append(r)
    for g in by.values():
        a = np.array([r[ix] for r in g]); b = np.array([r[iy] for r in g])
        if a.std() == 0 or b.std() == 0:
            continue
        xs += list(a - a.mean()); ys += list(b - b.mean())
    return np.corrcoef(xs, ys)[0, 1]


print("\n[CHECKLIST 2x2]  required clinical-item judgments = %d" % nclin)
print("  convergent   substance~clinical %.3f | empathy~empathy %.3f" % (wr(1, 3), wr(2, 4)))
print("  discriminant empathy~clinical %+.3f | substance~empathy %+.3f" % (wr(2, 3), wr(1, 4)))

# 2b ---- empathy in the unsafe tail (Results, para 1; Supplementary Note 2) ----
emp_resp = {i: np.mean([x["empathy"] for x in resp[i]]) for i in ids}
uns_resp = {i for i in ids if all(x["safety"] <= 2 for x in resp[i])}
hi = lambda S: (sum(1 for i in S if emp_resp[i] >= 4), len(S))
a, na = hi(uns_resp); b, nb = hi(set(ids) - uns_resp)
print("\n[EMPATHY IN THE UNSAFE TAIL]")
print("  mean empathy >= 4:  both-unsafe %d/%d (%.1f%%) | remainder %d/%d (%.1f%%)"
      % (a, na, 100.0 * a / na, b, nb, 100.0 * b / nb))
print("  mean response-level empathy: corpus %.2f | accurate-but-unsafe %.2f"
      % (np.mean(list(emp_resp.values())),
         np.mean([emp_resp[i] for i in ids
                  if all(x["clinical_accuracy"] >= 4 for x in resp[i])
                  and all(x["safety"] <= 2 for x in resp[i])])))

# 3 ---- within-physician English -> Hebrew ----
byrs = collections.defaultdict(lambda: collections.defaultdict(list))
for x in R:
    byrs[x["rater_id"]][x["scenario_id"]].append(x["D"])
sc = sorted(set(byrs["R1"]) & set(byrs["R3"]) & set(byrs["R2"]) & set(byrs["R4"]))
dd = lambda e, h: np.array([np.mean(byrs[e][s]) - np.mean(byrs[h][s]) for s in sc])
d1, d2 = dd("R1", "R3"), dd("R2", "R4"); ds = (d1 + d2) / 2
rng = np.random.default_rng(SEED + 7)
print("\n[WITHIN-PHYSICIAN English->Hebrew  DeltaD]")
for nm, d in [("P01", d1), ("P02", d2), ("shared", ds)]:
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  %-6s %.3f  95%% CI [%.3f, %.3f]  wider-in-English %d/%d" %
          (nm, d.mean(), lo, hi, int((d > 0).sum()), len(d)))
k = int((ds > 0).sum()); n = len(ds)
print("  shared sign test %d/%d  two-sided p=%.4f"
      % (k, n, min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)))

rmean = lambda rid, key: np.mean([x[key] for x in R if x["rater_id"] == rid])
print("\n[PER-PHYSICIAN DECOMPOSITION]")
for phys, e, h in [("P01", "R1", "R3"), ("P02", "R2", "R4")]:
    print("  %s English D=%.2f sub=%.2f emp=%.2f -> Hebrew D=%.2f sub=%.2f emp=%.2f" % (
        phys, rmean(e, "D"), rmean(e, "sub"), rmean(e, "empathy"),
        rmean(h, "D"), rmean(h, "sub"), rmean(h, "empathy")))
print("  P01 French D=%.2f (vs English %.2f)" % (rmean("R5", "D"), rmean("R1", "D")))

print("\n[PANEL-LEVEL gap D by language]")
lang = collections.defaultdict(list)
for x in R:
    lang[x["language"]].append(x["D"])
for L in ["English", "Hebrew", "French", "Russian", "Arabic", "Thai"]:
    print("  %-8s %.2f" % (L, np.mean(lang[L])))

# 4 ---- scenario clustering ----
cs, cu = collections.defaultdict(list), collections.defaultdict(list)
for x in R:
    k2 = (x["rater_id"], x["chatbot"]); cs[k2].append(x["scenario_id"])
    if x["safety"] <= 2:
        cu[k2].append(x["scenario_id"])


def stat(a):
    s1 = collections.defaultdict(int); bl = collections.defaultdict(set)
    for (r, c), uns in a.items():
        for s in uns:
            s1[s] += 1; bl[(r, s)].add(c)
    return sum(sorted(s1.values(), reverse=True)[:3]), sum(1 for v in bl.values() if len(v) >= 2)


obs = stat(cu); rng2 = np.random.default_rng(SEED + 8); t3, mb = [], []
for _ in range(PERM):
    a = {k2: (list(rng2.choice(cs[k2], len(cu[k2]), replace=False)) if cu[k2] else []) for k2 in cs}
    s = stat(a); t3.append(s[0]); mb.append(s[1])
t3, mb = np.array(t3), np.array(mb)
print("\n[SCENARIO CLUSTERING]  (%d permutations)" % PERM)
print("  top-3 concentration %d vs null %.1f  p=%.4f" % (obs[0], t3.mean(), (t3 >= obs[0]).mean()))
nblocks = len({(x["rater_id"], x["scenario_id"]) for x in R if x["safety"] <= 2})
print("  multi-product blocks %d of %d rater-by-scenario blocks with any unsafe rating, vs null %.1f  p=%.4f"
      % (obs[1], nblocks, mb.mean(), (mb >= obs[1]).mean()))

# 5 ---- accurate but unsafe ----
abu = [(i, resp[i][0]["chatbot"]) for i in ids
       if all(x["clinical_accuracy"] >= 4 for x in resp[i]) and all(x["safety"] <= 2 for x in resp[i])]
ref = sum(1 for i, c in abu if all(x["referral"] <= 3 for x in resp[i]))
# Per-product counts are reported in the manuscript's Supplementary Table S4 under
# randomised labels, because the event counts do not support a product ranking.
# The `chatbot` column of data/ratings.csv is retained, so any reader can recompute them.
print("\n[ACCURATE BUT UNSAFE]  n=%d | under-referred %d | scenarios %d | languages %d | products %d"
      % (len(abu), ref,
         len({resp[i][0]["scenario_id"] for i, _ in abu}),
         len({resp[i][0]["language"] for i, _ in abu}),
         len({c for _, c in abu})))

# 6 ---- safety-tail shift ----
cell = lambda x: (x["scenario_id"], x["chatbot"])
bysr = lambda rid: {cell(x): x for x in R if x["rater_id"] == rid}
uns = lambda x: x["safety"] <= 2


def both(a, b):
    A, B = bysr(a), bysr(b); ks = set(A) & set(B)
    return sum(1 for k in ks if uns(A[k]) and uns(B[k])), len(ks)


def fisher(a, b, c, d):
    pr = lambda a, b, c, d: math.comb(a + b, a) * math.comb(c + d, c) / math.comb(a + b + c + d, a + c)
    o = pr(a, b, c, d); s = 0.0; r1, r2, c1 = a + b, c + d, a + c
    for aa in range(0, min(r1, c1) + 1):
        bb, cc = r1 - aa, c1 - aa; ddd = r2 - cc
        if bb < 0 or cc < 0 or ddd < 0:
            continue
        p = pr(aa, bb, cc, ddd)
        if p <= o + 1e-12:
            s += p
    return s


def mcnemar(rE, rH):
    E, H = bysr(rE), bysr(rH); ks = set(E) & set(H)
    ho = sum(1 for k in ks if uns(H[k]) and not uns(E[k]))
    eo = sum(1 for k in ks if uns(E[k]) and not uns(H[k]))
    sc2 = collections.defaultdict(int)
    for k in ks:
        if uns(H[k]) and not uns(E[k]): sc2[k[0]] += 1
        elif uns(E[k]) and not uns(H[k]): sc2[k[0]] -= 1
    sg = [1 if v > 0 else -1 for v in sc2.values() if v]
    m = len(sg); kp = max(sum(1 for s in sg if s > 0), m - sum(1 for s in sg if s > 0))
    pc = min(1.0, 2 * sum(math.comb(m, i) for i in range(kp, m + 1)) / 2 ** m) if m else 1.0
    return ho, eo, pc, m


eng, heb = both("R1", "R2"), both("R3", "R4")
print("\n[SAFETY-TAIL SHIFT  shared English-Hebrew panel]")
print("  both-unsafe English %d/%d vs Hebrew %d/%d  Fisher p=%.3f"
      % (eng[0], eng[1], heb[0], heb[1], fisher(eng[0], eng[1] - eng[0], heb[0], heb[1] - heb[0])))
for phys, e, h in [("P01", "R1", "R3"), ("P02", "R2", "R4")]:
    ho, eo, pc, m = mcnemar(e, h)
    print("  %s Hebrew-only %d, English-only %d | scenario-cluster-robust p=%.3f (%d scenarios)"
          % (phys, ho, eo, pc, m))
ho, eo, pc, m = mcnemar("R1", "R5")
print("  P01 English->French control: French-only %d, English-only %d" % (ho, eo))

# 8 ---- sensitivity to per-language scenario adaptation (Supplementary Note 4) ----
# Scenarios were adapted independently per language. A scenario is treated as
# non-corresponding between English and Hebrew when the set of numeric clinical
# details carried by the two prompts differs.
import re as _re
P = list(csv.DictReader(open(D("prompts.csv"), encoding="utf-8-sig")))
ptxt = {(x["scenario_id"], x["language"]): x["prompt_sent"] for x in P}
digits = lambda t: tuple(sorted(set(_re.findall(r"\d+(?:[.,]\d+)?", t or ""))))
mismatch = sorted({s0 for s0, _ in ptxt
                   if digits(ptxt.get((s0, "English"))) != digits(ptxt.get((s0, "Hebrew")))})
keep = [s0 for s0 in sc if s0 not in mismatch]
ki = [sc.index(s0) for s0 in keep]
dsk = ds[ki]
rng3 = np.random.default_rng(SEED + 11)
bs = [dsk[rng3.integers(0, len(dsk), len(dsk))].mean() for _ in range(BOOT)]
lo, hi2 = np.percentile(bs, [2.5, 97.5])
kk, nn = int((dsk > 0).sum()), len(dsk)
pk = min(1.0, 2 * sum(math.comb(nn, i) for i in range(kk, nn + 1)) / 2 ** nn)
print("\n[SENSITIVITY: corresponding English/Hebrew scenarios only]")
print("  non-corresponding scenarios excluded: %s" % ", ".join(mismatch))
print("  shared DeltaD %.3f  95%% CI [%.3f, %.3f]  wider-in-English %d/%d  sign p=%.4f"
      % (dsk.mean(), lo, hi2, kk, nn, pk))
E4, H4 = bysr("R1"), bysr("R3"); E5, H5 = bysr("R2"), bysr("R4")
ks4 = [k for k in (set(E4) & set(H4) & set(E5) & set(H5)) if k[0] in keep]
be = sum(1 for k in ks4 if uns(E4[k]) and uns(E5[k]))
bh = sum(1 for k in ks4 if uns(H4[k]) and uns(H5[k]))
print("  both-unsafe English %d/%d vs Hebrew %d/%d (restricted set)" % (be, len(ks4), bh, len(ks4)))


# 9 ---- further sensitivity analyses (Supplementary Notes 2 and 4) ----
print("\n[CHECKLIST COMPLETION by required-item dimension]")
for lab, dims in [("clinical", ("clinical_accuracy", "safety", "referral")),
                  ("cultural", ("cultural",)), ("empathy", ("empathy",))]:
    row = []
    for rid in ("R1", "R2", "R5", "R3", "R4"):
        num = den = 0
        for (rr, _), items in chk.items():
            if rr != rid:
                continue
            for i in items:
                if i["item_type"] == "must" and i["dimension"] in dims:
                    den += 1; num += int(i["checked"])
        row.append("%s %.1f%%" % (rid, 100.0 * num / den))
    print("  %-9s %s" % (lab, " | ".join(row)))

print("\n[SENSITIVITY: unsafe cutpoint]")
gm = {i: (np.mean([x["empathy"] for x in resp[i]]), np.mean([x["safety"] for x in resp[i]])) for i in ids}
for cut in (2.0, 2.5, 3.0):
    u = [i for i in ids if gm[i][1] <= cut]; o = [i for i in ids if gm[i][1] > cut]
    hu = sum(1 for i in u if gm[i][0] >= 4); ho2 = sum(1 for i in o if gm[i][0] >= 4)
    print("  safety<=%.1f  mean empathy>=4: unsafe %d/%d (%.1f%%) vs rest %d/%d (%.1f%%)"
          % (cut, hu, len(u), 100.0 * hu / len(u), ho2, len(o), 100.0 * ho2 / len(o)))

print("\n[SENSITIVITY: composite including cultural appropriateness]")
for x in R:
    x["sub4"] = (x["clinical_accuracy"] + x["safety"] + x["referral"] + x["cultural"]) / 4.0
    x["D4"] = x["sub4"] - x["empathy"]
b4 = collections.defaultdict(lambda: collections.defaultdict(list))
for x in R:
    b4[x["rater_id"]][x["scenario_id"]].append(x["D4"])
d4 = np.array([((np.mean(b4["R1"][t]) - np.mean(b4["R3"][t]))
                + (np.mean(b4["R2"][t]) - np.mean(b4["R4"][t]))) / 2 for t in sc])
k4 = int((d4 > 0).sum())
print("  4-dimension substance: DeltaD %.3f  wider-in-English %d/%d  (3-dimension %.3f, %d/%d)"
      % (d4.mean(), k4, len(d4), ds.mean(), int((ds > 0).sum()), len(ds)))

print("\n[SENSITIVITY: responses both physicians judged fluent]")
flu = collections.defaultdict(list)
for x in R:
    flu[(x["scenario_id"], x["chatbot"], x["language"])].append(x["language_fidelity"])
ok = {k3 for k3, v in flu.items() if all(f == "fluent" for f in v)}
bf = collections.defaultdict(lambda: collections.defaultdict(list))
for x in R:
    if (x["scenario_id"], x["chatbot"], x["language"]) in ok:
        bf[x["rater_id"]][x["scenario_id"]].append(x["D"])
scf = [t for t in sc if bf["R1"].get(t) and bf["R3"].get(t) and bf["R2"].get(t) and bf["R4"].get(t)]
df = np.array([((np.mean(bf["R1"][t]) - np.mean(bf["R3"][t]))
                + (np.mean(bf["R2"][t]) - np.mean(bf["R4"][t]))) / 2 for t in scf])
kf, nf = int((df > 0).sum()), len(df)
pf = min(1.0, 2 * sum(math.comb(nf, i) for i in range(kf, nf + 1)) / 2 ** nf)
print("  both-rater-fluent cells: English %d/84, Hebrew %d/84"
      % (sum(1 for k3 in ok if k3[2] == "English"), sum(1 for k3 in ok if k3[2] == "Hebrew")))
print("  DeltaD %.3f  wider-in-English %d/%d  sign p=%.4f" % (df.mean(), kf, nf, pf))
fm = lambda rid, key: np.mean([x[key] for x in R if x["rater_id"] == rid
                               and (x["scenario_id"], x["chatbot"], x["language"]) in ok])
for phys, e, h in [("P01", "R1", "R3"), ("P02", "R2", "R4")]:
    print("    %s substance %.2f -> %.2f | empathy %.2f -> %.2f"
          % (phys, fm(e, "sub"), fm(h, "sub"), fm(e, "empathy"), fm(h, "empathy")))
print("  language_fidelity by language:")
lf = collections.defaultdict(Counter)
for x in R:
    lf[x["language"]][x["language_fidelity"]] += 1
for L in ["English", "Hebrew", "French", "Russian", "Arabic", "Thai"]:
    print("    %-8s %s" % (L, dict(lf[L])))

print("\n[SENSITIVITY: per-product English->Hebrew DeltaD]")
bp = collections.defaultdict(lambda: collections.defaultdict(list))
for x in R:
    bp[(x["rater_id"], x["chatbot"])][x["scenario_id"]].append(x["D"])
for cb in sorted({x["chatbot"] for x in R}):
    v = [((np.mean(bp[("R1", cb)][t]) - np.mean(bp[("R3", cb)][t]))
          + (np.mean(bp[("R2", cb)][t]) - np.mean(bp[("R4", cb)][t]))) / 2 for t in sc]
    print("  product DeltaD %+.3f" % np.mean(v))


# 10 ---- separate endpoints, product heterogeneity, paired tail (Results; Note 5) ----
def scen_gap(rows, rE, rH, key):
    a = collections.defaultdict(list); b = collections.defaultdict(list)
    for x in rows:
        if x["rater_id"] == rE: a[x["scenario_id"]].append(x[key])
        elif x["rater_id"] == rH: b[x["scenario_id"]].append(x[key])
    ts = [t for t in sc if a.get(t) and b.get(t)]
    return np.array([np.mean(a[t]) - np.mean(b[t]) for t in ts])

def paired_gap(rows, key, seed):
    d = (scen_gap(rows, "R1", "R3", key) + scen_gap(rows, "R2", "R4", key)) / 2
    rg = np.random.default_rng(seed)
    bs = [d[rg.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    k = int((d > 0).sum())
    p = min(1.0, 2 * sum(math.comb(len(d), i) for i in range(k, len(d) + 1)) / 2 ** len(d))
    return d.mean(), lo, hi, k, len(d), p

for x in R:
    x["D3"] = x["sub"] - x["empathy"]
shared = [x for x in R if x["rater_id"] in ("R1", "R2", "R3", "R4")]
print("\n[SEPARATE ENDPOINTS  English minus Hebrew, shared panel]")
# D is printed once, in the WITHIN-PHYSICIAN block above; it is not recomputed here so that
# the paper quotes a single interval for it.
for key, lab in (("sub", "clinical substance"), ("empathy", "empathy"),
                 ("cultural", "cultural approp.")):
    m, lo, hi, k, n, p = paired_gap(shared, key, SEED + 7)
    print("  %-18s %+.3f  95%% CI [%+.3f, %+.3f]  lower-in-Hebrew %d/%d  sign p=%.4f"
          % (lab, m, lo, hi, k, n, p))

sm0, _, _, _, _, _ = paired_gap(shared, "sub", SEED + 7)
em0, _, _, _, _, _ = paired_gap(shared, "empathy", SEED + 7)
dm0, _, _, _, _, _ = paired_gap(shared, "D", SEED + 7)
print("  substance share of the change in D: %.1f%%" % (100 * sm0 / dm0))
print("  shared-panel means by language (substance | empathy | cultural):")
for L in ("English", "Hebrew"):
    rows = [x for x in shared if x["language"] == L]
    print("    %-8s %.2f | %.2f | %.2f" % (L, np.mean([x["sub"] for x in rows]),
          np.mean([x["empathy"] for x in rows]), np.mean([x["cultural"] for x in rows])))

print("\n[TABLE 1  representative responses rated unsafe by both physicians]")
for rid in ("claude_S02_Russian", "deepseek_S12_Russian", "claude_S16_Arabic",
            "deepseek_S14_Hebrew", "gemini_S07_Hebrew"):
    g = resp[rid]
    print("  %-22s empathy %.1f | accuracy %.1f | safety %.1f | referral %.1f"
          % (rid, np.mean([z["empathy"] for z in g]), np.mean([z["clinical_accuracy"] for z in g]),
             np.mean([z["safety"] for z in g]), np.mean([z["referral"] for z in g])))

print("\n[PRODUCT HETEROGENEITY  leave-one-out and per-product]")
# Stable pseudonyms matching Supplementary Table S4, so the supplement and this
# output can be cross-checked without naming products. The `chatbot` column is
# retained in the data, so the mapping is recoverable by any reader.
LBL = {"deepseek": "A", "claude": "B", "chatgpt": "C", "gemini": "D"}
prods = sorted({x["chatbot"] for x in R}, key=lambda c: LBL[c])
for lab, keep in [("all four", None)] + [("omit " + LBL[c], c) for c in prods]:
    rows = shared if keep is None else [x for x in shared if x["chatbot"] != keep]
    sd = SEED + 7 if keep is None else SEED + 22   # full-sample rows share the seed used above
    dm, dlo, dhi, dk, dn, dp = paired_gap(rows, "D", sd)
    sm, slo, shi, sk, sn, _ = paired_gap(rows, "sub", sd)
    print("  %-14s DeltaD %+.3f [%+.3f,%+.3f] %2d/%d p=%.4f | substance %+.3f [%+.3f,%+.3f] %2d/%d"
          % (lab, dm, dlo, dhi, dk, dn, dp, sm, slo, shi, sk, sn))
print("  per-product:")
for c in prods:
    rows = [x for x in shared if x["chatbot"] == c]
    dm, dlo, dhi, dk, dn, _ = paired_gap(rows, "D", SEED + 23)
    sm, slo, shi, _, _, _ = paired_gap(rows, "sub", SEED + 23)
    print("    %s  DeltaD %+.3f [%+.3f,%+.3f] %2d/%d | substance %+.3f [%+.3f,%+.3f]"
          % (LBL[c], dm, dlo, dhi, dk, dn, sm, slo, shi))

print("\n[SENSITIVITY: responses fluent in BOTH languages]")
pairfl = [k3 for k3 in {(x["scenario_id"], x["chatbot"]) for x in shared}
          if (k3[0], k3[1], "English") in ok and (k3[0], k3[1], "Hebrew") in ok]
pf = [x for x in shared if (x["scenario_id"], x["chatbot"]) in pairfl]
print("  cells fluent in BOTH languages: %d of 84" % len(pairfl))
for key, lab in (("sub", "clinical substance"), ("empathy", "empathy"), ("D", "D")):
    m3, lo3, hi3, k3n, n3, p3 = paired_gap(pf, key, SEED + 61)
    print("    paired-fluent %-18s %+.3f  95%% CI [%+.3f, %+.3f]  %d/%d  sign p=%.4f"
          % (lab, m3, lo3, hi3, k3n, n3, p3))

print("\n[PAIRED SAFETY TAIL  exact McNemar on scenario-product cells]")
E, H = bysr("R1"), bysr("R2"); E3, H3 = bysr("R3"), bysr("R4")
ks = sorted(set(E) & set(H) & set(E3) & set(H3))
eu = {k3: uns(E[k3]) and uns(H[k3]) for k3 in ks}
hu = {k3: uns(E3[k3]) and uns(H3[k3]) for k3 in ks}
n11 = sum(1 for k3 in ks if eu[k3] and hu[k3]); n10 = sum(1 for k3 in ks if eu[k3] and not hu[k3])
n01 = sum(1 for k3 in ks if hu[k3] and not eu[k3]); n00 = len(ks) - n11 - n10 - n01
pm = min(1.0, 2 * sum(math.comb(n10 + n01, i) for i in range(max(n10, n01), n10 + n01 + 1)) / 2 ** (n10 + n01)) if n10 + n01 else 1.0
print("  both %d | English-only %d | Hebrew-only %d | neither %d  (n=%d)" % (n11, n10, n01, n00, len(ks)))
print("  English %d/%d vs Hebrew %d/%d  exact McNemar p=%.4g"
      % (sum(eu.values()), len(ks), sum(hu.values()), len(ks), pm))
byscen = collections.defaultdict(list)
for k3 in ks:
    byscen[k3[0]].append(int(hu[k3]) - int(eu[k3]))
dvals = [v for vs in byscen.values() for v in vs]
rgd = np.random.default_rng(SEED + 51)
scl = sorted(byscen)
bsd = [np.mean([v for x in rgd.choice(scl, len(scl), replace=True) for v in byscen[x]])
       for _ in range(BOOT)]
lo_d, hi_d = np.percentile(bsd, [2.5, 97.5])
print("  paired risk difference %.1f percentage points, scenario-cluster bootstrap 95%% CI %.1f to %.1f"
      % (100 * np.mean(dvals), 100 * lo_d, 100 * hi_d))
disc = {x: sum(v) for x, v in byscen.items() if sum(v) != 0}
sg = [1 if v > 0 else -1 for v in disc.values()]
md = len(sg); kd = sum(1 for x in sg if x > 0)
pd_ = min(1.0, 2 * sum(math.comb(md, i) for i in range(max(kd, md - kd), md + 1)) / 2 ** md)
print("  discordant cells in %d distinct scenarios, %s one-directional, scenario-level exact sign p=%.4f"
      % (md, "all" if kd in (0, md) else "not all", pd_))
for phys, e2, h2 in [("P01", "R1", "R3"), ("P02", "R2", "R4")]:
    A2, B2 = bysr(e2), bysr(h2); kk2 = sorted(set(A2) & set(B2))
    eo2 = sum(1 for k3 in kk2 if uns(A2[k3]) and not uns(B2[k3]))
    ho2 = sum(1 for k3 in kk2 if uns(B2[k3]) and not uns(A2[k3]))
    pm2 = min(1.0, 2 * sum(math.comb(eo2 + ho2, i) for i in range(max(eo2, ho2), eo2 + ho2 + 1)) / 2 ** (eo2 + ho2)) if eo2 + ho2 else 1.0
    print("  %s rating-level: English-only %d, Hebrew-only %d, exact McNemar p=%.5f" % (phys, eo2, ho2, pm2))
print("  Hebrew unsafe cells per product %s | English %s"
      % (sorted(Counter(k3[1] for k3 in ks if hu[k3]).values(), reverse=True),
         sorted(Counter(k3[1] for k3 in ks if eu[k3]).values(), reverse=True)))
for c in prods:
    kk = [k3 for k3 in ks if k3[1] != c]
    print("    omit %s: English %d/%d vs Hebrew %d/%d"
          % (LBL[c], sum(eu[k3] for k3 in kk), len(kk), sum(hu[k3] for k3 in kk), len(kk)))

print("\n[EMPATHY SCORE vs REQUIRED EMPATHY ITEMS  rubric anchoring]")
tab = collections.Counter()
for (rr, rid2), items in chk.items():
    e = [i for i in items if i["item_type"] == "must" and i["dimension"] == "empathy"]
    if not e: continue
    sc_ = [x for x in R if x["rater_id"] == rr and x["response_id"] == rid2]
    if not sc_: continue
    tab[(int(sc_[0]["empathy"]), sum(int(i["checked"]) for i in e))] += 1
print("  empathy score x items credited:")
for e in range(1, 6):
    print("    score %d: %s" % (e, [tab[(e, k4)] for k4 in (0, 1, 2)]))
unmet = sum(v for (e, k4), v in tab.items() if k4 < 2)
print("  ratings with an unmet required empathy item that scored >=4: %d of %d"
      % (sum(v for (e, k4), v in tab.items() if k4 < 2 and e >= 4), unmet))

print("\n[PROMPT WORD COUNTS English vs Hebrew]")
we = sum(len(ptxt[(t, "English")].split()) for t in sc)
wh = sum(len(ptxt[(t, "Hebrew")].split()) for t in sc)
short = sum(1 for t in sc if len(ptxt[(t, "Hebrew")].split()) < len(ptxt[(t, "English")].split()))
print("  English %d words | Hebrew %d words | Hebrew shorter in %d/%d scenarios" % (we, wh, short, len(sc)))

# ---------------------------------------------------------------------------
# [SI NOTE 4]  ORIGIN LANGUAGE OF THE SCENARIO, and prompt length
# Each scenario came from a forum post in one language and was adapted into the
# other five. Where Hebrew is the origin language, the Hebrew prompt is the
# original and the English one the adaptation.
_SCEN = sorted({x["scenario_id"] for x in R})
SRC = {}
for _lang, _ss in [("Russian", "S01 S03 S08 S17"), ("French", "S04 S06 S09"),
                   ("Arabic", "S07 S14 S16 S19"), ("Hebrew", "S02 S18 S20"),
                   ("English", "S05 S10 S11 S12 S13 S15"), ("Thai", "S21")]:
    for _s in _ss.split():
        SRC[_s] = _lang

def _pair_deficit(key):
    out = {}
    for t in _SCEN:
        vals = []
        for a, b in (("R1", "R3"), ("R2", "R4")):
            ea = [x[key] for x in R if x["rater_id"] == a and x["scenario_id"] == t]
            eb = [x[key] for x in R if x["rater_id"] == b and x["scenario_id"] == t]
            vals.append(sum(ea) / len(ea) - sum(eb) / len(eb))
        out[t] = sum(vals) / len(vals)
    return out

print("\n[SI NOTE 4]  substance deficit by the scenario's origin language")
_d = _pair_deficit("sub")
_rng = np.random.default_rng(SEED)
# An interval is printed only where the subset is large enough for one to mean anything.
# With three scenarios a bootstrap has ten distinct resample means and its 2.5th and 97.5th
# percentiles are just the smallest and largest observed value.
for _label, _keep in (("Hebrew is the original", lambda t: SRC[t] == "Hebrew"),
                      ("English is the original", lambda t: SRC[t] == "English"),
                      ("a third language", lambda t: SRC[t] not in ("Hebrew", "English"))):
    _v = np.array([_d[t] for t in _SCEN if _keep(t)])
    if len(_v) >= 10:
        _b = [_v[_rng.integers(0, len(_v), len(_v))].mean() for _ in range(BOOT)]
        _ci = "95%% CI %+.3f to %+.3f" % tuple(np.percentile(_b, [2.5, 97.5]))
    else:
        _ci = "no interval, n too small"
    print("  %-24s n=%2d  %+.3f  %-26s lower in Hebrew %d/%d"
          % (_label, len(_v), _v.mean(), _ci, (_v > 0).sum(), len(_v)))

_pw = {}
for _r in csv.DictReader(open(os.path.join(HERE, "..", "data", "prompts.csv"), encoding="utf-8-sig")):
    _pw[(_r["scenario_id"], _r["language"])] = len(_r["prompt_sent"].split())
_gap = np.array([_pw[(t, "Hebrew")] - _pw[(t, "English")] for t in _SCEN], float)
_def = np.array([_d[t] for t in _SCEN], float)
_r = np.corrcoef(_gap, _def)[0, 1]
_z = math.atanh(_r); _se = 1.0 / math.sqrt(len(_SCEN) - 3)
_rlo, _rhi = math.tanh(_z - 1.96 * _se), math.tanh(_z + 1.96 * _se)
print("  prompt length: Hebrew %d words, English %d, correlation with the deficit %+.3f"
      " (Fisher 95%% CI %+.2f to %+.2f, so 21 scenarios bound this loosely)"
      % (sum(_pw[(t, "Hebrew")] for t in _SCEN), sum(_pw[(t, "English")] for t in _SCEN),
         _r, _rlo, _rhi))
print("    deficit where the Hebrew prompt is shorter (n=%d) %+.3f, longer (n=%d) %+.3f"
      % ((_gap < 0).sum(), _def[_gap < 0].mean(), (_gap >= 0).sum(), _def[_gap >= 0].mean()))

print("\n# done")
