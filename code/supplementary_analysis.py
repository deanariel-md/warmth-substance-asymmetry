#!/usr/bin/env python3
"""Reproduces the Supplementary Information quantities that need more than numpy.

Covers: Table S3 (quadratic-weighted Cohen's kappa by language and dimension),
Supplementary Note 1 (probability of superiority; proportional-odds models,
including the cultural-appropriateness model), and the interval and exact tests
quoted in Supplementary Note 2.

Everything reported in the main text is in code/analysis.py, which needs numpy
only. Run:  pip install pandas scipy statsmodels && python code/supplementary_analysis.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "..", "data", "ratings.csv"))
DIMS = ["clinical_accuracy", "safety", "referral", "cultural", "empathy"]
LANGS = ["English", "French", "Russian", "Arabic", "Hebrew", "Thai"]


def qwk(a, b, k=5):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x - 1, y - 1] += 1
    W = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    ha = np.array([(a == c).sum() for c in range(1, k + 1)]) / len(a)
    hb = np.array([(b == c).sum() for c in range(1, k + 1)]) / len(b)
    E = np.outer(ha, hb) * len(a)
    return 1 - (W * O).sum() / (W * E).sum()


print("[TABLE S3]  quadratic-weighted Cohen's kappa")
cells = []
for L in LANGS:
    s = R[R.language == L]
    rids = sorted(s.rater_id.unique(), key=lambda x: int(x[1:]))
    p = s.pivot(index="response_id", columns="rater_id", values=DIMS)
    row = [qwk(p[(d, rids[0])], p[(d, rids[1])]) for d in DIMS]
    cells += row
    print("  %-8s %s  row mean %.3f"
          % (L, " ".join("%.3f" % v for v in row), np.mean(row)))
print("  pooled mean over the 30 cells %.4f  (range %.3f to %.3f)"
      % (np.mean(cells), min(cells), max(cells)))

print("\n[NOTE 1]  probability of superiority against English (response-level means)")
R["sub"] = R[["clinical_accuracy", "safety", "referral"]].mean(axis=1)
G = R.groupby(["response_id", "language"])[["sub", "empathy"]].mean().reset_index()
for L in [x for x in LANGS if x != "English"]:
    out = []
    for col in ("sub", "empathy"):
        a = G.loc[G.language == "English", col].to_numpy()
        b = G.loc[G.language == L, col].to_numpy()
        u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        out.append("%s %.3f" % (col, u / (len(a) * len(b))))
    print("  English vs %-8s %s" % (L, " | ".join(out)))

print("\n[NOTE 1]  proportional-odds models on within-rater-centred empathy")
R["ec"] = R.empathy - R.groupby("rater_id").empathy.transform("mean")
for y in ["safety", "clinical_accuracy", "referral", "cultural"]:
    m = OrderedModel(R[y].astype(int), R[["ec"]], distr="logit").fit(method="bfgs", disp=False)
    print("  %-18s OR %.2f  p=%.3g" % (y, np.exp(m.params["ec"]), m.pvalues["ec"]))

print("\n[NOTE 2]  empathy in the unsafe tail, with intervals")
g = R.groupby("response_id").agg(emp=("empathy", "mean"), saf=("safety", "mean"))
for cut in (2.0, 2.5, 3.0):
    u, o = g[g.saf <= cut], g[g.saf > cut]
    a, na, b, nb = (u.emp >= 4).sum(), len(u), (o.emp >= 4).sum(), len(o)
    lo, hi = confint_proportions_2indep(a, na, b, nb, method="wald")
    print("  safety<=%.1f  %d/%d (%.1f%%) vs %d/%d (%.1f%%)  Fisher p=%.3f  diff 95%% CI [%+.3f, %+.3f]"
          % (cut, a, na, 100 * a / na, b, nb, 100 * b / nb,
             stats.fisher_exact([[a, na - a], [b, nb - b]])[1], lo, hi))
print("  Wilson 95%% CI on 7/37: [%.3f, %.3f]"
      % tuple(proportion_confint(7, 37, method="wilson")))

print("\n[NOTE 2]  residual empathy-safety correlation after centring")
d = R.copy()
for _ in range(50):
    for col, out in (("empathy", "e"), ("safety", "s")):
        v = d[col] - d.groupby("rater_id")[col].transform("mean")
        d[out] = v - v.groupby(d.scenario_id).transform("mean")
print("  rater and scenario centred %+.3f" % np.corrcoef(d.e, d.s)[0, 1])
d["e2"] = d.empathy - d.groupby(["rater_id", "scenario_id"]).empathy.transform("mean")
d["s2"] = d.safety - d.groupby(["rater_id", "scenario_id"]).safety.transform("mean")
print("  rater-by-scenario cell centred %+.3f" % np.corrcoef(d.e2, d.s2)[0, 1])

print("\n[NOTE 2]  within-rater empathy-safety correlation, per rating assignment")
for rid, s in sorted(R.groupby("rater_id"), key=lambda kv: int(kv[0][1:])):
    print("  %-4s %-8s %+.3f" % (rid, s.language.iloc[0], np.corrcoef(s.empathy, s.safety)[0, 1]))

print("\n[RESULTS para 1]  equivalence margins for the empathy AUC")
G2 = R.groupby(["response_id", "scenario_id"]).agg(emp=("empathy", "mean"), smax=("safety", "max")).reset_index()
def _auc(sc, pos):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool)
    a, b = sc[pos], sc[~pos]
    return sum((1 if x > y else .5 if x == y else 0) for x in a for y in b) / (len(a) * len(b))
obs = _auc(G2.emp, G2.smax <= 2)
rng = np.random.default_rng(20260710)
scn = sorted(G2.scenario_id.unique()); boots = []
for _ in range(2000):
    d = pd.concat([G2[G2.scenario_id == x] for x in rng.choice(scn, len(scn), replace=True)])
    if (d.smax <= 2).sum() and (d.smax > 2).sum():
        boots.append(_auc(d.emp, d.smax <= 2))
boots = np.array(boots); far = np.maximum(boots, 1 - boots)
# The AUC and its interval are printed once, by code/analysis.py. This block adds only the
# post hoc equivalence margins, so the paper quotes a single interval for the AUC.
for margin in (0.70, 0.65, 0.60):
    print("    equivalence within max(AUC, 1-AUC) < %.2f: one-sided p = %.4f"
          % (margin, (far >= margin).mean()))
print("  largest orientation-free AUC compatible with the data: max(AUC, 1-AUC) = %.2f"
      % np.percentile(far, 97.5))

print("\n[RESULTS para 1]  rubric anchoring of the empathy score")
C = pd.read_csv(os.path.join(HERE, "..", "data", "checklist_items.csv"))
emp = C[(C.item_type == "must") & (C.dimension == "empathy")]
g = emp.groupby(["rater_id", "response_id"]).agg(n=("checked", "size"), k=("checked", "sum")).reset_index()
m = g.merge(R[["rater_id", "response_id", "empathy"]], on=["rater_id", "response_id"])
print("  ratings with an unmet required empathy item: %d, of which %d scored 4 or 5"
      % ((m.k < m.n).sum(), ((m.k < m.n) & (m.empathy >= 4)).sum()))
print("  ratings scoring 4 or 5: %d, of which %d had both required empathy items credited"
      % ((m.empathy >= 4).sum(), ((m.empathy >= 4) & (m.k == m.n)).sum()))
print("  ratings with both credited: %d, of which %d scored exactly 3"
      % ((m.k == m.n).sum(), ((m.k == m.n) & (m.empathy == 3)).sum()))
clin = C[(C.item_type == "must") & (C.dimension.isin(["clinical_accuracy", "safety", "referral"]))]
gc = clin.groupby(["rater_id", "response_id"]).agg(n=("checked", "size"), k=("checked", "sum")).reset_index()
mc = gc.merge(R[["rater_id", "response_id", "safety"]], on=["rater_id", "response_id"])
hi = mc[mc.safety >= 4]
print("  clinical side, ratings scoring safety 4 or above: %d, of which %d had an unmet required item"
      % (len(hi), (hi.k < hi.n).sum()))
print("  score-checklist correlation, clinical: %.3f" % np.corrcoef(mc.safety, mc.k / mc.n)[0, 1])

print("\n[NOTE 6]  source-context re-rating substudy")
P = pd.read_csv(os.path.join(HERE, "..", "data", "paired_english_ratings.csv"))
P["sub"] = P[["clinical_accuracy", "safety", "referral"]].mean(axis=1)
print("  %d ratings by %s of %d English responses under %d stated source contexts"
      % (len(P), P.rater_id.unique()[0], P.response_id.nunique(), P.source_context_language.nunique()))
print(P.groupby("source_context_language")[["sub", "empathy"]].mean().round(3).to_string())
for col in ("sub", "empathy"):
    w = P.pivot_table(index="response_id", columns="source_context_language", values=col)
    print("  Friedman across contexts, %-8s chi2=%.2f p=%.3f"
          % (col, *stats.friedmanchisquare(*[w[c] for c in w.columns])[:2]))
base = R[R.rater_id == "R1"].set_index("response_id")
base["sub"] = base[["clinical_accuracy", "safety", "referral"]].mean(axis=1)
common = [i for i in P.response_id.unique() if i in base.index]
for col in ("sub", "empathy"):
    A = np.array([[base.loc[i, col]] + list(P.loc[P.response_id == i, col]) for i in common], float)
    kk = A.shape[1]
    msb = A.mean(axis=1).var(ddof=1) * kk; msw = np.mean(A.var(axis=1, ddof=1))
    print("  intra-rater one-way ICC over %d ratings of the same %d texts, %-8s %.3f  (mean within-text range %.2f)"
          % (kk, len(common), col, (msb - msw) / (msb + (kk - 1) * msw), (A.max(axis=1) - A.min(axis=1)).mean()))
Ad = np.array([[base.loc[i, "sub"] - base.loc[i, "empathy"]]
               + list(P.loc[P.response_id == i, "sub"] - P.loc[P.response_id == i, "empathy"]) for i in common], float)
kk = Ad.shape[1]; msb = Ad.mean(axis=1).var(ddof=1) * kk; msw = np.mean(Ad.var(axis=1, ddof=1))
print("  intra-rater one-way ICC, D %.3f  (mean within-text range %.2f)"
      % ((msb - msw) / (msb + (kk - 1) * msw), (Ad.max(axis=1) - Ad.min(axis=1)).mean()))

print("\n# done")
