# Verified results

Every number below is printed by `code/analysis.py` or `code/supplementary_analysis.py`
(seed 20260710) from the files in `data/`. Raw output is in `results/analysis_output.txt`
and `results/supplementary_output.txt`.

## Empathy null
- within-rater slopes: empathy to safety -0.006, empathy to clinical substance -0.004
- response-level empathy AUC 0.492, 95% CI 0.39 to 0.62 (37 of 504 rated unsafe by both)
- mean empathy >= 4: 7 of 37 both-unsafe (18.9%) against 89 of 467 (19.1%)
- equivalence: holds against a margin of 0.70 (p = 0.0005) and 0.65 (p = 0.0095), not 0.60 (p = 0.104)
- proportional-odds OR for empathy: safety 1.00, accuracy 0.97, referral 1.00, cultural 1.38 (p < 0.001)
- rubric anchoring: 0 of 685 ratings with an unmet required empathy item scored 4 or 5;
  299 of 299 scoring 4 or 5 had both credited; clinical side r = 0.691 and 225 of 656 safety>=4
  ratings had an unmet required item

## English to Hebrew, shared two-physician panel
- clinical substance +0.510, 95% CI 0.395 to 0.605, lower in Hebrew in 20 of 21 scenarios
- empathy -0.018, 95% CI -0.232 to +0.173, 10 of 21
- cultural appropriateness +0.869, 95% CI 0.708 to 1.012, 20 of 21
- D 0.528, 95% CI 0.300 to 0.768, 18 of 21, sign test p = 0.0015 (descriptive only)
- substance share of the change in D: 96.6%

## Product heterogeneity
- per-product D: A 0.429, B 0.254, C 0.151, D 1.278; only D's interval excludes zero
- omitting D: pooled D 0.278, 95% CI -0.019 to 0.577, 13 of 21, p = 0.38
- per-product substance deficit: 0.357, 0.421, 0.579, 0.683, every interval excluding zero
- leave-one-out substance: 0.452 to 0.561

## Safety tail
- 84 scenario-product cells: 2 both, 0 English-only, 8 Hebrew-only, 74 neither
- exact McNemar p = 0.0078; paired risk difference 9.5 points, bootstrap 95% CI 3.6 to 16.7
- discordant cells in 6 scenarios, all one-directional, scenario-level exact sign p = 0.0312
- per physician, rating level: 10 Hebrew-only, 0 English-only, exact McNemar p = 0.00195 each
- Hebrew unsafe cells spread 3/3/2/2 across products; both English cells from one product

## Sensitivity
- corresponding English/Hebrew scenarios only (S09, S10, S13, S18 excluded):
  D 0.544, 95% CI 0.257 to 0.819, 14 of 17, p = 0.0127; tail 2 of 68 against 10 of 68
- fluent in both languages (45 of 84 cells): substance +0.540 (0.388 to 0.690, 19 of 21),
  empathy -0.123, D +0.663 (0.217 to 1.076, 17 of 21)
- composite including cultural appropriateness: D 0.618, 18 of 21
- unsafe cutpoint sweep: 18.9 vs 19.1% at <=2, 17.9 vs 19.2% at <=2.5, 14.6 vs 20.8% at <=3
- prompt word counts: English 797, Hebrew 717, Hebrew shorter in 12 of 21

## Other
- scenario clustering: top-three 49 of 95 unsafe ratings vs null 23.1 (p < 0.001);
  multi-product blocks 25 of 66 rater-by-scenario blocks vs null 13.3 (p < 0.001)
- accurate but unsafe: 10 responses, 9 under-referred, 5 scenarios, 5 languages, 3 products
- inter-rater QW-kappa: pooled 0.7191 over 30 cells (range 0.484 to 0.824)
- source-context substudy: stated Hebrew context gave the highest substance (4.033) of five
  contexts, Arabic the lowest (3.733), Friedman p = 0.06; intra-rater ICC substance 0.635,
  empathy 0.422, D 0.560
- Table 1 rows reproduce exactly for all five responses
