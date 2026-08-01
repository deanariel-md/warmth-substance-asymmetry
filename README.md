# Reproduction kit — "Empathy did not distinguish safe from unsafe consumer chatbot health advice in six languages"

Archived at Zenodo concept DOI 10.5281/zenodo.20100653; repository https://github.com/deanariel-md/warmth-substance-asymmetry .

Physician ratings of 504 consumer-chatbot answers to 21 clinician-adapted, forum-derived
patient scenarios in six languages, each rated by two language-matched physicians on five
1–5 dimensions and on an item-level clinical checklist.

## Quick start

```bash
pip install -r requirements.txt      # numpy only
python code/analysis.py              # every main-text statistic

pip install pandas scipy statsmodels
python code/supplementary_analysis.py   # the Supplementary Information quantities
```

Two scripts, two output files. Nothing else to run.

Expected output is in `results/analysis_output.txt` and `results/supplementary_output.txt`.
Both scripts are deterministic (fixed seeds). `analysis.py` needs numpy only;
`supplementary_analysis.py` covers the quantities that need a statistics library
(quadratic-weighted kappa, probability of superiority, proportional-odds models, and the
proportion intervals quoted in Supplementary Note 2).

## What it prints

The empathy–quality null (within-rater slopes and the AUC with its scenario-cluster
bootstrap CI); the checklist 2×2 convergent and discriminant correlations over 4,416
required clinical-item judgments; the within-physician English-to-Hebrew gap with bootstrap
CIs, the cluster-valid sign test, and the per-physician decomposition; the panel-level gap
by language; the cross-product scenario-clustering permutation; the accurate-but-unsafe
set; the safety-tail shift with Fisher and scenario-cluster-robust tests; the proportion of
warm answers inside and outside the unsafe tail; and the sensitivity analysis restricted to
the scenarios whose English and Hebrew prompts correspond.

`analysis.py` also prints the sensitivity analyses: the unsafe cutpoint sweep, the composite
with cultural appropriateness included, the subset both physicians judged linguistically
fluent, the per-product direction check, and the restriction to scenarios whose English and
Hebrew prompts carry matching numeric details.

Per-product counts are not printed. They are reported under randomised labels in
Supplementary Table S4, because the event counts do not support a product ranking. The
`chatbot` column is retained in `ratings.csv` and `responses.csv`, so any reader can
recompute them.

## Citation

Ariel D, Grumberg LR, Supakul S, et al. Multilingual consumer-chatbot health-advice
evaluation across six languages: physician ratings, prompts, and reproduction kit.
Zenodo. Concept DOI 10.5281/zenodo.20100653 (cite the concept DOI, which always resolves
to the current version).

## Data (`data/`)

| File | Rows | Contents |
|---|---|---|
| `ratings.csv` | 1,008 | one row per physician per response: five 1–5 dimensions and language fidelity |
| `responses.csv` | 504 | the verbatim chatbot output for every cell, with model version and collection timestamp |
| `checklist_items.csv` | 12,576 | one row per checklist item judgment: dimension, `must`/`bonus`, checked, item text |
| `raters.csv` | 12 | rater ID → language and anonymized `person_id` (R1/R3/R5 = P01; R2/R4 = P02) |
| `prompts.csv` | 126 | adapted scenario prompt text per scenario × language |
| `paired_english_ratings.csv` | 200 | the paired English-with-country-context arm |

The four products were not queried on the same dates: ChatGPT and Claude on 2026-04-07,
Gemini between 2026-04-07 and 2026-04-09, DeepSeek on 2026-05-01. `response_collected_at`
in `responses.csv` carries the per-response timestamp.

Physicians appear only as anonymized identifiers; no names are released. The checklist
files contain clinical criteria and check marks only, with no protected health information.
Scenario text is adapted from publicly posted patient forum content with identifying detail
removed, and source URLs are withheld.

The rating tool also recorded an advisory red-flag field intended to cap a score when a
specific danger was present. It was never applied in scoring and is used in no analysis, so
it is not released; the recorded 1–5 scores are each physician's final judgment.

## License

Code is MIT; data, figures, and documentation are CC BY 4.0. See `LICENSE`.
