# Consumer health AI across six languages — data and reproduction kit

This repository contains the de-identified data and self-contained analysis scripts for the study *"Asymmetry between warmth and clinical substance in multilingual consumer health AI"* (Ariel et al.). The kit reproduces all headline numbers in the manuscript directly from the released CSVs.

## Study at a glance

- **Design**: 4 chatbots × 6 languages × 21 forum-derived clinical scenarios = **504 responses**
- **Rating**: 2 language-matched native-speaker clinicians per language = **1,008 ratings** on five 1–5 Likert dimensions (clinical accuracy, safety, referral appropriateness, cultural appropriateness, empathy)
- **Chatbots**: ChatGPT (`gpt-5.3-chat-latest`), Claude (`claude-sonnet-4-6`), Gemini (`gemini-2.5-flash`), DeepSeek (`deepseek-v3.2`); temperature 0.7, zero-shot, single-turn
- **Languages**: English, Hebrew, French, Russian, Arabic, Thai

## Repository layout

```
multilingual_chatbot_safety_release/
├── README.md
├── LICENSE                 (MIT, applies to scripts/)
├── LICENSE_DATA.md         (CC-BY 4.0, applies to data/)
├── requirements.txt
├── data/
│   ├── responses.csv                504 chatbot responses (verbatim, primary corpus)
│   ├── ratings.csv                  1,008 human ratings (5 Likert dims + language fidelity)
│   ├── raters.csv                   12 raters (ID + assigned language; identities withheld)
│   ├── scenarios.csv                21 scenarios with clinical metadata
│   ├── framework_predictors.csv     24 chatbot × language cells: predictors + mean safety
│   ├── s15_rerating.csv             Post-hoc human re-rating of the 8 S15 LLM-judge-flagged
│   │                                anchoring-arm baseline cells (16 ratings = 8 cells × 2 raters)
│   ├── anchoring_arm.csv            240 cross-lingual anchoring-arm responses (5 scenarios ×
│   │                                6 languages × 4 chatbots × 2 conditions) with locked-pair
│   │                                LLM-judge safety/referral scores (Qwen3.5-Plus, Mistral
│   │                                Large 2512) and consensus
│   ├── paired_english_controls.csv  200 paired-English controls (10 scenarios × 5 source-context
│   │                                languages × 4 chatbots) with PI ratings
│   ├── reproducibility_n3.csv       N = 3 response-resampling reproducibility: 5 scenarios × 6
│   │                                languages × 4 chatbots × 2 dimensions = 240 rows of
│   │                                locked-pair judge consensus across three independent samples
│   ├── mitigation_arm.csv           70 remediation stress-test responses (English safety prefix,
│   │                                language-matched prefix, country-cue PoC) with locked-pair
│   │                                judge scores
│   ├── emergency_numbers_audit.csv  Clinician hand-coded audit of local-emergency-number
│   │                                provision in 200 non-English emergency-relevant responses
│   │                                (Methods/Localization-audit; Supplementary §S5)
│   └── llm_judge_pilot.csv          48-response designed-pilot LLM-judge cross-validation set
│                                    (full factorial of 2 scenarios × 6 languages × 4 chatbots)
│                                    with locked-pair safety/referral scores and consensus
└── scripts/
    ├── 01_descriptive_stats.py     Per-cell mean safety, marginals, catastrophic rates
    ├── 02_variance_decomposition.py Type II ANOVA η² across the five dimensions
    ├── 03_inter_rater_reliability.py Quadratic-weighted Cohen κ per language × dimension
    ├── 04_omission_analysis.py     S16 stroke + S04 PEP-French regex counts; emergency-number
    │                                rates from clinician audit; S08 / S11 / sentinel-fact
    │                                counts referenced from manuscript supplement
    ├── 05_risk_framework.py        Three-predictor framework (URIEL + Joshi + fertility) with LOOCV
    └── 06_judge_human_agreement.py LLM-judge vs human-rater QW-kappa on the 48-cell pilot
```

## Quick start

Python 3.10+ recommended.

```bash
pip install -r requirements.txt

python scripts/01_descriptive_stats.py
python scripts/02_variance_decomposition.py
python scripts/03_inter_rater_reliability.py
python scripts/04_omission_analysis.py
python scripts/05_risk_framework.py
python scripts/06_judge_human_agreement.py
```

Each script reads only from `data/*.csv` and prints results to stdout. No network calls, no external state.

## What each script reproduces

| Script | Manuscript anchor | Headline numbers it prints |
|---|---|---|
| `01_descriptive_stats.py` | Table 2; Discussion ¶4 | Per-language safety means; catastrophic rate per language (3.6% English to 15.5% Hebrew/Thai, 4.34-fold disparity, 62.1% excess) |
| `02_variance_decomposition.py` | Figure 2a; Results §1 | Type II ANOVA η² for the five rated dimensions; safety language η² 0.1133 vs chatbot η² 0.0034 (33.5× ratio) |
| `03_inter_rater_reliability.py` | ED Table 1; Methods | QW-κ per language × dimension cell; mean 0.72; min 0.48 (Thai cultural) |
| `04_omission_analysis.py` | Results §2 / §3; Figure 3 | S16 stroke 0/24; S04 PEP 48-h cap (Claude/DeepSeek French); local-emergency-number rates (DeepSeek 50%, Claude 38%, Gemini 32%, ChatGPT 18%; 0/200 US-911 leakage). S08, S11, and sentinel-fact counts (each 0/24 or 0/120) require clinician audit and are referenced from manuscript supplement |
| `05_risk_framework.py` | Figure 4; ED Figure 5; Supp Figure S10 | Composite ρ = -0.85, in-sample AUC 0.8951, LOOCV pooled AUC 0.9161, leave-Thai-out AUC 0.8352; per-chatbot URIEL ρ -0.90 to -0.99 |
| `06_judge_human_agreement.py` | Methods/Reproducibility; Supp §S6 | Inter-judge QW-κ 0.84 (safety) / 0.83 (referral); inter-rater QW-κ 0.89 / 0.87 on the 48 cells; judge-vs-human QW-κ 0.59 / 0.68 with judges +0.6 lenient on safety |

## Data dictionaries

### `responses.csv`

| Column | Description |
|---|---|
| `response_id` | Unique key: `{chatbot}_{scenario_id}_{language}` |
| `scenario_id` | One of S01–S21 |
| `language` | English / Hebrew / French / Russian / Arabic / Thai |
| `chatbot` | chatgpt / claude / gemini / deepseek |
| `model_version` | Pinned model alias at collection date |
| `prompt_sent` | Verbatim prompt sent to the chatbot |
| `response_text` | Verbatim chatbot response (UTF-8) |
| `response_collected_at` | ISO 8601 UTC timestamp of API response. Collection window 2026-04-07 to 2026-05-01. |

### `ratings.csv`

| Column | Description |
|---|---|
| `rater_id` | R1–R12 (cross-reference with `raters.csv`) |
| `response_id` | Foreign key to `responses.csv` |
| `scenario_id`, `language`, `chatbot` | Convenience columns (denormalized from `response_id`) |
| `clinical_accuracy`, `safety`, `referral`, `cultural`, `empathy` | 1–5 Likert ratings |
| `language_fidelity` | 5-way categorical: `fluent`, `minor_errors`, `many_errors`, `partial`, `wrong` |

### `raters.csv`

| Column | Description |
|---|---|
| `rater_id` | R1–R12 |
| `language` | Rater's assigned language |

Each language is rated by two raters; rater identities and roles (PI vs clinician) are withheld here and documented in the manuscript byline and acknowledgments.

### `scenarios.csv`

| Column | Description |
|---|---|
| `scenario_id` | S01–S21 |
| `clinical_domain` | Free-text domain label |
| `one_line_summary` | Brief description |
| `post_type` | Proxy / Crisis / Minimization / Reassurance / Barrier / Advice |
| `failure_mechanism` | Recognition / Knowledge / Reasoning / Validation |
| `source_language` | Language of the original forum post |

### `s15_rerating.csv`

Post-hoc human re-rating of the 8 anchoring-arm baseline S15 cells flagged by the LLM-judge consensus as catastrophic (Gemini × {English, French, Arabic, Thai} and DeepSeek × {English, Hebrew, Arabic, Thai}). 16 rows = 8 cells × 2 language-matched raters; rated through the same blinded interface and rubric as the primary 504-corpus. Source for the "8/8 catastrophic by ≥1 rater, 6/8 by both" finding in Results §4 and Supplementary Section S15. Reference data file; no kit script consumes it.

| Column | Description |
|---|---|
| `response_id_internal` | `cross_lingual_anchoring_S15_{language}_baseline_{chatbot}` |
| `scenario`, `chatbot`, `language` | Cell identifiers |
| `rater_id` | R1–R12 (cross-reference with `raters.csv`) |
| `clinical_accuracy_1to5`, `safety_1to5`, `referral_1to5`, `cultural_1to5`, `empathy_1to5` | 1–5 Likert |
| `language_fidelity` | 5-way categorical (same scheme as `ratings.csv`) |
| `prompt_text`, `response_text` | Verbatim prompt and chatbot response |

### `framework_predictors.csv`

| Column | Description |
|---|---|
| `chatbot`, `language` | Cell identifiers |
| `mean_fertility` | Tokens/word for that chatbot's tokenizer on that language |
| `joshi_tier` | Joshi resource-availability tier (3–5) |
| `uriel_distance` | URIEL typological distance from English (6-bin ordinal) |
| `composite_risk_score` | z-score sum of the three predictors |
| `mean_safety` | Observed mean safety in that chatbot × language cell |

### `anchoring_arm.csv`

Cross-lingual anchoring arm: 5 scenarios (S02, S05, S08, S09, S15) × 6 languages × 4 chatbots × 2 conditions (baseline + family-voice anchored) = 240 responses, scored by the locked LLM-judge pair on referral and safety. Source for the anchoring-arm results in Methods, "Cross-lingual anchoring arm." Within-cell paired Δ (anchored − baseline) on consensus scores is the primary outcome.

| Column | Description |
|---|---|
| `response_id` | `cross_lingual_anchoring_{scenario}_{language}_{condition}_{chatbot}` |
| `scenario_id`, `language`, `condition`, `chatbot` | Cell identifiers (`condition` ∈ baseline / anchored) |
| `model_version` | Pinned model alias |
| `prompt_text`, `response_text` | Verbatim prompt and chatbot response |
| `response_collected_at` | ISO 8601 UTC timestamp of API response. Arm collection window 2026-05-01. |
| `qwen_safety`, `qwen_referral` | Qwen3.5-Plus (2026-04-20) judge scores, 1–5 |
| `mistral_safety`, `mistral_referral` | Mistral Large 2512 judge scores, 1–5 |
| `consensus_safety`, `consensus_referral` | Mean of the two judges (primary outcome) |

### `paired_english_controls.csv`

200 English responses to the same prompts used in the five non-English source-context languages, with an English wrapper that explicitly opens with country context (*"I live in [country] and …"*) and preserves local brands and culturally specific detail. Rated by the PI alone with recall-bias mitigations (rating sessions ≥3 weeks apart from primary-corpus English rating; rubric re-read; 30–50 re-rated primary-corpus responses interleaved per session); 40-pair inter-rater check by the primary-corpus English co-rater on a seeded subset (Methods, "Paired English controls").

| Column | Description |
|---|---|
| `response_id` | `paired_english_controls_{scenario}_{source_context_language}_{chatbot}` |
| `scenario_id`, `source_context_language`, `chatbot` | Cell identifiers |
| `model_version` | Pinned model alias |
| `prompt_text`, `response_text` | Verbatim English-with-country-opener prompt and chatbot response |
| `response_collected_at` | ISO 8601 UTC timestamp of API response. Arm collection window 2026-04-30. |
| `rater_id` | `PI` (sole rater for this arm) |
| `clinical_accuracy`, `safety`, `referral`, `cultural`, `empathy` | 1–5 Likert |
| `language_fidelity` | 5-way categorical (always English here) |

### `reproducibility_n3.csv`

N = 3 response-resampling reproducibility on five scenarios × six languages × four chatbots = 120 cells, scored on safety + referral by the locked judge pair across three independent chatbot response sets (primary corpus, anchoring-arm baseline rerun, additional independent rerun). Source for the ICC values in Methods, "Reproducibility."

| Column | Description |
|---|---|
| `chatbot`, `scenario_id`, `language` | Cell identifiers |
| `dimension` | `safety` or `referral` |
| `score_N1`, `score_N2`, `score_N3` | Locked-pair consensus score in each of the three samples |
| `mean`, `sd`, `range` | Across-sample summary statistics |

### `mitigation_arm.csv`

Three remediation stress-test layers run on worst-performing primary-corpus cells: an English safety-prefix arm (n = 20), a language-matched localized prefix arm (n = 20), and a country-cue proof-of-concept (n = 30) for French, Arabic, and Russian with strong locale-identifying cues planted without naming the country. Framed as proof-of-concept; not a controlled mitigation evaluation.

| Column | Description |
|---|---|
| `response_id` | Per-arm prompt identifier (e.g., `mitigation_safety_prefix_*`, `mitigation_localized_*`, `mitigation_countrycue_*`) |
| `arm` | `english_safety_prefix` / `language_matched_prefix` / `country_cue_proof_of_concept` |
| `scenario_id`, `language`, `target_context_language`, `chatbot` | Cell identifiers |
| `condition` | Per-arm condition label |
| `model_version` | Pinned model alias |
| `prompt_text`, `response_text` | Verbatim prompt and chatbot response |
| `response_collected_at` | ISO 8601 UTC timestamp of API response. Arm collection windows: english_safety_prefix and language_matched_prefix 2026-04-30; country_cue_proof_of_concept 2026-05-03. |
| `qwen_safety`, `qwen_referral`, `mistral_safety`, `mistral_referral`, `consensus_safety`, `consensus_referral` | Locked-pair judge scores (same as anchoring) |

### `emergency_numbers_audit.csv`

Clinician hand-coded audit of the local-emergency-number provision question across the 200 non-English emergency-relevant responses (10 emergency scenarios × 5 non-English languages × 4 chatbots). Each response was scored for whether it named the correct local emergency number for the inferred-deployment country, gave only generic emergency phrasing, omitted any emergency reference, defaulted to a wrong non-US number, or leaked the US default (911). Source for the localization analysis in Results §3 and Figure 3b / Extended Data Figure 5.

| Column | Description |
|---|---|
| `response_id` | Foreign key to `responses.csv` |
| `scenario_id`, `language`, `chatbot` | Cell identifiers |
| `emergency_number_mentioned` | Boolean: any emergency number string detected |
| `number_found` | The literal number string the response gave (or empty) |
| `classification` | One of: `correct_local`, `generic_no_number`, `none`, `other_wrong` |
| `brief_note` | Auditor's one-line rationale for the classification |

### `llm_judge_pilot.csv`

48-response designed-pilot validation set: full factorial of two scenarios × six languages × four chatbots, drawn from the primary corpus. The two scenarios were chosen as a complementary pair: S08 (carbon monoxide poisoning with family-stress misattribution) — a physical-toxicology emergency with clear must-have content criteria and high human inter-rater agreement — and S02 (indirect suicidal ideation) — a psychiatric crisis-handling stress test on which judge calibration was substantively informative. Source for the LLM-judge cross-validation reported in Methods, "Reproducibility," and Supplementary Section S6.

All 48 cells have scores from both judges and the consensus.

| Column | Description |
|---|---|
| `response_id` | Foreign key to `responses.csv` |
| `scenario_id`, `language`, `chatbot` | Cell identifiers |
| `qwen_safety`, `qwen_referral`, `mistral_safety`, `mistral_referral` | Per-judge scores, 1–5 |
| `consensus_safety`, `consensus_referral` | Mean of the two judges |

## Reproducibility notes

- **ANOVA spec**: Type II, response-level means (n = 504), with chatbot × language interaction. eta squared is reported as SS_effect / SS_total throughout `02_variance_decomposition.py`. The Methods section additionally reports partial eta squared from rating-level mixed-effects models; see Methods.
- **IRR**: quadratic-weighted Cohen kappa per (language, dimension) cell, computed from the two raters paired in each language; equivalent to ICC for equally-spaced ordinal categories.
- **Risk-framework cutoff**: clinical-cutoff binarization at mean safety < 3.7; this cutoff was committed in writing before validation. Cutoff stability is reported across 3.5 / 3.7 / 4.0.
- **Random seeds**: bootstrap and permutation analyses in the manuscript use `numpy.random.default_rng(seed=42)`; not all scripts in this kit re-bootstrap (point estimates are presented).

## Citation

If you use this data, please cite:

> Ariel D, Grumberg LR, Supakul S, Wannasri S, Mitchnik IY, Lev A, Ariyamethanon W, Agbarieh M, Miari S, Laban G, Hasid B. Asymmetry between warmth and clinical substance in multilingual consumer health AI. medRxiv preprint, 2026.

Data and code archive: Zenodo, https://doi.org/10.5281/zenodo.20100653 (concept DOI — always resolves to the latest version).

## Licenses

- Code (everything in `scripts/`): MIT — see `LICENSE`
- Data (everything in `data/`): CC-BY 4.0 — see `LICENSE_DATA.md`

Source-forum types and scenario provenance categories are described in the manuscript's supplementary appendix (Table S3). **Specific forum names, domains, direct source URLs, usernames, post dates, and verbatim source-post text are not released.**
