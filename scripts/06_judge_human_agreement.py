"""
LLM-judge vs human-rater agreement on the 48-response designed-pilot validation set.

The 48 cells are the full factorial of two scenarios (S02 indirect suicidal ideation;
S08 carbon monoxide poisoning) by six languages by four chatbots. Each cell has:
  - two LLM judges (Qwen3.5-Plus, Mistral Large 2512) scoring safety and referral
  - two language-matched human raters scoring all five Likert dimensions

Reproduces the headline numbers in Methods/Reproducibility and Supplementary Section S6:
  - inter-judge QW-kappa
  - inter-rater (human) QW-kappa on the 48-cell subset
  - judge-consensus vs human-consensus QW-kappa
on safety and referral.

Usage:
    python scripts/06_judge_human_agreement.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DATA = Path(__file__).resolve().parent.parent / 'data'
PILOT_SCENARIOS = ['S02', 'S08']
DIMENSIONS = ['safety', 'referral']


def qwk(a, b):
    return cohen_kappa_score(list(a), list(b), weights='quadratic')


judges = pd.read_csv(DATA / 'llm_judge_pilot.csv')
ratings = pd.read_csv(DATA / 'ratings.csv')
ratings = ratings[ratings['scenario_id'].isin(PILOT_SCENARIOS)].copy()

# Build human consensus: per response_id, mean of the two raters, rounded to nearest integer
# (Likert ratings are integer 1-5; rounding the two-rater mean preserves the 1-5 categorical
# scale required for quadratic-weighted Cohen kappa.)
human_consensus = (ratings.groupby(['response_id', 'scenario_id', 'language', 'chatbot'])
                          [DIMENSIONS]
                          .mean()
                          .round()
                          .astype(int)
                          .reset_index())

# All 48 cells in the pilot release have scores from both judges.
judges_complete = judges.copy()

# --- inter-judge QWK ---
print(f'=== Inter-judge QW-kappa (Qwen3.5-Plus vs Mistral Large 2512) on {len(judges_complete)} cells ===')
for dim in DIMENSIONS:
    a = judges_complete[f'qwen_{dim}'].astype(int)
    b = judges_complete[f'mistral_{dim}'].astype(int)
    print(f'  {dim:9s}: QW-kappa = {qwk(a, b):.3f}  (n = {len(a)})')

# --- inter-rater (human) QWK on the same 48 cells ---
print('\n=== Inter-rater (human) QW-kappa on the 48 cells ===')
# Pair the two raters per cell. Within each language pair, raters are consistent;
# pivot ratings to one row per (response_id, dim) with two columns.
for dim in DIMENSIONS:
    wide = (ratings.pivot_table(index='response_id', columns='rater_id', values=dim, aggfunc='first'))
    pairs = []
    for response_id, row in wide.iterrows():
        present = row.dropna()
        if len(present) >= 2:
            pairs.append((int(present.iloc[0]), int(present.iloc[1])))
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    print(f'  {dim:9s}: QW-kappa = {qwk(a, b):.3f}  (n = {len(pairs)})')

# --- judge-consensus vs human-consensus QWK ---
print('\n=== Judge-consensus vs human-consensus QW-kappa ===')
# Judge consensus: mean of the two judges, rounded to integer 1-5 (complete cells only)
judge_consensus = judges_complete.copy()
for dim in DIMENSIONS:
    judge_consensus[f'{dim}_judge_consensus'] = (
        (judge_consensus[f'qwen_{dim}'] + judge_consensus[f'mistral_{dim}']) / 2
    ).round().astype(int)

merged = judge_consensus.merge(
    human_consensus,
    on=['response_id', 'scenario_id', 'language', 'chatbot'],
    suffixes=('', '_human'),
    how='inner',
)
print(f'  cells matched: {len(merged)} / {len(judge_consensus)} judge cells')
for dim in DIMENSIONS:
    a = merged[f'{dim}_judge_consensus'].astype(int)
    b = merged[dim].astype(int)
    print(f'  {dim:9s}: QW-kappa = {qwk(a, b):.3f}  (n = {len(merged)})')

# --- mean signed difference (judge minus human) ---
print('\n=== Mean signed difference (judge consensus minus human consensus) ===')
for dim in DIMENSIONS:
    diff = (merged[f'{dim}_judge_consensus'] - merged[dim]).mean()
    print(f'  {dim:9s}: judge - human mean = {diff:+.2f}')
