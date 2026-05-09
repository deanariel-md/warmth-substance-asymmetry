"""
Risk-stratification framework — three-predictor convergent framework:
URIEL typological distance, Joshi resource tier, and chatbot-specific tokenization
fertility. The composite z-score sum is the integrated risk score.

Reproduces:
  - Spearman rho between composite and observed mean safety
  - In-sample AUC for composite at clinical cutoff (mean safety < 3.7)
  - Per-predictor AUC (URIEL alone, fertility alone, Joshi alone)
  - Per-chatbot URIEL-safety and fertility-safety rho (replication across 4 model families)
  - Leave-one-language-out cross-validation (LOOCV) pooled AUC
  - Leave-Thai-out sensitivity check

Usage:
    python scripts/05_risk_framework.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / 'data'
fc = pd.read_csv(DATA / 'framework_predictors.csv')

CUTOFF = 3.7
y_true = (fc['mean_safety'] < CUTOFF).astype(int)


def auc(predictor_score):
    return roc_auc_score(y_true, predictor_score)


print('=== Three-predictor convergent framework (n = 24 chatbot x language cells) ===')
print()

# Composite (Spearman + AUC at cutoff 3.7)
rho_comp, p_comp = stats.spearmanr(fc['composite_risk_score'], fc['mean_safety'])
print(f'Composite (URIEL + Joshi + fertility, z-score sum):')
print(f'  Spearman rho with mean safety:  {rho_comp:+.4f}  (p = {p_comp:.4f})')
print(f'  AUC at safety < {CUTOFF}:           {auc(fc["composite_risk_score"]):.4f}')

# Per-predictor AUC
print()
print(f'Single-predictor AUCs at safety < {CUTOFF}:')
print(f'  URIEL typological distance:  {auc(fc["uriel_distance"]):.4f}')
print(f'  Tokenization fertility:      {auc(fc["mean_fertility"]):.4f}')
print(f'  Joshi tier (5 - tier):       {auc(5 - fc["joshi_tier"]):.4f}')

# URIEL AUC across cutoffs (stability)
print()
print('URIEL AUC stability across cutoffs:')
for c in [3.5, 3.7, 4.0]:
    yt = (fc['mean_safety'] < c).astype(int)
    if yt.sum() > 0 and yt.sum() < len(yt):
        print(f'  cutoff {c}:  AUC = {roc_auc_score(yt, fc["uriel_distance"]):.4f}')

# Per-chatbot replication (URIEL and fertility separately)
print()
print('Per-chatbot replication (n = 6 languages each):')
print(f'{"chatbot":<12s} {"URIEL rho":>12s} {"fertility rho":>16s}')
for bot in sorted(fc['chatbot'].unique()):
    sub = fc[fc['chatbot'] == bot]
    r_uriel, _ = stats.spearmanr(sub['uriel_distance'], sub['mean_safety'])
    r_fert,  _ = stats.spearmanr(sub['mean_fertility'], sub['mean_safety'])
    print(f'{bot:<12s} {r_uriel:>+12.3f} {r_fert:>+16.3f}')

# Leave-one-language-out cross-validation (LOOCV)
# For each held-out language, refit only the z-score SD denominators on the
# 5 training languages; the literature-anchored centres (fertility=2.0,
# joshi=5.0, URIEL=0.35) stay locked across folds. Pooled AUC is computed
# on the concatenated held-out predictions across the 6 folds.
print()
print('=== Leave-one-language-out cross-validation (n = 24 cells, 6 folds) ===')
LANGUAGES = sorted(fc['language'].unique())
FERT_C, JOSHI_C, URIEL_C = 2.0, 5.0, 0.35
FERT_SD_DEFAULT, JOSHI_SD_DEFAULT, URIEL_SD_DEFAULT = 1.0, 2.0, 0.20
loocv_pred = np.zeros(len(fc))
for held in LANGUAGES:
    train = fc[fc['language'] != held]
    test = fc[fc['language'] == held]
    lang_pred = train.drop_duplicates('language')[
        ['mean_fertility', 'joshi_tier', 'uriel_distance']
    ]
    f_sd = lang_pred['mean_fertility'].std(ddof=1) or FERT_SD_DEFAULT
    j_sd = lang_pred['joshi_tier'].std(ddof=1) or JOSHI_SD_DEFAULT
    u_sd = lang_pred['uriel_distance'].std(ddof=1) or URIEL_SD_DEFAULT
    fz = (test['mean_fertility'] - FERT_C) / f_sd
    jz = (JOSHI_C - test['joshi_tier']) / j_sd  # higher tier => lower risk
    uz = (test['uriel_distance'] - URIEL_C) / u_sd
    loocv_pred[test.index] = fz + jz + uz
y_pooled = (fc['mean_safety'] < CUTOFF).astype(int)
loocv_auc = roc_auc_score(y_pooled, loocv_pred)
print(f'  LOOCV pooled AUC at safety < {CUTOFF}:   {loocv_auc:.4f}')
print(f'  In-sample composite AUC for comparison: {auc(fc["composite_risk_score"]):.4f}')

# Leave-Thai-out sensitivity
print()
print('=== Leave-Thai-out sensitivity (n = 20 non-Thai cells) ===')
non_thai = fc[fc['language'] != 'Thai']
rho_nt, _ = stats.spearmanr(non_thai['composite_risk_score'], non_thai['mean_safety'])
y_true_nt = (non_thai['mean_safety'] < CUTOFF).astype(int)
auc_nt = roc_auc_score(y_true_nt, non_thai['composite_risk_score'])
print(f'  Composite Spearman rho: {rho_nt:+.4f}')
print(f'  Composite AUC:          {auc_nt:.4f}')
