"""
Inter-rater reliability — quadratic-weighted Cohen kappa per language x dimension cell,
plus mean QW-kappa across the 30 cells.

Reproduces the headline numbers in Methods/Inter-rater reliability:
mean QW-kappa = 0.72; minimum = 0.48 (Thai cultural).

Usage:
    python scripts/03_inter_rater_reliability.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DATA = Path(__file__).resolve().parent.parent / 'data'
ratings = pd.read_csv(DATA / 'ratings.csv')

DIMENSIONS = ['clinical_accuracy', 'safety', 'referral', 'cultural', 'empathy']
LANG_ORDER = ['English', 'French', 'Russian', 'Arabic', 'Hebrew', 'Thai']


def qwk_for_cell(group, dim):
    wide = group.pivot_table(index='response_id', columns='rater_id', values=dim, aggfunc='first')
    raters = wide.columns.tolist()
    if len(raters) < 2:
        return None, 0
    paired = wide[raters[:2]].dropna()
    if len(paired) < 5:
        return None, len(paired)
    return cohen_kappa_score(paired.iloc[:, 0], paired.iloc[:, 1], weights='quadratic'), len(paired)


print('=== Quadratic-weighted Cohen kappa (language x dimension) ===')
header = f'{"":8s}  ' + '  '.join(f'{d[:8]:>8s}' for d in DIMENSIONS) + '  | mean'
print(header)

all_kappas = []
for lang in LANG_ORDER:
    sub = ratings[ratings['language'] == lang]
    line = f'{lang:8s}  '
    row_kappas = []
    for dim in DIMENSIONS:
        k, n = qwk_for_cell(sub, dim)
        if k is None:
            line += f'{"---":>8s}  '
        else:
            line += f'{k:>8.3f}  '
            row_kappas.append(k)
            all_kappas.append(k)
    if row_kappas:
        line += f'| {sum(row_kappas) / len(row_kappas):.3f}'
    print(line)

print()
print(f'Mean QW-kappa across {len(all_kappas)} cells: {sum(all_kappas) / len(all_kappas):.4f}')
print(f'Min QW-kappa: {min(all_kappas):.4f}')
print(f'Max QW-kappa: {max(all_kappas):.4f}')
