"""
Descriptive statistics — per-cell mean safety, per-language and per-chatbot marginals,
catastrophic-rating rates by language.

Reproduces the headline numbers in Results §1.

Usage:
    python scripts/01_descriptive_stats.py
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'

ratings  = pd.read_csv(DATA / 'ratings.csv')
LANG_ORDER = ['English', 'French', 'Russian', 'Arabic', 'Hebrew', 'Thai']
BOT_ORDER  = ['chatgpt', 'claude', 'deepseek', 'gemini']


def section(title):
    print(f'\n=== {title} ===')


section('Per-cell mean safety (chatbot x language)')
cell_means = (ratings.groupby(['chatbot', 'language'])['safety']
              .mean().unstack().reindex(index=BOT_ORDER, columns=LANG_ORDER).round(3))
print(cell_means)

section('Worst and best cells (by mean safety)')
flat = ratings.groupby(['chatbot', 'language'])['safety'].mean().sort_values()
print(f'Worst cell: {flat.index[0]} = {flat.iloc[0]:.3f}')
print(f'Best cell:  {flat.index[-1]} = {flat.iloc[-1]:.3f}')

section('Per-cell unweighted safety failure rate (% ratings with safety <= 2)')
cell_fail = (ratings.assign(fail=lambda d: (d['safety'] <= 2).astype(int))
             .groupby(['chatbot', 'language'])['fail'].mean()
             .mul(100).round(1)
             .unstack().reindex(index=BOT_ORDER, columns=LANG_ORDER))
print(cell_fail)

section('Per-language marginal mean safety')
for lang in sorted(ratings['language'].unique(),
                   key=lambda x: -ratings[ratings['language'] == x]['safety'].mean()):
    print(f'  {lang:8s} {ratings[ratings["language"] == lang]["safety"].mean():.3f}')

section('Per-chatbot marginal mean safety')
for bot in sorted(ratings['chatbot'].unique(),
                  key=lambda x: -ratings[ratings['chatbot'] == x]['safety'].mean()):
    print(f'  {bot:10s} {ratings[ratings["chatbot"] == bot]["safety"].mean():.3f}')

section('Catastrophic safety ratings by language (individual ratings of safety <= 2)')
cat_lang = (ratings.assign(cat=lambda d: (d['safety'] <= 2).astype(int))
            .groupby('language')['cat'].agg(['sum', 'count', 'mean']))
cat_lang.columns = ['n_catastrophic', 'n_ratings', 'rate']
cat_lang['percent'] = (cat_lang['rate'] * 100).round(2)
cat_lang = cat_lang.reindex(LANG_ORDER)
print(cat_lang[['n_catastrophic', 'n_ratings', 'percent']])
ratio = cat_lang['percent'].max() / cat_lang['percent'].min()
print(f'\nWorst-vs-best disparity ratio: {ratio:.2f}-fold')

# Standardized excess (population-attributable fraction style)
en_rate = cat_lang.loc['English', 'rate']
expected_under_en = en_rate * cat_lang['n_ratings'].sum()
observed = cat_lang['n_catastrophic'].sum()
paf = (observed - expected_under_en) / observed * 100
print(f'Catastrophic ratings attributable to non-English exposure: {paf:.1f}%')
