"""
Variance decomposition — Type II ANOVA partial eta squared for language and chatbot
main effects across the five rated dimensions.

Reproduces the headline numbers in Results §1: language >> chatbot, ~33-fold ratio
on safety; cultural appropriateness has the largest language effect.

Usage:
    python scripts/02_variance_decomposition.py
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

DATA = Path(__file__).resolve().parent.parent / 'data'
ratings = pd.read_csv(DATA / 'ratings.csv')

DIMENSIONS = ['clinical_accuracy', 'safety', 'referral', 'cultural', 'empathy']


def eta_squared(formula, data):
    model = ols(formula, data=data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov['sum_sq'].sum()
    out = {}
    for term in aov.index:
        ss = aov.loc[term, 'sum_sq']
        out[term] = ss / ss_total
    return out


print('=== Type II ANOVA - eta squared per dimension (response-level means, n = 504) ===')
print(f'{"":18s}  {"lang_eta2":>10s}  {"chatbot_eta2":>13s}  {"interaction":>13s}  {"ratio":>8s}')

for dim in DIMENSIONS:
    resp_means = ratings.groupby(['response_id', 'language', 'chatbot'])[dim].mean().reset_index()
    eta = eta_squared(f'{dim} ~ C(language) * C(chatbot)', resp_means)
    el = eta['C(language)']
    eb = eta['C(chatbot)']
    ei = eta['C(language):C(chatbot)']
    ratio = el / eb if eb > 0 else float('inf')
    print(f'{dim:18s}  {el:>10.4f}  {eb:>13.4f}  {ei:>13.4f}  {ratio:>7.1f}x')

print()
print('Note: eta squared (SS_effect / SS_total) is reported here, matching tier1_results.json')
print('canonical analysis. Methods text reports partial eta squared in mixed-effects models')
print('(rating-level), which differ slightly because the residual denominator changes.')
