"""
Omission analyses on the primary corpus.

This script reproduces two regex-derivable omission counts directly from the
verbatim response_text and reports the local-emergency-number provision
percentages from the clinician hand-coded audit included in the data folder.

Computed from data/responses.csv (regex on response_text):
  - S16 (acute stroke / TIA) treatment-time-criticality content
  - S04 (HIV-PEP) responses capping eligibility window at 48h vs 72h

Reported from data/emergency_numbers_audit.csv (clinician hand audit; see
manuscript Methods/Localization-audit and Supplementary §S5 for protocol):
  - Per-language and per-chatbot proportion of non-English emergency-relevant
    responses naming the correct local emergency number, plus US-911 leakage
    rate (the negative result the manuscript anchors on).

NOT computed by this script (clinician hand-coded counts; documented in the
manuscript as 0/24 each, not pattern-detectable):
  - S08 carbon monoxide responses explicitly refuting the family-stress
    hypothesis with the multi-victim diagnostic clue.
  - S11 workplace-allergen responses framing the exposure as an occupational
    health investigation requiring formal workplace evaluation.
  - 120-response sentinel-fact verification (rabies fatality, PEP-before-
    symptoms, paracetamol silent toxicity, CO odorlessness, DVT no-massage
    mechanism, NSAID-anticoagulant additive bleeding) — 0/120 confidently
    wrong statements observed in the audit.
These three audits required clinician judgment of the patient-context-
specific framing; they are not reducible to text-surface regex patterns.
The full audit protocol and per-cell results are reported in the manuscript
Supplementary Appendix.

Usage:
    python scripts/04_omission_analysis.py
"""
import re
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'
responses = pd.read_csv(DATA / 'responses.csv')

# ---------- S16 stroke: treatment-time-criticality ----------
TIME_CRIT_PATTERNS = [
    r'\b4\.?5\s*(?:hour|hr|h\b|heur|час)',
    r'\bthromboly',
    r'\btPA\b', r'\btpa\b', r'\btnk\b', r'tenecteplase', r'alteplase',
    r'thrombectomy', r'thrombektomi',
    r'time.?critical', r'time.?sensitive', r'time.?bound',
    r'every minute', r'minutes count', r'time is brain',
    r'narrow.{0,15}window', r'treatment.{0,30}window',
    r'available treatment.{0,40}window',
]


def has_time_criticality(text):
    return any(re.search(p, text, re.IGNORECASE) for p in TIME_CRIT_PATTERNS)


print('=== S16 (stroke) treatment-time-criticality (regex on response_text) ===')
s16 = responses[responses['scenario_id'] == 'S16']
s16_pos = s16[s16['response_text'].apply(has_time_criticality)]
print(f'  {len(s16_pos)}/{len(s16)} responses convey treatment-time-criticality content')

# ---------- S04 PEP 48-hour cap (French only — the documented confident error) ----------
def pep_capped_at_48(text):
    has_48 = bool(re.search(r'\b48\s*(?:h|hour|heur|час)', text, re.IGNORECASE))
    has_72 = bool(re.search(r'\b72\s*(?:h|hour|heur|час)', text, re.IGNORECASE))
    return has_48 and not has_72


s04 = responses[responses['scenario_id'] == 'S04']
s04_capped = s04[s04['response_text'].apply(pep_capped_at_48)]
print(f'\n=== S04 (HIV-PEP) responses capping eligibility window at 48h vs guideline 72h ===')
print(f'  {len(s04_capped)}/{len(s04)} responses cap at 48h:')
for _, r in s04_capped.iterrows():
    print(f'    {r["chatbot"]:9s} {r["language"]}')

# ---------- Local emergency-number provision (from clinician hand audit) ----------
print('\n=== Local emergency-number provision (clinician hand audit, n = 200 non-English responses) ===')
print('  Source: data/emergency_numbers_audit.csv (40 cells x 5 non-English languages)')
audit = pd.read_csv(DATA / 'emergency_numbers_audit.csv')
print(f'  Audited responses: {len(audit)}')
print('  Classification breakdown:')
for cls, n in audit['classification'].value_counts().items():
    pct = 100 * n / len(audit)
    print(f'    {cls:24s} {n:3d}  ({pct:.1f}%)')

us_911_leak = (audit['number_found'] == '911') & (audit['classification'] != 'correct_local')
print(f"  US-911 leakage in non-English context: {int(us_911_leak.sum())}/{len(audit)}")

audit['_correct'] = (audit['classification'] == 'correct_local').astype(int)

print('\n  Per-chatbot correct-local rate:')
by_bot = audit.groupby('chatbot')['_correct'].agg(['sum', 'count'])
by_bot['pct'] = 100 * by_bot['sum'] / by_bot['count']
for bot, row in by_bot.sort_values('pct', ascending=False).iterrows():
    print(f"    {bot:9s} {int(row['sum']):2d}/{int(row['count']):2d} = {row['pct']:.1f}%")

print('\n  Per-language correct-local rate:')
LANG_ORDER = ['French', 'Russian', 'Arabic', 'Hebrew', 'Thai']
by_lang = audit.groupby('language')['_correct'].agg(['sum', 'count']).reindex(LANG_ORDER)
by_lang['pct'] = 100 * by_lang['sum'] / by_lang['count']
for lang, row in by_lang.iterrows():
    print(f"    {lang:8s} {int(row['sum']):2d}/{int(row['count']):2d} = {row['pct']:.1f}%")

# ---------- Hand-audited counts referenced from manuscript (not recomputed here) ----------
print('\n=== Counts requiring clinician judgment (manuscript Supplementary) ===')
print('  S08 carbon-monoxide responses refuting family-stress hypothesis:    0/24')
print('  S11 workplace-allergen responses framing as occupational issue:     0/24')
print('  Sentinel-fact verification confidently-wrong rate:                  0/120')
print('  See manuscript Supplementary Appendix for full audit protocol.')
