"""Phase 4a: scenario sensitivity. Scores each scenario cohort with the FROZEN anchors and compares it customer by customer
with a reference cohort (same PAN = same master-seed draws; pairing is approximate because replay-time RNG draws diverge).
Shock scenarios are compared with `control_noshock` (shock=none for everyone); all others with `baseline`.
Expected directions are hypotheses fixed BEFORE looking at results: -1 down, +1 up, 0 = no change expected.
Usage: python validation/run_scenarios.py <scenarios_root> <anchors.json> <out_dir>"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucrii.cli import raw_table                       # noqa: E402
from ucrii.scoring import load_anchors, score_table   # noqa: E402

EXPECTED = {   # scenario: (reference cohort, {component: expected direction})
    'high_spend_ratio':    ('baseline',        {'PCS': -1, 'FRS': -1, 'LSS': -1, 'UCRII': -1}),
    'low_spend_ratio':     ('baseline',        {'PCS': +1, 'FRS': +1, 'UCRII': +1}),
    'income_disruption':   ('control_noshock', {'PCS': -1, 'LSS': -1, 'FRS': -1, 'UCRII': -1}),
    'expense_spike':       ('control_noshock', {'PCS': -1, 'FRS': -1, 'UCRII': -1}),
    'irregular_recurring': ('baseline',        {'FRS': -1, 'UCRII': -1}),
    'activity_decay':      ('baseline',        {'TFS': -1, 'UCRII': -1}),
    'technical_flaky':     ('baseline',        {'PCS': 0, 'UCRII': 0}),      # technical failures are excluded from PCS
    'sparse':              ('baseline',        {'TFS': -1, 'UCRII': -1}),
}
COMP = ['TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII', 'UCRII_excl_TFS']
EXTRA = ['n_obligations', 'pcs_incl_technical', 'n_successful_debits']


def main(root, anchors_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    A = load_anchors(anchors_path)
    S = {}
    for name in ['baseline', 'control_noshock'] + list(EXPECTED):
        raw, fl = raw_table(os.path.join(root, name))
        S[name] = score_table(raw, fl, A).set_index('customer_id')
        S[name].to_csv(os.path.join(out_dir, f'scores_{name}.csv'))
    rows = []
    for name, (refname, exp) in EXPECTED.items():
        ref, s = S[refname], S[name].reindex(S[refname].index)
        for c in COMP + EXTRA:
            ok = ref[c].notna() & s[c].notna()
            d = (s[c] - ref[c])[ok]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            tol = 0.5 if c in COMP else 1e-9
            rows.append({'scenario': name, 'reference': refname, 'metric': c, 'n_pairs': int(ok.sum()),
                         'ref_mean': ref[c][ok].mean(), 'scenario_mean': s[c][ok].mean(), 'mean_diff': d.mean(),
                         'share_up': float((d > tol).mean()), 'share_tied': float((d.abs() <= tol).mean()),
                         'share_down': float((d < -tol).mean()), 'expected': exp.get(c), 'wilcoxon_p': p})
        fl = lambda df, f: df['flags'].fillna('').str.contains(f).mean()
        rows.append({'scenario': name, 'reference': refname, 'metric': 'share_low_history_flag', 'ref_mean': fl(ref, 'low_history'), 'scenario_mean': fl(s, 'low_history')})
        rows.append({'scenario': name, 'reference': refname, 'metric': 'share_no_obligations_flag', 'ref_mean': fl(ref, 'no_obligations'), 'scenario_mean': fl(s, 'no_obligations')})
        rows.append({'scenario': name, 'reference': refname, 'metric': 'share_TFS_zero', 'ref_mean': (ref.TFS <= .01).mean(), 'scenario_mean': (s.TFS <= .01).mean()})
        rows.append({'scenario': name, 'reference': refname, 'metric': 'share_PCS_zero', 'ref_mean': (ref.PCS <= .01).mean(), 'scenario_mean': (s.PCS <= .01).mean()})
        rows.append({'scenario': name, 'reference': refname, 'metric': 'share_PCS_hundred', 'ref_mean': (ref.PCS >= 99.99).mean(), 'scenario_mean': (s.PCS >= 99.99).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'scenario_comparison.csv'), index=False)
    ir = S['irregular_recurring'].reindex(S['baseline'].index)
    print('paired-difference noise floor (irregular_recurring vs baseline), sd: ' +
          ' | '.join(f'{c} {(ir[c] - S["baseline"][c]).std():.1f}' for c in ['UCRII', 'TFS', 'PCS', 'LSS', 'FRS']))
    return res


if __name__ == '__main__':
    main(*sys.argv[1:4])
