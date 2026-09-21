"""Phase 4c: fairness / bias breakdown + weight robustness on a fresh baseline cohort (several dirs allowed).
Group attributes come from the manifest, the xlsx Profile sheet and _latent/ - VALIDATION ONLY, never used in scoring.
Usage: python validation/run_fairness.py <anchors.json> <out.csv> <cohort_dir> [<cohort_dir> ...]"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucrii.cli import raw_table                                   # noqa: E402
from ucrii.scoring import load_anchors, normalise, score_table    # noqa: E402


def load(anchors, dirs):
    frames = []
    for d in dirs:
        raw, fl = raw_table(d)
        sc = score_table(raw, fl, anchors).set_index('customer_id')
        man = pd.read_csv(f'{d}/manifest.csv').set_index('pan')
        sc['age'] = man.age_group.reindex(sc.index)
        lat = [json.load(open(f'{d}/_latent/{p}.json')) for p in sc.index]
        sc['income_type'] = [l['income_type'] for l in lat]
        sc['tech_rate'] = [l['tech_failure_rate'] for l in lat]
        sc['rho'] = [l['rho'] for l in lat]
        prof = [pd.read_excel(f'{d}/{p}.xlsx', sheet_name='Profile').set_index('Field').Value for p in sc.index]
        sc['state'], sc['bank'], sc['device'] = ([p[k] for p in prof] for k in ('State', 'Bank', 'Device_Type'))
        frames.append(sc)
    return pd.concat(frames)


def main(anchors_path, out_csv, dirs):
    A = load_anchors(anchors_path)
    S = load(A, dirs)
    S.to_csv(out_csv)
    print('n =', len(S), '| UCRII mean %.1f sd %.1f' % (S.UCRII.mean(), S.UCRII.std()))
    for col in ['age', 'income_type', 'state', 'bank', 'device']:
        g = S.groupby(col).UCRII
        H, p = kruskal(*[x.values for _, x in g if len(x) >= 5])
        print(f"\n{col}: Kruskal-Wallis p={p:.3g}, epsilon^2={H / (len(S) - 1):.3f}")
        print(g.agg(['mean', 'count']).round(1).T.to_string())
    print('\ncomponent means by income type:')
    print(S.groupby('income_type')[['TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII', 'UCRII_excl_TFS']].mean().round(1).to_string())
    pb = A['anchors']['pcs_raw']
    # counterfactual: what the paper-literal PCS (technical failures counted) would have done; pcs_raw anchors reused (approximation)
    S['PCS_literal'] = 100 * S.pcs_incl_technical.map(lambda v: normalise(v, pb))
    S['UCRII_literal'] = S.UCRII + 0.25 * (S.PCS_literal - S.PCS)
    S['tech_tercile'] = pd.qcut(S.tech_rate, 3, labels=['low', 'mid', 'high'])
    print('\nconnectivity terciles (latent technical failure rate):')
    print(S.groupby('tech_tercile')[['UCRII', 'PCS', 'pcs_raw', 'pcs_incl_technical', 'UCRII_literal']].mean().round(3).to_string())
    band = lambda u: np.digitize(u, [40, 60, 80])
    print('Spearman(UCRII, tech_rate) %.2f | paper-literal UCRII_literal vs tech_rate %.2f | mean|diff| %.1f | rank corr %.3f | band change %.2f' % (
        spearmanr(S.UCRII, S.tech_rate)[0], spearmanr(S.UCRII_literal, S.tech_rate)[0], (S.UCRII_literal - S.UCRII).abs().mean(),
        spearmanr(S.UCRII, S.UCRII_literal)[0], (band(S.UCRII) != band(S.UCRII_literal)).mean()))
    S['vol_tercile'] = pd.qcut(S.n_successful_debits, 3, labels=['low', 'mid', 'high'])
    print('\nactivity terciles:')
    print(S.groupby('vol_tercile')[['n_successful_debits', 'UCRII', 'UCRII_excl_TFS', 'TFS']].mean().round(1).to_string())
    print('Spearman with volume: UCRII %.2f, UCRII_excl_TFS %.2f' % (spearmanr(S.UCRII, S.n_successful_debits)[0], spearmanr(S.UCRII_excl_TFS, S.n_successful_debits)[0]))
    print('Spearman with latent spend ratio: ' + ' | '.join(f'{c} {spearmanr(S[c], S.rho)[0]:.2f}' for c in ['UCRII', 'UCRII_excl_TFS', 'PCS', 'LSS', 'FRS']))
    w0 = np.array([.25, .25, .20, .15, .15]); X = S[['TFS', 'PCS', 'LSS', 'MDS', 'FRS']].to_numpy(); base = X @ w0
    rng = np.random.default_rng(0); rho, chg = [], []
    for _ in range(2000):
        u = X @ rng.dirichlet(50 * w0); rho.append(spearmanr(base, u)[0]); chg.append((band(u) != band(base)).mean())
    print('\nweight perturbation Dirichlet(50w): rank corr median %.3f (p5 %.3f), band change median %.3f (p95 %.3f)' % (
        np.median(rho), np.percentile(rho, 5), np.median(chg), np.percentile(chg, 95)))
    for i, k in enumerate(['TFS', 'PCS', 'LSS', 'MDS', 'FRS']):
        w = w0.copy(); w[i] = 0; u = X @ (w / w.sum())
        print(f'  without {k}: rank corr {spearmanr(base, u)[0]:.3f}, band change {(band(u) != band(base)).mean():.3f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
