import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'Synthetic_UPI_Txn_Generator'))

import driver                                                   # noqa: E402
from upi_user_generator import generate_user_transactions       # noqa: E402
from ucrii import config as C                                    # noqa: E402
from ucrii.explain import explain, window_slices                 # noqa: E402
from ucrii.features import compute_raw_features                 # noqa: E402
from ucrii.insights import insights, what_if                     # noqa: E402
from ucrii.scoring import load_anchors, score_customer           # noqa: E402
from ui import services as S                                     # noqa: E402

ANCH = load_anchors(ROOT / 'ucrii' / 'anchors_reference_v1_1.json')
BANNED = re.compile(r'\b(irresponsible|bad|poor|poorly|risky behaviou?r|reckless|careless|worst|blame|fault)\b', re.I)
WS, WE = C.WINDOW_START, C.WINDOW_END


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from strings(v)


@pytest.fixture(scope='module')
def customers():
    out = []
    presets = [p for p in driver.SCENARIOS if p != 'sparse']
    for i in range(12):
        n = 100 if i % 6 == 5 else 250 + 60 * (i % 4)
        out.append(generate_user_transactions(driver.AGE_GROUPS[i % 5], n, driver.START_DATE, driver.END_DATE, seed=900 + i,
                                              overrides=dict(driver.SCENARIOS[presets[i % len(presets)]])))
    return out


def test_json_and_no_mutation_and_banned_words(customers):
    for df in customers:
        before = df.copy()
        r = S._clean(insights(df, WS, WE))
        json.dumps(r)
        pd.testing.assert_frame_equal(df, before)
        for s in strings(r):
            assert not BANNED.search(s), s
        assert set(r['readings']) == {'affordability', 'income', 'servicing', 'liquidity', 'stress', 'direction', 'mix', 'sufficiency'}
        assert all(re.search(r'\d', t) or k == 'servicing' for k, t in r['readings'].items())


def test_identities_against_transactions_and_scorer(customers):
    for df in customers:
        r = insights(df, WS, WE)
        raw, flags = compute_raw_features(df)
        _, _, ok = window_slices(df, WS, WE)
        a = r['affordability']
        n = r['sufficiency']['n_months']
        assert abs(a['mean_monthly_spend'] * n - ok['Amount'].sum()) < 1e-6
        assert abs(sum(m['Net'] for m in a['monthly']) - sum(m['Inflow'] - m['Spent'] for m in a['monthly'])) < 1e-6
        assert abs(sum(g['Share'] for g in r['mix']['groups']) - 1) < 1e-9
        assert r['sufficiency']['successful_debits'] == raw['n_successful_debits']
        # obligation servicing must agree with the scorer's own obligation detection
        sc = score_customer(raw, flags, ANCH)
        if sc['status'] == 'ok':
            f = explain(df, WS, WE, raw, flags, sc, ANCH)['facts']
            assert r['servicing']['total_cycles'] == f['obligation_cycles']
            assert r['servicing']['paid_cycles'] == f['obligation_paid_cycles']
            assert abs((1 - (r['liquidity']['low_balance_days'] / f['n_days'])) - raw['F2_buffer_adequacy']) < 1e-9
        lo, hi = r['liquidity']['low_balance_episodes'], r['liquidity']['low_balance_days']
        assert (lo == 0) == (hi == 0) and r['liquidity']['longest_low_run_days'] <= hi


def test_no_obligations_customer(customers):
    df = customers[0]
    df = df[~(df['Merchant_Category'].isin(C.OBLIGATION_CATEGORIES) | (df['Transaction_Type'] == 'P2P'))].reset_index(drop=True)
    r = insights(df, WS, WE)
    assert r['servicing']['total_cycles'] == 0 and r['affordability']['fixed_payment_load'] == 0.0
    assert 'No recurring obligations' in r['readings']['servicing']


def test_what_if_behaviour(customers):
    for df in customers[:4]:
        w0 = what_if(df, WS, WE, 0)
        assert (w0['series']['Base'] == w0['series']['With instalment']).all()
        assert w0['with_instalment_low_days'] == w0['base_low_days'] and w0['days_balance_negative'] == 0
        prev = None
        for inst in (1000, 5000, 20000, 60000):
            w = what_if(df, WS, WE, inst)
            assert w['with_instalment_low_days'] >= w['base_low_days']
            if prev is not None:
                assert w['days_balance_negative'] >= prev['days_balance_negative'] and w['lowest_balance'] <= prev['lowest_balance']
            prev = w
        assert prev['first_shortfall_date'] is not None and prev['months_with_shortfall'] >= 1
        # first instalment falls on the due day of the first window month: balance is unchanged before it
        s = what_if(df, WS, WE, 5000, due_day=15)['series']
        assert (s.loc[s['Date'] < '2024-02-15', 'Base'] == s.loc[s['Date'] < '2024-02-15', 'With instalment']).all()


def test_services_wrappers(tmp_path):
    S.score_pan('ABCPD1234E', data=tmp_path)
    p = tmp_path / 'ABCPD1234E.csv'
    assert S.credit_signals(p)['sufficiency']['n_months'] == 11
    assert S.instalment_what_if(p, 3000)['instalment'] == 3000.0
