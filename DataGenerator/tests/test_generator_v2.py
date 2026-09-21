"""Tests for the v2 ledger-based UPI generator and cohort driver."""
import functools
import importlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

import driver
import upi_user_generator as g

HERE = os.path.dirname(os.path.abspath(__file__))
AGES = driver.AGE_GROUPS
SEEDS = range(30)


@functools.lru_cache(maxsize=None)
def gen(age='26-35', seed=1, n=500, overrides=None):
    return g.generate_user_transactions(age, n, '2024-01-01', '2024-12-31', seed=seed,
                                        overrides=dict(overrides) if overrides else None, return_profile=True)


def ov(**kw):
    return tuple(sorted(kw.items()))


def all_customers():
    return [(a, s, *gen(a, s)) for a in AGES for s in SEEDS]


# ---------------- untouched original functions ----------------
def test_get_amt_and_hourly_unchanged():
    snap = json.load(open(os.path.join(HERE, 'fixtures', 'get_amt_snapshot.json')))
    np.random.seed(0)
    for key in snap['get_amt']:            # same order as the snapshot was generated in
        age, cat = key.split('|')
        assert [int(x) for x in g.get_amt([cat] * 20, age)] == snap['get_amt'][key], key
    assert [float(x) for x in g.hourly_distribution()] == snap['hourly_distribution']


# ---------------- schema ----------------
def test_schema_and_no_leakage():
    df, prof = gen()
    assert list(df.columns) == g.OUTPUT_COLUMNS
    assert df.attrs['generator_version'] == '2.0'
    banned = {'Customer_ID', 'User_ID', 'isRecurring', 'Sender_Age_Group', 'Fraud_Flag', 'Income', 'Salary',
              'Receiver_Name', 'Receiver_Bank', 'Sender_Bank'}
    assert not banned & set(df.columns)
    assert not [c for c in df.columns if 'score' in c.lower()]
    latent = {'income_type', 'rho', 'affluence', 'spend_ratio', 'h', 'resilience', 'shock', 'pay_day'}
    assert not latent & set(df.columns)
    assert prof['income_type'] in g.INCOME_TYPES


def test_debit_only_device_network_and_credit_types():
    for age, s, df, _ in all_customers()[::7]:
        cre, deb = df[df.Direction == 'CREDIT'], df[df.Direction == 'DEBIT']
        assert cre[['Device_Type', 'Network_Type']].isna().all().all()
        assert deb[['Device_Type', 'Network_Type']].notna().all().all()
        assert set(cre.Transaction_Type) <= {'P2P', 'P2M', 'Refund'}
        assert set(deb.Transaction_Type) <= {'P2P', 'P2M', 'Bill Payment', 'Recharge'}
        assert set(cre[cre.Transaction_Type == 'P2M'].Merchant_Category) <= {'Collection'}


def test_determinism_and_override_validation():
    a = g.generate_user_transactions('36-45', 300, seed=5, overrides={'spend_ratio': 1.1})
    b = g.generate_user_transactions('36-45', 300, seed=5, overrides={'spend_ratio': 1.1})
    c = g.generate_user_transactions('36-45', 300, seed=6, overrides={'spend_ratio': 1.1})
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    with pytest.raises(ValueError):
        g.generate_user_transactions('26-35', 100, overrides={'not_a_key': 1})


# ---------------- ledger invariants ----------------
def test_ledger_invariants_all_ages():
    for age, s, df, prof in all_customers():
        tag = f"{age}/{s}"
        assert (df.Balance_After >= 0).all(), tag
        assert (df.loc[df.Direction == 'CREDIT', 'Payment_Status'] == 'SUCCESS').all(), tag
        assert (df.Failure_Reason.isna() == (df.Payment_Status == 'SUCCESS')).all(), tag
        assert set(df.Failure_Reason.dropna()) <= {'INSUFFICIENT_FUNDS', 'TECHNICAL'}, tag
        assert df.Timestamp.is_monotonic_increasing, tag
        isf = df[df.Failure_Reason == 'INSUFFICIENT_FUNDS']
        assert (isf.Amount > isf.Balance_After).all(), tag
        bad, worst = g.check_reconciliation(df, prof)
        assert bad == 0, f"{tag}: worst residual {worst}"
        # no successful debit exceeds the prior balance (prior balance + hidden flows applied just before it)
        net = np.zeros(len(df) + 1)
        for h in prof['hidden_events']:
            net[h['after_row']] += h['amount']
        prior = np.concatenate([[prof['opening_balance']], df.Balance_After.to_numpy()[:-1]]) + net[:len(df)]
        ok_deb = ((df.Direction == 'DEBIT') & (df.Payment_Status == 'SUCCESS')).to_numpy()
        assert (df.Amount.to_numpy()[ok_deb] <= prior[ok_deb] + 0.01).all(), tag


def test_refunds_follow_a_successful_debit():
    n = 0
    for age, s, df, _ in all_customers():
        for r in df[df.Transaction_Type == 'Refund'].itertuples():
            src = df[(df.Direction == 'DEBIT') & (df.Payment_Status == 'SUCCESS') & (df.Counterparty_Name == r.Counterparty_Name)
                     & (df.Merchant_Category == r.Merchant_Category) & (df.Timestamp < r.Timestamp)]
            assert len(src), (age, s, r)
            n += 1
    assert n > 0


def test_retry_chains_exist_and_are_bounded():
    chains = [c for _, _, _, p in all_customers() for c in p['retry_chains']]
    assert chains
    assert max(c['attempts'] for c in chains) <= 1 + g.MAX_RETRIES
    assert any(c['outcome'] == 'FAILED' for c in chains)   # retries are not guaranteed to succeed
    assert any(c['outcome'] == 'SUCCESS' for c in chains)


def test_row_count_sanity():
    for age, s, df, prof in all_customers():
        deb, cre = (df.Direction == 'DEBIT').sum(), (df.Direction == 'CREDIT').sum()
        assert len(df) > prof['planned_debit_attempts'], (age, s)
        assert cre <= g.MAX_CREDIT_ROW_RATIO * deb, (age, s, cre, deb)


def test_merchant_credit_guard_at_low_activity():
    df, prof = g.generate_user_transactions('36-45', 200, seed=3, overrides={'income_type': 'merchant'}, return_profile=True)
    assert (df.Direction == 'CREDIT').sum() <= g.MAX_CREDIT_ROW_RATIO * (df.Direction == 'DEBIT').sum()


# ---------------- pay-cycle recovery ----------------
def test_pay_cycle_recovery_from_balances_alone():
    hits, total = 0, 0
    for age, s, df, prof in all_customers():
        if prof['income_type'] not in ('salaried', 'pensioner'):
            continue
        total += 1
        day = g.infer_pay_cycle_day(df)
        hits += day is not None and g._circ_dist(day, prof['pay_day']) <= 3
    assert total >= 50
    assert hits / total >= 0.80, f"{hits}/{total}"


# ---------------- scenario directions (aggregated over 20 seeds) ----------------
def _mean(fn, **kw):
    return np.mean([fn(*gen('26-35', s, 500, ov(**kw))) for s in range(20)])


def test_spend_ratio_drives_insufficient_funds():
    isf = lambda df, p: (df.Failure_Reason == 'INSUFFICIENT_FUNDS').sum()
    assert _mean(isf, spend_ratio=1.15) > _mean(isf, spend_ratio=0.60)


def test_income_disruption_lowers_window_min_balance():
    def window_min(df, p):
        s = p['shock']
        w = df[(df.Timestamp >= s['start']) & (df.Timestamp < s['end'] + pd.DateOffset(months=1))]
        return w.Balance_After.min() / p['monthly_income'] if len(w) else np.nan
    seeds = [s for s in range(40) if gen('26-35', s, 500, ov(shock='none'))[1]['income_type'] != 'pensioner'][:20]
    dis = np.nanmean([window_min(*gen('26-35', s, 500, ov(shock='income_disruption'))) for s in seeds])
    non = np.nanmean([window_min(*gen('26-35', s, 500, ov(shock='none'))) for s in seeds])
    assert dis < non


def test_tech_failure_rate_override():
    tech = lambda df, p: (df.Failure_Reason == 'TECHNICAL').sum()
    assert _mean(tech, tech_failure_rate=0.08) > _mean(tech, tech_failure_rate=0.005)


# ---------------- recurring / rent ----------------
def test_detect_recurring_finds_rent_and_services():
    rent = {'stable': [0, 0], 'gig': [0, 0]}
    svc_found = svc_total = 0
    for age, s, df, prof in all_customers():
        det = g.detect_recurring(df)
        if prof['has_rent']:
            r = rent['gig' if prof['income_type'] == 'gig' else 'stable']
            r[1] += 1
            r[0] += prof['landlord_name'] in det
        for name in prof['receiver_name'].values():
            if name in ('Electricity', 'Gas'):
                continue
            svc_total += 1
            svc_found += name in det
    # lumpy-income (gig) customers sometimes miss rent for lack of funds, which legitimately breaks regularity
    assert rent['stable'][1] > 15 and rent['stable'][0] / rent['stable'][1] >= 0.95, rent
    assert (rent['stable'][0] + rent['gig'][0]) / (rent['stable'][1] + rent['gig'][1]) >= 0.80, rent
    assert svc_total > 100 and svc_found / svc_total >= 0.85


def test_rent_is_p2p_to_landlord_outside_circle():
    n = 0
    for age, s, df, prof in all_customers():
        if prof['has_rent']:
            assert prof['landlord_name'] not in prof['circle']
            r = df[(df.Counterparty_Name == prof['landlord_name']) & (df.Direction == 'DEBIT')]
            assert set(r.Transaction_Type) == {'P2P'} and set(r.Merchant_Category) == {'P2P'}
            n += 1
    assert n > 0


def test_rng_alignment_across_overrides():
    np.random.seed(11)
    p0 = g.build_user_profile('26-35')
    np.random.seed(11)
    p1 = g.build_user_profile('26-35', {'spend_ratio': 1.3, 'shock': 'none', 'has_rent': True})
    assert p0['circle'] == p1['circle'] and p0['recurring_services'] == p1['recurring_services']
    assert p0['h'] == p1['h'] and p0['landlord_name'] == p1['landlord_name'] and p0['resilience'] == p1['resilience']
    assert p1['rho'] == 1.3 and p1['has_rent']


# ---------------- driver ----------------
def run_cohort(tmp_path, name='c', n=6, **kw):
    out = str(tmp_path / name)
    mdf, summary = driver.run(n, outdir=out, verbose=False, **kw)
    return out, mdf, summary


def test_import_driver_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['pytest', 'not-an-int', '--bogus'])
    importlib.reload(driver)
    assert os.listdir(tmp_path) == []


def test_cohort_files_manifest_latent(tmp_path):
    out, mdf, _ = run_cohort(tmp_path)
    csvs = [f for f in os.listdir(out) if f.endswith('.csv') and f != 'manifest.csv']
    assert len(csvs) == 6 and all(re.match(r'^[A-Z]{3}P[A-Z][0-9]{4}[A-Z]\.csv$', f) for f in csvs)
    m = pd.read_csv(os.path.join(out, 'manifest.csv'))
    assert list(m.columns) == driver.MANIFEST_COLUMNS and len(m) == 6
    latent = [f for f in os.listdir(os.path.join(out, '_latent')) if f.endswith('.json')]
    assert len(latent) == 6
    assert 'VALIDATION ONLY' in open(os.path.join(out, '_latent', 'README.txt')).read()
    assert 'synthetic' in open(os.path.join(out, 'README.txt')).read().lower()
    forbidden = {'Customer_ID', 'PAN', 'income_type', 'rho', 'affluence', 'spend_ratio'}
    for f in csvs:
        assert not forbidden & set(pd.read_csv(os.path.join(out, f), nrows=1).columns)
    j = json.load(open(os.path.join(out, '_latent', latent[0])))
    assert 'income_type' in j and 'shock' in j


def test_driver_seed_reproducibility(tmp_path):
    o1, m1, _ = run_cohort(tmp_path, 'a', seed=7)
    o2, m2, _ = run_cohort(tmp_path, 'b', seed=7)
    o3, m3, _ = run_cohort(tmp_path, 'c', seed=8)
    pd.testing.assert_frame_equal(m1, m2)
    for pan in m1.pan:
        assert open(os.path.join(o1, pan + '.csv'), 'rb').read() == open(os.path.join(o2, pan + '.csv'), 'rb').read()
    assert set(m1.pan).isdisjoint(m3.pan)


def test_scenarios_share_identity_but_differ(tmp_path):
    ob, base, _ = run_cohort(tmp_path, 'base', scenario='baseline')
    oh, hsr, _ = run_cohort(tmp_path, 'hsr', scenario='high_spend_ratio')
    key = ['pan', 'age_group', 'seed', 'n_records_target']
    pd.testing.assert_frame_equal(base[key], hsr[key])
    assert base.overrides_json.iloc[0] == '{}' and 'spend_ratio' in hsr.overrides_json.iloc[0]
    assert any(open(os.path.join(ob, p + '.csv'), 'rb').read() != open(os.path.join(oh, p + '.csv'), 'rb').read() for p in base.pan)
    # sparse consumes the same master draws: PANs / ages / seeds unchanged, fewer records
    _, sp, _ = run_cohort(tmp_path, 'sp', scenario='sparse')
    pd.testing.assert_frame_equal(base[['pan', 'age_group', 'seed']], sp[['pan', 'age_group', 'seed']])
    assert sp.n_records_target.between(60, 150).all()


def test_explicit_override_beats_preset_and_unknown_raises(tmp_path):
    _, m, _ = run_cohort(tmp_path, 'o', scenario='high_spend_ratio', overrides={'spend_ratio': 0.7})
    assert json.loads(m.overrides_json.iloc[0])['spend_ratio'] == 0.7
    with pytest.raises(ValueError):
        run_cohort(tmp_path, 'bad', overrides={'bogus': 1})
    assert driver.parse_overrides(['a=1', 'b=2.5', 'c=true', 'd=x']) == {'a': 1, 'b': 2.5, 'c': True, 'd': 'x'}


def test_cli_positional_and_flags(tmp_path):
    out = str(tmp_path / 'cli')
    mdf, _ = driver.main(['3', '--outdir', out, '--age', '56+', '--override', 'has_rent=false'])
    assert len(mdf) == 3 and set(mdf.age_group) == {'56+'}


def test_summary_failed_share_is_over_debit_rows_only(tmp_path):
    out, mdf, summary = run_cohort(tmp_path, 's', n=8)
    for age, a in summary['by_age'].items():
        failed = deb = 0
        for pan in mdf[mdf.age_group == age].pan:
            df = pd.read_csv(os.path.join(out, pan + '.csv'))
            d = df[df.Direction == 'DEBIT']
            failed += (d.Payment_Status == 'FAILED').sum()
            deb += len(d)
        assert a['failed_share'] == pytest.approx(failed / deb)
    assert sum(pd.read_csv(os.path.join(out, p + '.csv')).shape[0] for p in mdf.pan) == summary['total_rows']
