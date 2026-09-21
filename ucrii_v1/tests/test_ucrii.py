import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucrii import config as C                                    # noqa: E402
from ucrii.cli import customer_files, raw_table                  # noqa: E402
from ucrii.features import compute_raw_features, detect_obligations, load_transactions, unexplained_flows  # noqa: E402
from ucrii.scoring import (band, fit_anchors, load_anchors, normalise, save_anchors, score_customer,  # noqa: E402
                           INDICATORS)

WS, WE = '2024-02-01', '2024-03-31'      # two-month window for the small unit tests


def mk(rows, balance=1000.0):
    """rows: dicts with any of ts, dir, ttype, cat, cpty, amt, status, reason, bal. Missing -> defaults."""
    recs = []
    for r in rows:
        recs.append({'Timestamp': pd.Timestamp(r['ts']), 'Direction': r.get('dir', 'DEBIT'),
                     'Transaction_Type': r.get('ttype', 'P2M'), 'Merchant_Category': r.get('cat', 'Food'),
                     'Counterparty_Name': r.get('cpty', 'Shop'), 'Amount': r.get('amt', 100),
                     'Payment_Status': r.get('status', 'SUCCESS'), 'Failure_Reason': r.get('reason'),
                     'Balance_After': r.get('bal', balance)})
    return pd.DataFrame(recs).sort_values('Timestamp', kind='mergesort').reset_index(drop=True)


def feats(rows, ws=WS, we=WE, **kw):
    return compute_raw_features(mk(rows, **kw), ws, we)


# ---------------- TFS / PCS ----------------
def test_tfs_counts_successes_per_month_and_ignores_failures():
    rows = ([{'ts': f'2024-02-0{i}'} for i in (1, 2, 3)] + [{'ts': '2024-03-05'}] +
            [{'ts': '2024-03-06', 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'}])
    f, _ = feats(rows)
    assert f['tfs_raw'] == pytest.approx((3 + 1) / 2)


def test_tfs_counts_debits_only_not_credits():
    rows = [{'ts': '2024-02-01'}, {'ts': '2024-02-02', 'dir': 'CREDIT', 'ttype': 'P2P', 'cat': 'P2P'}]
    f, _ = feats(rows)
    assert f['tfs_raw'] == pytest.approx(0.5)


def test_pcs_attempt_level_retry_is_one_failure_one_success():
    rows = [{'ts': '2024-02-01', 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'},
            {'ts': '2024-02-02'},            # the successful retry
            {'ts': '2024-02-03'}, {'ts': '2024-02-04'}]
    f, _ = feats(rows)
    assert f['pcs_raw'] == pytest.approx(3 / 4)


def test_pcs_repeated_failed_retry_counts_as_second_failure():
    rows = [{'ts': '2024-02-01', 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'},
            {'ts': '2024-02-02', 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'},
            {'ts': '2024-02-03'}, {'ts': '2024-02-04'}]
    f, _ = feats(rows)
    assert f['pcs_raw'] == pytest.approx(0.5)


def test_pcs_excl_technical_diagnostic():
    rows = [{'ts': '2024-02-01', 'status': 'FAILED', 'reason': 'TECHNICAL'}, {'ts': '2024-02-02'},
            {'ts': '2024-02-03'}, {'ts': '2024-02-04'}]
    f, _ = feats(rows)
    assert f['pcs_raw'] == pytest.approx(3 / 4) and f['pcs_excl_technical'] == pytest.approx(1.0)


def test_rows_outside_window_are_ignored():
    rows = [{'ts': '2024-01-15'}, {'ts': '2024-02-10'}, {'ts': '2024-04-02'}]
    f, _ = feats(rows)
    assert f['n_successful_debits'] == 1


# ---------------- MDS ----------------
def test_mds_uniform_over_all_categories_is_one_and_single_category_is_zero():
    rows = [{'ts': '2024-02-01', 'cat': c, 'cpty': f'M{i}'} for i, c in enumerate(C.CATEGORIES) for _ in range(2)]
    f, _ = feats(rows)
    assert f['mds_raw'] == pytest.approx(1.0)
    f2, _ = feats([{'ts': '2024-02-01', 'cat': 'Food'}] * 5)
    assert f2['mds_raw'] == pytest.approx(0.0)


def test_mds_excludes_p2p_and_failed_rows():
    rows = [{'ts': '2024-02-01', 'cat': 'Food'}, {'ts': '2024-02-02', 'cat': 'Grocery'},
            {'ts': '2024-02-03', 'ttype': 'P2P', 'cat': 'P2P', 'cpty': 'Friend'},
            {'ts': '2024-02-04', 'cat': 'Fuel', 'status': 'FAILED', 'reason': 'TECHNICAL'}]
    f, _ = feats(rows)
    assert f['mds_raw'] == pytest.approx(np.log(2) / np.log(11))
    assert f['n_unique_merchants'] == 1     # both merchant rows use the default counterparty 'Shop'


def test_no_merchant_payments_flagged():
    _, flags = feats([{'ts': '2024-02-01', 'ttype': 'P2P', 'cat': 'P2P', 'cpty': 'Friend'}])
    assert 'no_merchant_payments' in flags


# ---------------- LSS ----------------
def test_l1_zero_for_constant_balance_and_positive_when_it_varies():
    rows = [{'ts': '2024-02-10'}, {'ts': '2024-03-10'}]
    f, _ = feats(rows, balance=5000.0)
    assert f['L1_balance_cv'] == pytest.approx(0.0)
    rows2 = [{'ts': '2024-02-10', 'bal': 1000.0}, {'ts': '2024-03-10', 'bal': 3000.0}]
    f2, _ = feats(rows2)
    assert f2['L1_balance_cv'] > 0.3


def test_unexplained_flow_recovers_hidden_credit():
    df = mk([{'ts': '2024-02-01', 'amt': 100, 'bal': 900.0},
             {'ts': '2024-02-02', 'amt': 100, 'bal': 5800.0}])    # 900 - 100 + hidden 5000 = 5800
    u = unexplained_flows(df)
    assert np.isnan(u[0]) and u[1] == pytest.approx(5000.0)


def test_failed_row_does_not_create_unexplained_flow():
    df = mk([{'ts': '2024-02-01', 'amt': 100, 'bal': 900.0},
             {'ts': '2024-02-02', 'amt': 500, 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS', 'bal': 900.0}])
    assert unexplained_flows(df)[1] == pytest.approx(0.0)


def test_l2_inflow_includes_upi_credits_and_hidden_jumps_and_flags_missing_inflow():
    rows = [{'ts': '2024-02-01', 'amt': 100, 'bal': 900.0},
            {'ts': '2024-02-15', 'amt': 100, 'bal': 5800.0},                                        # hidden +5000 in Feb
            {'ts': '2024-03-05', 'dir': 'CREDIT', 'ttype': 'P2P', 'cat': 'P2P', 'amt': 5000, 'bal': 10800.0}]  # UPI credit in Mar
    f, flags = feats(rows)
    assert f['L2_inflow_cv'] == pytest.approx(0.0, abs=1e-9) and 'no_inflow' not in flags
    _, flags2 = feats([{'ts': '2024-02-01', 'bal': 900.0}, {'ts': '2024-03-01', 'bal': 800.0}])
    assert 'no_inflow' in flags2


# ---------------- FRS ----------------
def _monthly(name, months, cat='Entertainment', ttype='P2M', day=10, amt=199, **extra):
    return [{'ts': f'2024-{m:02d}-{day:02d}', 'cat': cat, 'ttype': ttype, 'cpty': name, 'amt': amt, **extra} for m in months]


def test_obligation_detected_and_retry_recovers_cycle():
    rows = _monthly('Netflix', range(2, 13))
    rows[0] = {**rows[0], 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'}
    rows.insert(1, {'ts': '2024-02-12', 'cat': 'Entertainment', 'ttype': 'P2M', 'cpty': 'Netflix', 'amt': 199})   # retry
    f, flags = feats(rows, ws='2024-02-01', we='2024-12-31')
    assert f['n_obligations'] == 1 and f['F1_obligation_continuity'] == pytest.approx(1.0)
    assert 'no_obligations' not in flags


def test_unrecovered_cycle_lowers_continuity():
    rows = _monthly('Netflix', range(2, 13))
    rows[3] = {**rows[3], 'status': 'FAILED', 'reason': 'INSUFFICIENT_FUNDS'}      # one cycle with no success
    f, _ = feats(rows, ws='2024-02-01', we='2024-12-31')
    assert f['F1_obligation_continuity'] == pytest.approx(10 / 11)


def test_rent_detected_only_with_constant_amount():
    rent = _monthly('Landlord', range(2, 13), cat='P2P', ttype='P2P', day=3, amt=9000)
    assert len(detect_obligations(mk(rent))) == 1
    varying = [{**r, 'amt': 5000 + 900 * i} for i, r in enumerate(rent)]
    assert detect_obligations(mk(varying)) == []


def test_irregular_or_short_history_is_not_an_obligation():
    assert detect_obligations(mk(_monthly('Spotify', [2, 3, 4]))) == []
    irregular = [{'ts': d, 'cat': 'Entertainment', 'ttype': 'P2M', 'cpty': 'BookMyShow', 'amt': 300}
                 for d in ('2024-02-01', '2024-02-20', '2024-05-02', '2024-05-30', '2024-08-15', '2024-09-01', '2024-12-20')]
    assert detect_obligations(mk(irregular)) == []


def test_no_obligations_flag_and_low_history_flag():
    _, flags = feats([{'ts': '2024-02-01'}, {'ts': '2024-02-02', 'cat': 'Grocery'}])
    assert 'no_obligations' in flags and 'low_history' in flags


def test_f2_buffer_share_of_low_balance_days():
    # outflow 100/month-window(2 months)=... balance 0 on all days -> every day is low
    f, _ = feats([{'ts': '2024-02-10', 'bal': 0.0}, {'ts': '2024-03-10', 'bal': 0.0}], balance=0.0)
    assert f['F2_buffer_adequacy'] == pytest.approx(0.0)
    f2, _ = feats([{'ts': '2024-02-10'}, {'ts': '2024-03-10'}], balance=1e6)
    assert f2['F2_buffer_adequacy'] == pytest.approx(1.0)


# ---------------- normalisation / anchors / scoring ----------------
def _raw_cohort(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'tfs_raw': rng.uniform(20, 80, n), 'pcs_raw': rng.uniform(0.85, 1.0, n),
                         'L1_balance_cv': rng.uniform(0.1, 0.7, n), 'L2_inflow_cv': rng.uniform(0.0, 0.6, n),
                         'mds_raw': rng.uniform(0.85, 0.92, n), 'F1_obligation_continuity': rng.uniform(0.95, 1.0, n),
                         'F2_buffer_adequacy': rng.uniform(0.7, 1.0, n)})


def test_fit_anchors_modes_follow_the_no_stretch_guard():
    a = fit_anchors(_raw_cohort())['anchors']
    assert a['tfs_raw']['mode'] == 'minmax' and a['pcs_raw']['mode'] == 'minmax' and a['F2_buffer_adequacy']['mode'] == 'minmax'
    assert a['mds_raw']['mode'] == 'native' and a['F1_obligation_continuity']['mode'] == 'native'
    assert a['L1_balance_cv']['mode'] == 'minmax'      # unbounded indicators are never native


def test_normalise_clips_and_native_passthrough():
    spec = {'mode': 'minmax', 'a': 10.0, 'b': 20.0}
    assert normalise(5, spec) == 0.0 and normalise(25, spec) == 1.0 and normalise(15, spec) == pytest.approx(0.5)
    assert normalise(0.88, {'mode': 'native'}) == pytest.approx(0.88) and np.isnan(normalise(np.nan, spec))


def test_anchor_json_roundtrip(tmp_path):
    anc = fit_anchors(_raw_cohort())
    p = tmp_path / 'a.json'
    save_anchors(anc, str(p))
    assert load_anchors(str(p))['anchors'] == json.loads(json.dumps(anc['anchors']))


def test_fit_anchors_rejects_tiny_or_degenerate_reference():
    with pytest.raises(ValueError):
        fit_anchors(_raw_cohort(n=10))
    bad = _raw_cohort()
    bad['tfs_raw'] = 50.0
    with pytest.raises(ValueError):
        fit_anchors(bad)


def test_weights_bands_and_composite_arithmetic():
    assert sum(C.WEIGHTS.values()) == pytest.approx(1.0)
    assert [band(s) for s in (100, 80, 79.6, 60, 59.9, 40, 39.9, 0)] == \
        ['Low risk', 'Low risk', 'Moderate risk', 'Moderate risk', 'High risk', 'High risk', 'Very high risk', 'Very high risk']
    anchors = fit_anchors(_raw_cohort())
    a = anchors['anchors']
    best = {'tfs_raw': a['tfs_raw']['b'], 'pcs_raw': a['pcs_raw']['b'], 'L1_balance_cv': a['L1_balance_cv']['a'],
            'L2_inflow_cv': a['L2_inflow_cv']['a'], 'mds_raw': 1.0, 'F1_obligation_continuity': 1.0,
            'F2_buffer_adequacy': a['F2_buffer_adequacy']['b']}
    s = score_customer(best, [], anchors)
    assert s['UCRII'] == pytest.approx(100.0) and s['band'] == 'Low risk'
    worst = {'tfs_raw': a['tfs_raw']['a'], 'pcs_raw': a['pcs_raw']['a'], 'L1_balance_cv': a['L1_balance_cv']['b'],
             'L2_inflow_cv': a['L2_inflow_cv']['b'], 'mds_raw': 0.0, 'F1_obligation_continuity': 0.0,
             'F2_buffer_adequacy': a['F2_buffer_adequacy']['a']}
    s0 = score_customer(worst, [], anchors)
    assert s0['UCRII'] == pytest.approx(0.0) and s0['band'] == 'Very high risk'
    assert sum(s[f'contrib_{k}'] for k in C.WEIGHTS) == pytest.approx(s['UCRII'])


def test_frs_falls_back_to_buffer_alone_without_obligations_and_lss_undefined_cv_is_worst():
    anchors = fit_anchors(_raw_cohort())
    a = anchors['anchors']
    base = {'tfs_raw': 50, 'pcs_raw': 0.95, 'L1_balance_cv': 0.3, 'L2_inflow_cv': 0.2, 'mds_raw': 0.88,
            'F1_obligation_continuity': np.nan, 'F2_buffer_adequacy': a['F2_buffer_adequacy']['b']}
    s = score_customer(base, ['no_obligations'], anchors)
    assert s['FRS'] == pytest.approx(100.0)
    s2 = score_customer({**base, 'L2_inflow_cv': np.nan}, ['no_inflow'], anchors)
    l1 = 1 - normalise(0.3, a['L1_balance_cv'])
    assert s2['LSS'] == pytest.approx(100 * (l1 + 0.0) / 2)


def test_missing_core_indicator_gives_insufficient_data_not_a_score():
    anchors = fit_anchors(_raw_cohort())
    s = score_customer({k: np.nan for k in INDICATORS}, ['no_debits'], anchors)
    assert s['status'] == 'insufficient_data' and np.isnan(s['UCRII'])


def test_score_is_monotone_in_each_indicator():
    anchors = fit_anchors(_raw_cohort())
    base = {'tfs_raw': 50, 'pcs_raw': 0.95, 'L1_balance_cv': 0.4, 'L2_inflow_cv': 0.3, 'mds_raw': 0.88,
            'F1_obligation_continuity': 0.98, 'F2_buffer_adequacy': 0.9}
    u0 = score_customer(base, [], anchors)['UCRII']
    for ind, delta in [('tfs_raw', +10), ('pcs_raw', +0.03), ('L1_balance_cv', -0.1), ('L2_inflow_cv', -0.1),
                       ('mds_raw', +0.02), ('F1_obligation_continuity', +0.01), ('F2_buffer_adequacy', +0.05)]:
        assert score_customer({**base, ind: base[ind] + delta}, [], anchors)['UCRII'] > u0, ind


# ---------------- loader / isolation ----------------
def test_loader_rejects_missing_columns(tmp_path):
    p = tmp_path / 'X.csv'
    pd.DataFrame({'Timestamp': ['2024-02-01'], 'Amount': [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_transactions(p)


def test_cohort_reader_ignores_manifest_latent_and_xlsx(tmp_path):
    good = mk([{'ts': '2024-02-01'}, {'ts': '2024-02-02', 'cat': 'Grocery'}])
    good.to_csv(tmp_path / 'AAAPA1111A.csv', index=False)
    (tmp_path / 'manifest.csv').write_text('pan,age_group\nAAAPA1111A,26-35\n')
    (tmp_path / '_latent').mkdir()
    (tmp_path / '_latent' / 'BBBPB2222B.csv').write_text('this,is,not,a,transaction,file\n')
    (tmp_path / 'AAAPA1111A.xlsx').write_bytes(b'not an excel file')
    assert [p.name for p in customer_files(tmp_path)] == ['AAAPA1111A.csv']
    raw, flags = raw_table(tmp_path, WS, WE)
    assert list(raw.index) == ['AAAPA1111A']
    assert set(raw.columns).isdisjoint({'age_group', 'Age', 'State', 'Bank', 'Name'})


def test_features_are_deterministic():
    rows = _monthly('Netflix', range(2, 13)) + [{'ts': '2024-05-05', 'cat': 'Grocery', 'cpty': 'DMart'}]
    a, fa = feats(rows, ws='2024-02-01', we='2024-12-31')
    b, fb = feats(rows, ws='2024-02-01', we='2024-12-31')
    assert fa == fb and all((a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)
