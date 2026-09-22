import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'Synthetic_UPI_Txn_Generator'))

import driver                                                   # noqa: E402
from upi_user_generator import generate_user_transactions       # noqa: E402
from ucrii import config as C                                    # noqa: E402
from ucrii.explain import explain, rank_drivers                  # noqa: E402
from ucrii.features import compute_raw_features, load_transactions   # noqa: E402
from ucrii.scoring import load_anchors, score_customer           # noqa: E402

ANCH = load_anchors(ROOT / 'ucrii' / 'anchors_reference_v1_1.json')
BANNED = re.compile(r'\b(irresponsible|bad|poor|poorly|risky behaviou?r|reckless|careless|worst|blame|fault)\b', re.I)
NO_EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')


def run(df):
    raw, flags = compute_raw_features(df)
    scores = score_customer(raw, flags, ANCH)
    return raw, flags, scores, explain(df, C.WINDOW_START, C.WINDOW_END, raw, flags, scores, ANCH)


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
    """32 generated customers of different types: age, activity, scenario presets."""
    out = []
    presets = [p for p in driver.SCENARIOS if p != 'sparse']
    for i in range(32):
        thin = i % 8 == 7
        n = 100 if thin else 250 + 40 * (i % 5)
        df = generate_user_transactions(driver.AGE_GROUPS[i % 5], n, driver.START_DATE, driver.END_DATE, seed=500 + i,
                                        overrides=dict(driver.SCENARIOS[presets[i % len(presets)]]))
        out.append(run(df) + (df,))
    return out


def test_deterministic_and_json(customers):
    raw, flags, scores, e, df = customers[0]
    again = explain(df, C.WINDOW_START, C.WINDOW_END, raw, flags, scores, ANCH)
    assert again == e
    json.dumps(e)


def test_contributions_sum_to_ucrii(customers):
    for raw, flags, scores, e, df in customers:
        if scores['status'] != 'ok':
            continue
        assert abs(sum(c['contribution'] for c in e['components'].values()) - scores['UCRII']) < 1e-9
        assert f"{scores['UCRII']:.1f}" in e['headline']


def test_no_moralised_language_or_emoji(customers):
    assert len(customers) >= 30
    for *_, e, _df in customers:
        for s in strings(e):
            assert not BANNED.search(s), s
            assert not NO_EMOJI.search(s), s


def test_level_statements_contain_numbers(customers):
    for raw, flags, scores, e, df in customers:
        if scores['status'] != 'ok':
            continue
        for k, c in e['components'].items():
            assert re.search(r'\d', c['text']), k
        for d in e['pulls_down'] + e['holds_up']:
            assert re.search(r'\d', d['text']) or d['component'] is None


def test_caveats_follow_flags(customers):
    seen_low = seen_obl = False
    for raw, flags, scores, e, df in customers:
        if scores['status'] != 'ok':
            continue
        cav = ' '.join(e['caveats'])
        assert ('Limited history' in cav) == ('low_history' in flags)
        assert ('No recurring obligations' in cav) == ('no_obligations' in flags)
        assert 'synthetic' in cav and 'probability of default' in cav
        seen_low |= 'low_history' in flags
    assert seen_low                                            # the thin-file customers exercised the branch


def test_no_obligations_fallback_wording(customers):
    df = customers[0][4]
    df = df[~(df['Merchant_Category'].isin(C.OBLIGATION_CATEGORIES) | (df['Transaction_Type'] == 'P2P'))]
    raw, flags, scores, e = run(df.reset_index(drop=True))
    assert 'no_obligations' in flags and scores['status'] == 'ok'
    assert 'No recurring payments detected' in e['components']['FRS']['text']
    assert any('No recurring obligations' in c for c in e['caveats'])


def test_component_facts_match_raw(customers):
    raw, flags, scores, e, df = customers[1]
    f = e['facts']
    assert f['n_successful_debits'] == raw['n_successful_debits'] and f['n_attempts'] == raw['n_attempts']
    assert f['n_obligations'] == raw['n_obligations']
    assert abs(f['n_successful_debits'] / f['n_counted_attempts'] - raw['pcs_raw']) < 1e-12


def test_driver_ordering_rule():
    s = {'TFS': 30.0, 'PCS': 50.0, 'LSS': 90.0, 'MDS': 88.0, 'FRS': 100.0}
    down, up = rank_drivers(s)
    assert [k for k, _ in down] == ['TFS', 'PCS']                    # lost 17.5 vs 12.5
    assert [k for k, _ in up] == ['LSS', 'FRS']                      # contributions 18.0 vs 15.0 (MDS 13.2 cut)
    assert up[0][1] >= up[1][1]


def test_driver_fallbacks():
    down, up = rank_drivers({k: 100.0 for k in C.WEIGHTS})
    assert down == [] and len(up) == 2
    down, up = rank_drivers({k: 10.0 for k in C.WEIGHTS})
    assert up == [] and len(down) == 2 and down[0][1] >= down[1][1]
    down, _ = rank_drivers({'TFS': 81, 'PCS': 81, 'LSS': 81, 'MDS': 81, 'FRS': 81})     # 4.75 / 3.8 points lost: none >= 5
    assert down == []


BASELINE = {   # UCRII/components of 5 cohort customers recorded BEFORE the UI work (v1.1 anchors)
    'AABPK1282C': (28.434477, 100.0, 77.826614, 87.654171, 100.0, 75.822068),
    'AHDPI1728R': (32.543011, 13.506335, 75.885399, 90.976151, 68.333333, 50.585839),
    'BTZPI1430X': (83.728494, 100.0, 66.887617, 85.545723, 100.0, 87.141505),
    'CASPV8562B': (3.098519, 0.0, 30.027815, 89.97578, 41.304348, 26.472212),
    'CGAPZ5746H': (57.878969, 100.0, 66.853935, 90.834521, 100.0, 81.465708)}


@pytest.mark.parametrize('pan', BASELINE)
def test_scores_unchanged_from_before_ui(pan):
    p = ROOT.parent / 'Synthetic_UPI_Txn_Generator' / 'UPI_Txns' / f'{pan}.csv'
    if not p.exists():
        pytest.skip('local cohort file not present (UPI_Txns is gitignored)')
    raw, flags = compute_raw_features(load_transactions(p))
    s = score_customer(raw, flags, ANCH)
    got = tuple(s[k] for k in ('TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII'))
    assert got == pytest.approx(BASELINE[pan], abs=1e-5)
