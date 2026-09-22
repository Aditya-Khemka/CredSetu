import builtins
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import services as S                 # noqa: E402
from ucrii import config as C                # noqa: E402
from ucrii.cli import raw_table              # noqa: E402
from ucrii.scoring import load_anchors, score_table   # noqa: E402

PAN = 'ABCPD1234E'
ANCHORS = ROOT / 'ucrii' / 'anchors_reference_v1_1.json'


@pytest.mark.parametrize('text,ok', [
    ('ABCPD1234E', True), ('  abcpd1234e ', True), ('ABCPD1234', False), ('ABCPD12345E', False),
    ('ABCXD1234E', False), ('ABCPD1234e!', False), ('../x', False), ('..\\ABCPD1234E', False),
    ('ABCPD1234E/../../x', False), ('', False), ('ABCPD1234E\n../x', False)])
def test_pan_validation(text, ok, tmp_path):
    if ok:
        assert S.validate_pan(text) == text.strip().upper()
    else:
        with pytest.raises(S.ServiceError) as e:
            S.score_pan(text, data=tmp_path)
        assert e.value.code == 'invalid_pan'
        assert list(tmp_path.iterdir()) == []


def test_seed_stable_across_processes():
    code = ("import sys; sys.path.insert(0, %r); from ui import services as S; "
            "print(S.derive(%r)[0], S.derive(%r)[1].random())" % (str(ROOT), PAN, PAN))
    a = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout
    b = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout
    assert a == b and a.strip()
    assert f"{S.derive(PAN)[0]} {S.derive(PAN)[1].random()}" == a.strip()


def test_generation_layout_and_determinism(tmp_path):
    d1, d2 = tmp_path / 'a', tmp_path / 'b'
    d1.mkdir(), d2.mkdir()
    r = S.score_pan(PAN, data=d1, anchors_file=ANCHORS)
    S.score_pan(PAN, data=d2, anchors_file=ANCHORS)
    assert r['generated'] and r['status'] == 'ok'
    assert (d1 / f'{PAN}.csv').read_bytes() == (d2 / f'{PAN}.csv').read_bytes()
    assert (d1 / f'{PAN}.xlsx').exists() and (d1 / '_latent' / f'{PAN}.json').exists()
    assert list(pd.read_csv(d1 / f'{PAN}.csv', nrows=0).columns)[:1] and \
        not set(C.REQUIRED_COLUMNS) - set(pd.read_csv(d1 / f'{PAN}.csv', nrows=0).columns)
    xl = pd.ExcelFile(d1 / f'{PAN}.xlsx')
    assert xl.sheet_names == ['Transactions', 'Profile']
    m = pd.read_csv(d1 / 'manifest.csv')
    assert list(m.columns) == S._gen()[0].MANIFEST_COLUMNS and list(m['pan']) == [PAN]
    assert r['source']['kind'] == 'UI-generated'
    assert not list(d1.glob('.tmp-*')) and not list(d1.glob('_latent/.tmp-*'))
    json.dumps(r)                                          # JSON-serialisable


def test_existing_reused_and_regenerate(tmp_path):
    S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    p = tmp_path / f'{PAN}.csv'
    t0 = p.stat().st_mtime_ns
    r = S.score_pan(PAN, age_group='56+', situation='Income disruption', data=tmp_path, anchors_file=ANCHORS)
    assert not r['generated'] and p.stat().st_mtime_ns == t0      # settings ignored
    before = r['scores']['UCRII']
    r2 = S.score_pan(PAN, situation='Income disruption', regenerate=True, data=tmp_path, anchors_file=ANCHORS)
    assert r2['generated'] and r2['scores']['UCRII'] != before
    m = pd.read_csv(tmp_path / 'manifest.csv')
    assert len(m) == 1 and m.loc[0, 'scenario'] == 'income_disruption'


def test_old_schema_friendly_error(tmp_path):
    pd.DataFrame({'Timestamp': ['2024-03-01'], 'Amount': [1]}).to_csv(tmp_path / f'{PAN}.csv', index=False)
    with pytest.raises(S.ServiceError) as e:
        S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    assert e.value.code == 'schema' and 'Direction' in str(e.value)


def test_thin_file_is_low_confidence(tmp_path):
    r = S.score_pan(PAN, activity=S.ACTIVITY_CHOICES[1], data=tmp_path, anchors_file=ANCHORS)
    assert r['raw']['n_successful_debits'] < 150 and r['confidence'] == 'low' and 'low_history' in r['flags']


def test_matches_cli(tmp_path):
    S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    raw, flags = raw_table(tmp_path)
    ref = score_table(raw, flags, load_anchors(ANCHORS)).iloc[0]
    got = S.score_file(tmp_path / f'{PAN}.csv', ANCHORS)['scores']
    for k in ['TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII', 'UCRII_excl_TFS']:
        assert abs(got[k] - ref[k]) < 1e-9


def test_scoring_never_touches_latent_or_profile(tmp_path, monkeypatch):
    S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    S._score_cached.cache_clear()
    seen = []
    real_open, real_csv, real_xl = builtins.open, pd.read_csv, pd.read_excel
    monkeypatch.setattr(builtins, 'open', lambda f, *a, **k: (seen.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(pd, 'read_csv', lambda f, *a, **k: (seen.append(str(f)), real_csv(f, *a, **k))[1])
    monkeypatch.setattr(pd, 'read_excel', lambda f, *a, **k: (seen.append(str(f)), real_xl(f, *a, **k))[1])
    S.score_file(tmp_path / f'{PAN}.csv', ANCHORS)
    assert seen and not [s for s in seen if '_latent' in s or s.endswith('.xlsx') or 'manifest' in s]
    monkeypatch.undo()
    S._score_cached.cache_clear()
    shutil.rmtree(tmp_path / '_latent')
    (tmp_path / f'{PAN}.xlsx').unlink()
    (tmp_path / 'manifest.csv').unlink()
    r = S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    assert r['status'] == 'ok' and not r['generated'] and r['source']['kind'] == 'unknown'


def test_analytics_consistent_with_transactions(tmp_path):
    S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)
    a = S.analytics(tmp_path / f'{PAN}.csv')
    k, r = a['kpis'], S.score_file(tmp_path / f'{PAN}.csv', ANCHORS)['raw']
    assert k['successful_debits'] == r['n_successful_debits']
    for t in ('by_category', 'by_type'):
        assert abs(a[t]['Amount'].sum() - k['total_spent']) < 1e-6 and abs(a[t]['Share'].sum() - 1) < 1e-9
    assert abs(a['monthly_flow']['Spent'].sum() - k['total_spent']) < 1e-6
    assert a['amount_hist']['Payments'].sum() == k['successful_debits'] == a['by_hour']['Payments'].sum()
    assert a['by_weekday']['Payments'].sum() == k['successful_debits'] and 0 <= k['top5_counterparty_share'] <= 1
    assert 'trajectory' not in S.score_pan(PAN, data=tmp_path, anchors_file=ANCHORS)


def test_random_pan(tmp_path):
    p = S.random_new_pan(tmp_path)
    assert S.PAN_RE.match(p) and p not in S.existing_pans(tmp_path)


def test_cohort_percentiles_and_cache_invalidation(tmp_path):
    pans = ['AAAPA1111A', 'BBBPB2222B', 'CCCPC3333C']
    for i, p in enumerate(pans):
        S.score_pan(p, situation='Income disruption' if i == 0 else S.DEFAULT_SITUATION, data=tmp_path, anchors_file=ANCHORS)
    cp = {p: S.cohort_percentiles(p, tmp_path, ANCHORS) for p in pans}
    assert all(c['n'] == 3 for c in cp.values())
    u = {p: cp[p]['percentiles']['UCRII'] for p in pans}
    assert max(u.values()) == 100.0 and min(u.values()) >= 100 / 3 - 1e-9
    assert S.cohort_percentiles('ZZZPZ9999Z', tmp_path, ANCHORS) is None
    S.score_pan('DDDPD4444D', data=tmp_path, anchors_file=ANCHORS)               # new file -> new signature
    assert S.cohort_percentiles(pans[0], tmp_path, ANCHORS)['n'] == 4


def test_activity_levels_draw_within_range():
    lo_hi = [(60, 150), (200, 500), (600, 1000), (1000, 1500)]
    for a, (lo, hi) in zip(S.ACTIVITY_CHOICES[1:], lo_hi):
        n = S.resolve_settings(PAN, activity=a)['n_records']
        assert lo <= n <= hi, (a, n)
    assert S.resolve_settings(PAN, activity=S.ACTIVITY_CHOICES[0])['n_records'] == S.resolve_settings(PAN)['n_records']
