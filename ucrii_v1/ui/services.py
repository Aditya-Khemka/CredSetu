"""UI logic without Streamlit: PAN validation, deterministic generation, scoring, analytics, report.
The scoring path reads ONLY the transaction CSV (never _latent/, the xlsx Profile sheet, or manifest.csv)."""
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd

from ui import settings

if str(settings.ROOT) not in sys.path:
    sys.path.insert(0, str(settings.ROOT))
from ucrii import __version__ as UCRII_VERSION, config as C                       # noqa: E402
from ucrii.features import compute_raw_features, load_transactions                 # noqa: E402
from ucrii.scoring import load_anchors, score_customer                             # noqa: E402

PAN_RE = re.compile(r'^[A-Z]{3}P[A-Z][0-9]{4}[A-Z]$')
AGE_CHOICES = ['Auto (from PAN)', '18-25', '26-35', '36-45', '46-55', '56+']
ACTIVITY_CHOICES = ['Typical (auto)', 'Thin-file (60-150 debit attempts/year)', 'Moderately active (200-500)',
                    'Very active (600-1000)', 'Extremely frequent (1000-1500)']
SITUATIONS = {                                   # friendly label -> driver.SCENARIOS key ('sparse' deliberately absent)
    'Natural variation (default)': 'baseline',
    'Income disruption': 'income_disruption',
    'One-off expense spike': 'expense_spike',
    'Spending above income': 'high_spend_ratio',
    'Spending well below income': 'low_spend_ratio',
    'Unreliable network': 'technical_flaky',
    'Irregular recurring payments': 'irregular_recurring',
    'Declining activity': 'activity_decay',
}
SEED_TAG = 'ucrii-ui-v1'
DEFAULT_SITUATION = list(SITUATIONS)[0]


class ServiceError(Exception):
    """User-facing problem (bad PAN, old schema). `code` lets the UI offer the right next step."""
    def __init__(self, msg, code='error'):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- generator import
def _gen():
    g = str(settings.generator_dir())
    if g not in sys.path:
        sys.path.insert(0, g)
    import driver, upi_user_generator            # type: ignore  # top-level modules on a runtime sys.path entry
    return driver, upi_user_generator


# ---------------------------------------------------------------- PAN / seeds
def normalise_pan(text):
    return (text or '').strip().upper()


def validate_pan(text):
    """Return the normalised PAN or raise ServiceError. The regex alone guarantees a safe file name."""
    pan = normalise_pan(text)
    if not PAN_RE.match(pan):
        raise ServiceError("Invalid PAN format. Expected 10 characters: 3 letters, 'P' (individual), 1 letter, "
                           "4 digits, 1 letter (e.g. ABCPD1234E).", 'invalid_pan')
    return pan


def derive(pan):
    digest = hashlib.sha256(f"{pan}|{SEED_TAG}".encode()).digest()
    return int.from_bytes(digest[:4], 'big') % 2**31, random.Random(int.from_bytes(digest[4:12], 'big'))


def resolve_settings(pan, age_group=AGE_CHOICES[0], activity=ACTIVITY_CHOICES[0], situation=DEFAULT_SITUATION,
                     income_type=None):
    """Deterministic generation settings for a PAN. Aux draws are always consumed in the same order."""
    driver, _ = _gen()
    seed, aux = derive(pan)
    auto_age = aux.choices(driver.AGE_GROUPS, weights=driver.AGE_P)[0]
    typical = int(np.clip(aux.lognormvariate(np.log(550), 0.35), 200, 1500))
    extreme = aux.randint(1000, 1500)                # draw order kept so earlier PANs are unchanged; new draws go after
    moderate, very_active = aux.randint(200, 500), aux.randint(600, 1000)
    thin = random.Random(f"sparse-{seed}").randint(60, 150)
    age = auto_age if age_group == AGE_CHOICES[0] else age_group
    n = dict(zip(ACTIVITY_CHOICES, [typical, thin, moderate, very_active, extreme]))[activity]
    preset = SITUATIONS[situation]
    overrides = dict(driver.SCENARIOS[preset])
    if income_type:
        overrides['income_type'] = income_type
    return {'age_group': age, 'n_records': n, 'seed': seed, 'preset': preset, 'overrides': overrides}


# ---------------------------------------------------------------- files
def _csv_path(pan, data=None):
    return (data or settings.data_dir()) / f"{pan}.csv"


def existing_pans(data=None):
    d = data or settings.data_dir()
    return sorted(p.stem for p in d.glob('*.csv') if PAN_RE.match(p.stem)) if d.is_dir() else []


def random_new_pan(data=None):
    driver, _ = _gen()
    used = set(existing_pans(data))
    return driver.make_pan(random.SystemRandom(), used)


def _atomic(path, write):
    """write(tmp_path) then os.replace, so readers never see a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_text(path, text):
    def w(t):
        with open(t, 'w') as f:
            f.write(text)
    _atomic(path, w)


def _upsert_manifest(data, row, columns):
    path = data / 'manifest.csv'
    old = pd.read_csv(path, dtype={'pan': str}) if path.exists() else pd.DataFrame(columns=columns)
    old = old[old['pan'] != row[0]]
    new = pd.concat([old, pd.DataFrame([row], columns=columns)], ignore_index=True)
    _atomic(path, lambda t: new.to_csv(t, index=False))


def generate(pan, data=None, **choices):
    """Generate ONE customer from the PAN and persist it in the driver's cohort layout. Returns the settings used."""
    driver, ugen = _gen()
    data = data or settings.data_dir()
    s = resolve_settings(pan, **choices)
    df, profile = ugen.generate_user_transactions(s['age_group'], s['n_records'], driver.START_DATE, driver.END_DATE,
                                                  seed=s['seed'], overrides=s['overrides'], return_profile=True)
    (data / '_latent').mkdir(parents=True, exist_ok=True)
    for p, txt in ((data / '_latent' / 'README.txt', driver.LATENT_README), (data / 'README.txt', driver.COHORT_README)):
        if not p.exists():
            _write_text(p, txt)
    _atomic(data / f"{pan}.csv", lambda t: df.to_csv(t, index=False))

    def write_xlsx(t):
        with pd.ExcelWriter(t, engine='openpyxl') as xw:
            df.to_excel(xw, sheet_name='Transactions', index=False)
            driver.profile_sheet(pan, s['age_group'], profile).to_excel(xw, sheet_name='Profile', index=False)
    _atomic(data / f"{pan}.xlsx", write_xlsx)
    _write_text(data / '_latent' / f"{pan}.json", json.dumps(driver.latent_record(profile), indent=1, sort_keys=True))
    deb = df[df['Direction'] == 'DEBIT']
    scenario = 'ui' if s['preset'] == 'baseline' else s['preset']
    _upsert_manifest(data, [pan, s['age_group'], s['seed'], s['n_records'], len(df), len(deb), len(df) - len(deb),
                            ugen.GENERATOR_VERSION, scenario, json.dumps(s['overrides'], sort_keys=True)],
                     driver.MANIFEST_COLUMNS)
    return s


def source_info(pan, data=None):
    """Provenance from the manifest row (metadata about the file, never used for scoring)."""
    data = data or settings.data_dir()
    try:
        m = pd.read_csv(data / 'manifest.csv', dtype={'pan': str})
        r = m[m['pan'] == pan]
        if len(r):
            r = r.iloc[0]
            kind = 'UI-generated' if int(r['seed']) == derive(pan)[0] else 'driver cohort'
            return {'kind': kind, 'generator_version': str(r['generator_version']), 'scenario': str(r['scenario'])}
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        pass
    return {'kind': 'unknown', 'generator_version': None, 'scenario': None}


# ---------------------------------------------------------------- scoring
def _clean(x):
    """numpy/NaN -> plain JSON types (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_df(path):
    """load_transactions with a friendly schema error."""
    head = pd.read_csv(path, nrows=0)
    missing = [c for c in C.REQUIRED_COLUMNS if c not in head.columns]
    if missing:
        raise ServiceError(f"This history file uses an older schema and lacks required columns: {missing}. "
                           "Tick 'Regenerate' to create a fresh history.", 'schema')
    return load_transactions(path)


@lru_cache(maxsize=32)
def _score_cached(path, mtime_ns, anchors_file, anchors_mtime_ns):
    df = load_df(path)
    anchors = load_anchors(anchors_file)
    raw, flags = compute_raw_features(df, C.WINDOW_START, C.WINDOW_END)
    scores = score_customer(raw, flags, anchors)
    expl = None
    if scores['status'] == 'ok':
        try:
            from ucrii.explain import explain
            expl = explain(df, C.WINDOW_START, C.WINDOW_END, raw, flags, scores, anchors)
        except ImportError:              # explain.py arrives in stage B
            pass
    return _clean({'raw': raw, 'flags': flags, 'scores': scores, 'explanation': expl})


def score_file(path, anchors_file=None):
    """Score one transaction CSV. Returns a JSON-serialisable dict."""
    a = anchors_file or settings.anchors_path()
    r = copy.deepcopy(_score_cached(str(path), os.stat(path).st_mtime_ns, str(a), os.stat(a).st_mtime_ns))
    s = r['scores']
    r['status'], r['confidence'] = s['status'], s['confidence']
    r['contributions'] = {k: s.get(f'contrib_{k}') for k in C.WEIGHTS}
    return r


def score_pan(pan, age_group=AGE_CHOICES[0], activity=ACTIVITY_CHOICES[0], situation=DEFAULT_SITUATION,
              income_type=None, regenerate=False, data=None, anchors_file=None):
    pan = validate_pan(pan)
    data = data or settings.data_dir()
    path = _csv_path(pan, data)
    generated = regenerate or not path.exists()
    used = None
    if generated:
        used = generate(pan, data, age_group=age_group, activity=activity, situation=situation, income_type=income_type)
    r = score_file(path, anchors_file)
    _, ugen = _gen()
    a = anchors_file or settings.anchors_path()
    return {'pan': pan, 'generated': generated, 'source': source_info(pan, data), 'settings': used,
            'versions': {'ucrii': UCRII_VERSION, 'generator': ugen.GENERATOR_VERSION, 'anchors': os.path.basename(a)},
            **r}


# ---------------------------------------------------------------- tables for the charts and docs tabs
def income_types():
    return list(_gen()[1].INCOME_TYPES)


def activity(path):
    """Chart tables for the Activity tab, from the transaction CSV only. Uses the same window slicing and inflow
    definition as compute_raw_features (credits + hidden-inflow jumps)."""
    from ucrii.explain import window_slices
    from ucrii.features import unexplained_flows
    df = load_df(path)
    ws, we = pd.Timestamp(C.WINDOW_START), pd.Timestamp(C.WINDOW_END)
    months = pd.period_range(ws, we, freq='M')
    w, att, ok = window_slices(df, C.WINDOW_START, C.WINDOW_END)
    daily = df.groupby(df['Timestamp'].dt.normalize())['Balance_After'].last()
    daily = daily.reindex(pd.date_range(daily.index.min(), we.normalize())).ffill().bfill()
    ok_m = ok.groupby(ok['Timestamp'].dt.to_period('M')).size().reindex(months, fill_value=0)
    failed = att[att['Payment_Status'] == 'FAILED']
    fail_m = (failed.groupby([failed['Timestamp'].dt.to_period('M'), 'Failure_Reason']).size().unstack(fill_value=0)
              .reindex(months, fill_value=0).reindex(columns=['INSUFFICIENT_FUNDS', 'TECHNICAL'], fill_value=0))
    cr = w[w['Direction'] == 'CREDIT']
    inflow = cr.groupby(cr['Timestamp'].dt.to_period('M'))['Amount'].sum().reindex(months, fill_value=0.0)
    un = pd.Series(unexplained_flows(df), index=df.index).loc[w.index]
    jumps = un[un > C.UNEXPLAINED_JUMP_MIN]
    inflow = inflow + jumps.groupby(w.loc[jumps.index, 'Timestamp'].dt.to_period('M')).sum().reindex(months, fill_value=0.0)
    lab = lambda s: s.set_axis(months.strftime('%Y-%m'))          # noqa: E731
    return {'daily_balance': daily.rename('Balance').rename_axis('Date').reset_index(),
            'monthly_successful': lab(ok_m).rename('Successful debits').rename_axis('Month').reset_index(),
            'monthly_failed': lab(fail_m.set_axis(months.strftime('%Y-%m'))).rename_axis('Month').reset_index(),
            'monthly_inflow': lab(inflow).rename('Estimated inflow').rename_axis('Month').reset_index()}


def limitations_markdown():
    """The 'Known limitations' section of docs/SPEC_features.md, verbatim."""
    text = (settings.ROOT / 'docs' / 'SPEC_features.md').read_text(encoding='utf-8')
    m = re.search(r'^## 8\. Known limitations\n(.*?)^---', text, re.S | re.M)
    return m.group(1).strip() if m else 'See docs/SPEC_features.md, section 8.'


# ---------------------------------------------------------------- Stage D: where this customer sits in the folder's cohort
def cohort_signature(data=None):
    """(folder, file count, newest mtime): the cache key that invalidates when the cohort changes."""
    from ucrii.cli import customer_files
    files = customer_files(data or settings.data_dir())
    return (str(data or settings.data_dir()), len(files), max((p.stat().st_mtime_ns for p in files), default=0))


@lru_cache(maxsize=4)
def _cohort_scores(signature, anchors_file):
    from ucrii.cli import raw_table
    from ucrii.scoring import score_table
    raw, flags = raw_table(signature[0])           # top-level CSVs only; never _latent/, xlsx or manifest
    t = score_table(raw, flags, load_anchors(anchors_file))
    return t[t['status'] == 'ok'].set_index('customer_id')[list(C.WEIGHTS) + ['UCRII']]


def cohort_percentiles(pan, data=None, anchors_file=None):
    """Percentile rank (share of scored cohort customers at or below this one) for UCRII and each component.
    The cohort is every scored customer in the data folder, including this one."""
    t = _cohort_scores(cohort_signature(data), str(anchors_file or settings.anchors_path()))
    if pan not in t.index:
        return None
    return {'n': int(len(t)), 'percentiles': {k: float(100 * (t[k] <= t.loc[pan, k]).mean()) for k in t.columns}}


def analytics(path):
    """Descriptive analytics for the Spending tab, from the transaction rows only (display-only; not used in scoring).
    Window = the fixed analysis window; 'spend' = successful debits, 'received' = credits."""
    from ucrii.explain import window_slices
    df = load_df(path)
    w, att, ok = window_slices(df, C.WINDOW_START, C.WINDOW_END)
    cr = w[(w['Direction'] == 'CREDIT') & (w['Payment_Status'] == 'SUCCESS')]
    days = int((pd.Timestamp(C.WINDOW_END) - pd.Timestamp(C.WINDOW_START)).days + 1)
    active = ok['Timestamp'].dt.normalize().nunique()
    spend = float(ok['Amount'].sum())
    agg = lambda g: (ok.groupby(g)['Amount'].agg(Payments='size', Amount='sum').sort_values('Amount', ascending=False)   # noqa: E731
                     .assign(Share=lambda t: t['Amount'] / spend if spend else 0.0).reset_index())
    cp = agg('Counterparty_Name')
    months = pd.period_range(C.WINDOW_START, C.WINDOW_END, freq='M')
    m = lambda d: d.groupby(d['Timestamp'].dt.to_period('M'))['Amount'].sum().reindex(months, fill_value=0.0)   # noqa: E731
    flow = pd.DataFrame({'Month': months.strftime('%Y-%m'), 'Spent': m(ok).to_numpy(), 'Received': m(cr).to_numpy()})
    flow['Net'] = flow['Received'] - flow['Spent']
    edges = [0, 100, 250, 500, 1000, 2500, 5000, 10000, float('inf')]
    labels = ['<100', '100-249', '250-499', '500-999', '1k-2.5k', '2.5k-5k', '5k-10k', '10k+']
    hist = pd.cut(ok['Amount'], edges, right=False, labels=labels).value_counts().reindex(labels).rename('Payments')
    wd = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day = ok.groupby(ok['Timestamp'].dt.day_name())['Amount'].agg(Payments='size', Amount='sum').reindex(wd, fill_value=0)
    by_hour = ok.groupby(ok['Timestamp'].dt.hour).size().reindex(range(24), fill_value=0).rename('Payments')
    daily_bal = df.groupby(df['Timestamp'].dt.normalize())['Balance_After'].last().reindex(
        pd.date_range(C.WINDOW_START, C.WINDOW_END)).ffill().bfill()
    return {
        'kpis': {'successful_debits': int(len(ok)), 'total_spent': spend, 'median_payment': float(ok['Amount'].median()),
                 'largest_payment': float(ok['Amount'].max()), 'credits_received': int(len(cr)),
                 'total_received': float(cr['Amount'].sum()), 'active_days': int(active), 'n_days': days,
                 'payments_per_active_day': float(len(ok) / active) if active else 0.0,
                 'weekend_share': float((ok['Timestamp'].dt.dayofweek >= 5).mean()),
                 'top5_counterparty_share': float(cp['Share'].head(5).sum()), 'n_counterparties': int(len(cp)),
                 'median_balance': float(daily_bal.median()), 'min_balance': float(daily_bal.min())},
        'by_category': agg('Merchant_Category'), 'by_type': agg('Transaction_Type'), 'top_counterparties': cp.head(10),
        'monthly_flow': flow, 'amount_hist': hist.rename_axis('Amount (INR)').reset_index(),
        'by_weekday': by_day.rename_axis('Day').reset_index(), 'by_hour': by_hour.rename_axis('Hour').reset_index()}


@lru_cache(maxsize=32)
def _signals_cached(path, mtime_ns):
    from ucrii.insights import insights
    return _clean(insights(load_df(path), C.WINDOW_START, C.WINDOW_END))


def credit_signals(path):
    """[EXTENSION] descriptive credit signals (not part of the score). JSON-serialisable."""
    return copy.deepcopy(_signals_cached(str(path), os.stat(path).st_mtime_ns))


def instalment_what_if(path, instalment, due_day=5):
    """Backtest of a hypothetical monthly instalment against this customer's own balance history."""
    from ucrii.insights import what_if
    return what_if(load_df(path), C.WINDOW_START, C.WINDOW_END, instalment, due_day)
