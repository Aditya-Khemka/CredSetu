"""Driver: generate ~12 months of UPI history for many synthetic users, one CSV per user named by (made-up) PAN.

Run:  python driver.py [n_users] [--seed S] [--outdir DIR] [--age GROUP] [--scenario NAME] [--override key=value ...]
      ->  UPI_Txns/<PAN>.csv  (+ <PAN>.xlsx with Transactions + Profile sheets, manifest.csv, README.txt and _latent/ ground truth for validation only)
"""
import argparse
import json
import os
import random
import string

import numpy as np
import pandas as pd

from upi_user_generator import FIRST_NAMES, GENERATOR_VERSION, LAST_NAMES, generate_user_transactions

N_USERS = 100
OUTPUT_DIR = "UPI_Txns"
SCENARIO_ROOT = "UPI_Txns_scenarios"
START_DATE, END_DATE = "2024-01-01", "2024-12-31"
MASTER_SEED = 2024

# Same age mix as the original generator's sender_age distribution (original Kaggle UPI generator)
AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '56+']
AGE_P = [0.25, 0.35, 0.25, 0.10, 0.05]

# Scenario presets = generator overrides (same master seed => same PANs, ages, n_records and per-user seeds).
# 'sparse' is cohort-level (fewer records per user), handled in run().
SCENARIOS = {
    'baseline': {},
    'high_spend_ratio': {'spend_ratio': 1.15},
    'low_spend_ratio': {'spend_ratio': 0.60},
    'income_disruption': {'shock': 'income_disruption', 'resilience': 0.1},
    'expense_spike': {'shock': 'expense_spike'},
    'irregular_recurring': {'recurring_jitter_sd': 6},
    'activity_decay': {'activity_decay': 0.6},
    'technical_flaky': {'tech_failure_rate': 0.08},
    'sparse': {},
}

MANIFEST_COLUMNS = ['pan', 'age_group', 'seed', 'n_records_target', 'rows', 'debit_rows', 'credit_rows',
                    'generator_version', 'scenario', 'overrides_json']

LATENT_README = ("Ground-truth generator parameters for VALIDATION ONLY. "
                 "Feature engineering and scoring must never read this folder.\n")
COHORT_README = ("Synthetic UPI transaction histories (one CSV per customer, file name = made-up PAN).\n"
                 "PANs are synthetic and unvalidated: they follow the PAN format but may coincide with real PANs.\n"
                 "All behaviour (income, spending, balances, failures, credits) is a synthetic-data assumption, "
                 "not an empirical statistic.\n"
                 "manifest.csv lists generation settings only; _latent/ holds validation-only ground truth.\n")


def make_pan(rng, used):
    """Made-up PAN in the real format AAAPL1234C: 3 letters, 'P' (individual), surname initial, 4 digits, check letter."""
    while True:
        L = string.ascii_uppercase
        pan = (''.join(rng.choices(L, k=3)) + 'P' + rng.choice(L) +
               ''.join(rng.choices(string.digits, k=4)) + rng.choice(L))
        if pan not in used:
            used.add(pan)
            return pan


def parse_value(text):
    """int / float / bool / str, in that order."""
    low = text.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        if '=' not in p:
            raise SystemExit(f"--override expects key=value, got {p!r}")
        k, v = p.split('=', 1)
        out[k.strip()] = parse_value(v)
    return out


def _jsonable(x):
    """numpy / pandas types -> plain JSON types."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


LATENT_SCALAR_KEYS = ['age_group', 'income_type', 'rho', 'affluence', 'h', 'opening_balance_multiple',
                      'opening_balance', 'monthly_income', 'monthly_outflow', 'upi_share', 'pay_day',
                      'tech_failure_rate', 'resilience', 'has_rent', 'rent_amount', 'rent_day', 'activity_decay',
                      'recurring_jitter_sd', 'n_records_target', 'planned_debit_attempts', 'generator_version']


def latent_record(profile):
    """Latent profile for _latent/<PAN>.json: scalars, shock spec and a summary (not the full list) of hidden events."""
    rec = {k: profile[k] for k in LATENT_SCALAR_KEYS}
    rec['shock'] = profile['shock']
    rec['recurring_services'] = profile['recurring_services']
    hid = profile['hidden_events']
    rec['hidden_events_summary'] = {
        'n_credits': sum(1 for h in hid if h['amount'] > 0), 'total_credits': round(sum(h['amount'] for h in hid if h['amount'] > 0), 2),
        'n_debits': sum(1 for h in hid if h['amount'] < 0), 'total_debits': round(-sum(h['amount'] for h in hid if h['amount'] < 0), 2)}
    return _jsonable(rec)


def profile_sheet(pan, age_group, profile):
    """Fixed, observable customer attributes for the 'Profile' sheet (no latent financial parameters).
    Name and exact age are not modelled by the generator, so they are drawn from a separate per-PAN stream
    (the master rng and the generator's rng are untouched)."""
    r = random.Random(f"profile-{pan}")
    lo, hi = (56, 75) if age_group == '56+' else map(int, age_group.split('-'))
    fields = [('PAN', pan), ('Name', f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"), ('Age', r.randint(lo, hi)),
              ('Age_Group', age_group), ('State', profile['sender_state']), ('Bank', profile['customer_bank']),
              ('Device_Type', profile['device_type'])]
    return pd.DataFrame(fields, columns=['Field', 'Value'])


def summarise(records):
    """Per age group and overall summary; every failed share is over DEBIT rows only (failed debits / debit rows)."""
    def agg(rs):
        deb = sum(r['debit_rows'] for r in rs)
        return {'users': len(rs), 'mean_rows': float(np.mean([r['rows'] for r in rs])),
                'mean_debit_rows': float(np.mean([r['debit_rows'] for r in rs])),
                'mean_credit_rows': float(np.mean([r['credit_rows'] for r in rs])),
                'failed_share': sum(r['failed_debits'] for r in rs) / deb,
                'isf_share': sum(r['isf_debits'] for r in rs) / deb,
                'tech_share': sum(r['tech_debits'] for r in rs) / deb}
    out = {'by_age': {a: agg([r for r in records if r['age_group'] == a]) for a in AGE_GROUPS
                      if any(r['age_group'] == a for r in records)}}
    rows = [r['rows'] for r in records]
    out.update({'rows_min': int(min(rows)), 'rows_median': float(np.median(rows)), 'rows_max': int(max(rows)),
                'total_rows': int(sum(rows)), 'overall': agg(records),
                'mean_user_isf_share': float(np.mean([r['isf_debits'] / r['debit_rows'] for r in records])),
                'mean_min_balance': float(np.mean([r['min_balance'] for r in records])),
                'income_types': {k: int(v) for k, v in pd.Series([r['income_type'] for r in records]).value_counts().items()},
                'low_balance_user_share': float(np.mean([r['min_balance'] < 0.05 * r['monthly_income'] for r in records]))})
    return out


def run(n_users=N_USERS, seed=MASTER_SEED, outdir=OUTPUT_DIR, age=None, scenario=None, overrides=None,
        verbose=True):
    """Generate a cohort into outdir. Returns (manifest DataFrame, summary dict)."""
    preset = dict(SCENARIOS[scenario]) if scenario else {}
    user_overrides = {**preset, **(overrides or {})}          # explicit --override keys win over the preset
    rng = random.Random(seed)                                 # PANs, ages, activity levels, per-user seeds
    used = set()
    os.makedirs(os.path.join(outdir, '_latent'), exist_ok=True)

    manifest, records = [], []
    for i in range(n_users):
        # Draw ORDER from the master rng must never change: pan, age_group, n_records, per-user seed
        pan = make_pan(rng, used)
        sampled_age = rng.choices(AGE_GROUPS, weights=AGE_P)[0]
        age_group = age or sampled_age
        # Users differ in how active they are: skewed around ~550 debit attempts/year, 200-1500
        n_records = int(np.clip(rng.lognormvariate(np.log(550), 0.35), 200, 1500))
        if scenario == 'sparse':      # separate stream so the master rng stays aligned with the other presets
            n_records = random.Random(f"sparse-{seed}-{i}").randint(60, 150)
        user_seed = rng.randrange(2**31)

        df, profile = generate_user_transactions(age_group, n_records, START_DATE, END_DATE, seed=user_seed,
                                                 overrides=user_overrides, return_profile=True)
        df.to_csv(os.path.join(outdir, f"{pan}.csv"), index=False)
        with pd.ExcelWriter(os.path.join(outdir, f"{pan}.xlsx")) as xw:      # sheet 1: history, sheet 2: fixed profile
            df.to_excel(xw, sheet_name='Transactions', index=False)
            profile_sheet(pan, age_group, profile).to_excel(xw, sheet_name='Profile', index=False)
        with open(os.path.join(outdir, '_latent', f"{pan}.json"), 'w') as f:
            json.dump(latent_record(profile), f, indent=1, sort_keys=True)

        deb = df[df['Direction'] == 'DEBIT']
        failed = deb[deb['Payment_Status'] == 'FAILED']
        manifest.append([pan, age_group, user_seed, n_records, len(df), len(deb), len(df) - len(deb),
                         GENERATOR_VERSION, scenario or 'none', json.dumps(user_overrides, sort_keys=True)])
        records.append({'age_group': age_group, 'rows': len(df), 'debit_rows': len(deb), 'credit_rows': len(df) - len(deb),
                        'failed_debits': len(failed),
                        'isf_debits': int((failed['Failure_Reason'] == 'INSUFFICIENT_FUNDS').sum()),
                        'tech_debits': int((failed['Failure_Reason'] == 'TECHNICAL').sum()),
                        'income_type': profile['income_type'], 'monthly_income': profile['monthly_income'],
                        'min_balance': float(df['Balance_After'].min())})

    with open(os.path.join(outdir, '_latent', 'README.txt'), 'w') as f:
        f.write(LATENT_README)
    with open(os.path.join(outdir, 'README.txt'), 'w') as f:
        f.write(COHORT_README)
    mdf = pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)
    mdf.to_csv(os.path.join(outdir, 'manifest.csv'), index=False)
    summary = summarise(records)
    if verbose:
        print_summary(summary, outdir, scenario)
    return mdf, summary


def print_summary(s, outdir, scenario=None):
    print(f"Saved {s['overall']['users']} users to {outdir}/" + (f" (scenario: {scenario})" if scenario else ''))
    print("Rows per user exceed n_records because credits, retry attempts and shock debits are added on top of the "
          "planned debit attempts. Failed shares are over DEBIT rows only.")
    for age, a in s['by_age'].items():
        print(f"  {age:6s} {a['users']:3d} users, mean rows {a['mean_rows']:.0f}, debit attempts {a['mean_debit_rows']:.0f}, "
              f"credit rows {a['mean_credit_rows']:.0f}, failed {a['failed_share']:.1%} "
              f"(INSUFFICIENT_FUNDS {a['isf_share']:.1%}, TECHNICAL {a['tech_share']:.1%})")
    print(f"Total rows: {s['total_rows']:,}; rows per user min/median/max: "
          f"{s['rows_min']}/{s['rows_median']:.0f}/{s['rows_max']}")
    print("income_type counts (latent):", s['income_types'])
    print(f"Users whose minimum Balance_After < 5% of monthly income: {s['low_balance_user_share']:.1%}")
    print("Sample files:", sorted(f for f in os.listdir(outdir) if f.endswith('.csv') and f != 'manifest.csv')[:3])


def build_parser():
    p = argparse.ArgumentParser(description="Generate a cohort of synthetic UPI histories (one CSV per PAN).")
    p.add_argument('n_users', nargs='?', type=int, default=N_USERS, help=f"number of users (default {N_USERS})")
    p.add_argument('--seed', type=int, default=MASTER_SEED, help="master seed (default %(default)s)")
    p.add_argument('--outdir', default=None, help=f"output dir (default {OUTPUT_DIR}, or {SCENARIO_ROOT}/<name> with --scenario)")
    p.add_argument('--age', choices=AGE_GROUPS, default=None, help="force one age group (default: sample from AGE_P)")
    p.add_argument('--scenario', choices=sorted(SCENARIOS), default=None)
    p.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                   help="generator override, repeatable; wins over the scenario preset")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = args.outdir or (os.path.join(SCENARIO_ROOT, args.scenario) if args.scenario else OUTPUT_DIR)
    return run(args.n_users, args.seed, outdir, args.age, args.scenario, parse_overrides(args.override))


if __name__ == "__main__":
    main()
