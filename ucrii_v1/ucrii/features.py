"""Raw behavioural indicators from ONE customer's UPI history (transaction rows only).

Reads nothing except the transaction DataFrame: no latent parameters, no profile sheet, no age/state/bank fields.
See docs/SPEC_features.md for the definition of every indicator and its provenance.
"""
import numpy as np
import pandas as pd

from . import config as C

RAW_COLUMNS = ['tfs_raw', 'pcs_raw', 'L1_balance_cv', 'L2_inflow_cv', 'mds_raw', 'F1_obligation_continuity',
               'F2_buffer_adequacy']
DIAGNOSTIC_COLUMNS = ['pcs_incl_technical', 'n_successful_debits', 'n_attempts', 'n_obligations',
                      'n_unique_merchants', 'effective_merchants', 'mean_monthly_outflow']


def load_transactions(path):
    """Read one customer CSV, validate the schema, return rows sorted by time (stable)."""
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df.sort_values('Timestamp', kind='mergesort').reset_index(drop=True)


def unexplained_flows(df):
    """Balance change between consecutive rows NOT explained by the row's own successful UPI amount
    (= net hidden flow). The first row is NaN because the opening balance is unknown."""
    sign = np.where(df['Direction'] == 'CREDIT', 1.0, -1.0)
    signed = np.where(df['Payment_Status'] == 'SUCCESS', sign * df['Amount'].to_numpy(dtype=float), 0.0)
    bal = df['Balance_After'].to_numpy(dtype=float)
    prev = np.concatenate([[np.nan], bal[:-1]])
    return bal - prev - signed


def _cv(x):
    """Coefficient of variation (sample sd / mean); NaN if the mean is not positive."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return float(x.std(ddof=1) / m) if m > 0 and len(x) > 1 else np.nan


def _entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def detect_obligations(attempts):
    """Recurring obligations inferred from ALL debit attempts in the window (failed ones included, so an obligation
    that keeps failing is still detected). Attempts within CYCLE_GAP_DAYS form one billing cycle; a cycle succeeds if
    any attempt in it succeeded. Returns [{'name', 'cycles', 'ok_cycles'}]."""
    cand = attempts[attempts['Merchant_Category'].isin(C.OBLIGATION_CATEGORIES) | (attempts['Transaction_Type'] == 'P2P')]
    if cand.empty:
        return []
    cand = cand.assign(_month=cand['Timestamp'].dt.to_period('M'), _ok=(cand['Payment_Status'] == 'SUCCESS'))
    # necessary condition (cycle start months are a subset of attempt months): skip counterparties seen in < 6 months
    months_seen = cand.groupby('Counterparty_Name')['_month'].nunique()
    cand = cand[cand['Counterparty_Name'].isin(months_seen[months_seen >= C.OBLIGATION_MIN_MONTHS].index)]
    out = []
    lo, hi = C.OBLIGATION_MEDIAN_GAP_DAYS
    for name, g in cand.groupby('Counterparty_Name', sort=True):
        g = g.sort_values('Timestamp', kind='mergesort')
        t = g['Timestamp'].to_numpy()
        gap_days = np.diff(t).astype('timedelta64[s]').astype(float) / 86400.0
        # attempts closer than CYCLE_GAP_DAYS (by whole days, as pandas .dt.days) belong to the same cycle
        new_cycle = np.concatenate([[True], np.floor(gap_days) > C.CYCLE_GAP_DAYS])
        cid = np.cumsum(new_cycle) - 1
        n_cycles = int(cid[-1] + 1)
        if n_cycles < C.OBLIGATION_MIN_MONTHS:
            continue
        starts = t[new_cycle]
        if pd.Series(starts).dt.to_period('M').nunique() < C.OBLIGATION_MIN_MONTHS:
            continue
        gaps = np.floor(np.diff(starts).astype('timedelta64[s]').astype(float) / 86400.0)
        if not (lo <= np.median(gaps) <= hi and np.std(gaps, ddof=1) < C.OBLIGATION_GAP_SD_MAX):
            continue
        if (g['Transaction_Type'] == 'P2P').all():
            amt = g['Amount'].astype(float)
            if amt.mean() <= 0 or amt.std() / amt.mean() >= C.RENT_AMOUNT_CV_MAX:
                continue
        ok_cycles = int(pd.Series(g['_ok'].to_numpy()).groupby(cid).any().sum())
        out.append({'name': name, 'cycles': n_cycles, 'ok_cycles': ok_cycles})
    return out


def compute_raw_features(df, window_start=C.WINDOW_START, window_end=C.WINDOW_END):
    """Raw indicators + diagnostics + flags for one customer. `df` must be sorted by Timestamp (use load_transactions).
    Rows before window_start are used only to carry the balance into the window (and for the previous-row balance)."""
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    months = pd.period_range(ws, we, freq='M')
    n_months = len(months)
    df = df.copy()
    df['unexplained'] = unexplained_flows(df)
    w = df[(df['Timestamp'] >= ws) & (df['Timestamp'] < we + pd.Timedelta(days=1))]
    att = w[w['Direction'] == 'DEBIT']
    ok = att[att['Payment_Status'] == 'SUCCESS']

    feats = {c: np.nan for c in RAW_COLUMNS + DIAGNOSTIC_COLUMNS}
    flags = []
    feats['n_attempts'], feats['n_successful_debits'] = int(len(att)), int(len(ok))
    if len(att) == 0 or len(ok) == 0:
        return feats, ['no_debits']

    # ---- TFS: mean monthly count of successful debits (retries counted once: only the success appears here)
    per_month = ok.groupby(ok['Timestamp'].dt.to_period('M')).size().reindex(months, fill_value=0)
    feats['tfs_raw'] = float(per_month.mean())

    # ---- PCS: attempt-level success ratio (paper: "payment success ratio"), computed over attempts that reflect the
    # customer's own behaviour. TECHNICAL failures (network/bank downtime) are not the customer's doing and are excluded
    # from the denominator; INSUFFICIENT_FUNDS failures count. [ENG] Requires Failure_Reason codes.
    # The paper-literal ratio (all failures counted) is kept as a diagnostic: pcs_incl_technical.
    tech = int(((att['Payment_Status'] == 'FAILED') & (att['Failure_Reason'] == 'TECHNICAL')).sum())
    feats['pcs_raw'] = len(ok) / (len(att) - tech)
    feats['pcs_incl_technical'] = len(ok) / len(att)

    # ---- daily end-of-day balance over the window (history before the window carries the balance in)
    last_day = df['Timestamp'].dt.normalize()
    daily = df.groupby(last_day)['Balance_After'].last()
    daily = daily.reindex(pd.date_range(daily.index.min(), we.normalize())).ffill().bfill()
    dw = daily[ws:we]

    # ---- LSS
    monthly_mean_bal = dw.groupby(dw.index.to_period('M')).mean().reindex(months)
    feats['L1_balance_cv'] = _cv(monthly_mean_bal.to_numpy())
    credits = w[w['Direction'] == 'CREDIT']
    inflow = credits.groupby(credits['Timestamp'].dt.to_period('M'))['Amount'].sum().reindex(months, fill_value=0.0)
    jumps = w[w['unexplained'] > C.UNEXPLAINED_JUMP_MIN]
    inflow = inflow + jumps.groupby(jumps['Timestamp'].dt.to_period('M'))['unexplained'].sum().reindex(months, fill_value=0.0)
    feats['L2_inflow_cv'] = _cv(inflow.to_numpy())
    if not np.isfinite(feats['L2_inflow_cv']):
        flags.append('no_inflow')
    if not np.isfinite(feats['L1_balance_cv']):
        flags.append('no_positive_balance')

    # ---- MDS: normalised entropy of the category mix over merchant-type successful debits
    pool = ok[ok['Transaction_Type'].isin(C.MERCHANT_TRANSACTION_TYPES)]
    if len(pool):
        cat_counts = pool['Merchant_Category'].value_counts().reindex(C.CATEGORIES, fill_value=0)
        feats['mds_raw'] = _entropy(cat_counts.to_numpy()) / np.log(len(C.CATEGORIES))
        mc = pool['Counterparty_Name'].value_counts()
        feats['n_unique_merchants'] = int(len(mc))
        feats['effective_merchants'] = float(np.exp(_entropy(mc.to_numpy())))   # diagnostic only, volume-confounded
    else:
        flags.append('no_merchant_payments')

    # ---- FRS
    out_m = float(ok['Amount'].sum() / n_months)
    feats['mean_monthly_outflow'] = out_m
    feats['F2_buffer_adequacy'] = 1.0 - float((dw < C.LOW_BALANCE_FRACTION * out_m).mean())
    obligations = detect_obligations(att)
    feats['n_obligations'] = len(obligations)
    if obligations:
        feats['F1_obligation_continuity'] = sum(o['ok_cycles'] for o in obligations) / sum(o['cycles'] for o in obligations)
    else:
        flags.append('no_obligations')

    if len(ok) < C.LOW_HISTORY_MIN_SUCCESSFUL_DEBITS:
        flags.append('low_history')
    return feats, flags
