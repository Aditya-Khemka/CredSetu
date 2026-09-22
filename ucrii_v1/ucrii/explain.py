"""Deterministic, template-based explanation of one UCRII score. Pure data + strings: no I/O, no Streamlit, no scoring.

[ENG] This module only describes numbers the scorer already produced, using the same window slicing as
features.compute_raw_features. Wording is descriptive; patterns are behavioural signals, not judgements.
"""
import math

import numpy as np
import pandas as pd

from . import config as C
from .features import detect_obligations

NAMES = {'TFS': 'Transaction frequency', 'PCS': 'Payment consistency', 'LSS': 'Liquidity stability',
         'MDS': 'Merchant diversity', 'FRS': 'Financial resilience'}
PULL_MIN_POINTS_LOST = 5.0
HOLD_MIN_SCORE = 75.0
MAX_DRIVERS = 2


def _num(x):
    """float or None (NaN / None / non-numeric -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rank_drivers(comp_scores):
    """comp_scores: {'TFS': 0-100, ...}. Returns (pulls_down, holds_up) as lists of (component, value):
    pulls_down = up to 2 components with points_lost >= 5, largest first (value = points lost);
    holds_up = up to 2 components with score >= 75, by contribution (value = contribution in points)."""
    lost = {k: 100.0 * C.WEIGHTS[k] - C.WEIGHTS[k] * comp_scores[k] for k in C.WEIGHTS}
    contrib = {k: C.WEIGHTS[k] * comp_scores[k] for k in C.WEIGHTS}
    order = list(C.WEIGHTS)                                           # stable tie-break: paper order
    down = sorted((k for k in order if lost[k] >= PULL_MIN_POINTS_LOST), key=lambda k: -lost[k])[:MAX_DRIVERS]
    up = sorted((k for k in order if comp_scores[k] >= HOLD_MIN_SCORE), key=lambda k: -contrib[k])[:MAX_DRIVERS]
    return [(k, lost[k]) for k in down], [(k, contrib[k]) for k in up]


def window_slices(df, window_start, window_end):
    """Same slicing as compute_raw_features: (window rows, debit attempts, successful debits)."""
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    w = df[(df['Timestamp'] >= ws) & (df['Timestamp'] < we + pd.Timedelta(days=1))]
    att = w[w['Direction'] == 'DEBIT']
    return w, att, att[att['Payment_Status'] == 'SUCCESS']


def _facts(df, window_start, window_end, raw):
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    _, att, ok = window_slices(df, window_start, window_end)
    failed = att[att['Payment_Status'] == 'FAILED']
    tech = int((failed['Failure_Reason'] == 'TECHNICAL').sum())
    isf = int((failed['Failure_Reason'] == 'INSUFFICIENT_FUNDS').sum())
    pool = ok[ok['Transaction_Type'].isin(C.MERCHANT_TRANSACTION_TYPES)]
    obl = detect_obligations(att)
    n_days = int((we.normalize() - ws.normalize()).days + 1)
    f2 = _num(raw.get('F2_buffer_adequacy'))
    return {
        'window': [str(ws.date()), str(we.date())], 'n_days': n_days,
        'n_attempts': int(len(att)), 'n_successful_debits': int(len(ok)), 'n_technical_failures': tech,
        'n_insufficient_funds': isf, 'n_counted_attempts': int(len(att) - tech),
        'n_other_counted_failures': int(len(att) - tech - len(ok) - isf),
        'categories_covered': int(pool.loc[pool['Merchant_Category'].isin(C.CATEGORIES), 'Merchant_Category'].nunique()),
        'n_categories': len(C.CATEGORIES),
        'obligations': [dict(o) for o in obl], 'n_obligations': len(obl),
        'obligation_cycles': int(sum(o['cycles'] for o in obl)), 'obligation_paid_cycles': int(sum(o['ok_cycles'] for o in obl)),
        'low_balance_days': None if f2 is None else int(round((1.0 - f2) * n_days)),
    }


def _component_texts(raw, scores, anchors, f):
    """Short, plain-language descriptions. Each still carries the number it's based on (required for level statements)."""
    a = anchors['anchors']['tfs_raw']
    tfs = _num(raw.get('tfs_raw')) or 0.0
    t = {'TFS': f"Averaged {tfs:.0f} payments a month (typical range {a.get('a', 0):.0f}-{a.get('b', 0):.0f})."}
    other = f", {f['n_other_counted_failures']} for other reasons" if f['n_other_counted_failures'] > 0 else ''
    t['PCS'] = (f"{f['n_successful_debits']} of {f['n_counted_attempts']} payments succeeded ({f['n_insufficient_funds']} "
                f"for insufficient funds{other}). {f['n_technical_failures']} technical failures excluded.")
    l1, l2 = _num(raw.get('L1_balance_cv')), _num(raw.get('L2_inflow_cv'))
    bal = f"{l1:.0%}" if l1 is not None else "not observed (scored as 0%)"
    inf = f"{l2:.0%}" if l2 is not None else "not observed (scored as 0%)"
    t['LSS'] = f"Balance varied {bal}; inflow varied {inf} month to month."
    mds = _num(raw.get('mds_raw')) or 0.0
    t['MDS'] = f"Payments covered {f['categories_covered']} of {f['n_categories']} categories (evenness {mds:.2f})."
    low = f"Balance was low ({C.LOW_BALANCE_FRACTION:.0%} of typical outflow) on {f['low_balance_days']} of {f['n_days']} days."
    if f['n_obligations']:
        t['FRS'] = f"{f['n_obligations']} recurring payments; {f['obligation_paid_cycles']} of {f['obligation_cycles']} cycles paid. {low}"
    else:
        t['FRS'] = f"No recurring payments detected. {low}"
    return t


def _caveats(flags, raw, anchors, f):
    c = []
    if 'low_history' in flags:
        c.append(f"Limited history ({f['n_successful_debits']} successful debits in the window; threshold "
                 f"{C.LOW_HISTORY_MIN_SUCCESSFUL_DEBITS}): TFS is floored and PCS/MDS are noisy; treat the band as low-confidence.")
    if 'no_obligations' in flags:
        c.append("No recurring obligations were detected, so FRS rests on the balance buffer alone.")
    if 'no_inflow' in flags:
        c.append("No inflow was observed in the window, so the inflow-variation part of LSS is scored at its minimum.")
    if 'no_positive_balance' in flags:
        c.append("No positive balance was observed, so the balance-variation part of LSS is scored at its minimum.")
    l2, up = _num(raw.get('L2_inflow_cv')), anchors['anchors']['L2_inflow_cv'].get('b')
    if l2 is not None and up is not None and l2 > up:
        c.append(f"Irregular monthly inflow (variation {l2:.0%} vs reference upper anchor {up:.0%}) lowers LSS by "
                 "construction; this is not evidence about repayment.")
    c += ["This is synthetic data; the score is an explainable alternative-data indicator, not a probability of default.",
          "Weights and bands come from an unvalidated preprint; the band thresholds are hypothetical.",
          "Technical failures are excluded from PCS; this needs failure reason codes in the data.",
          "TFS reflects activity volume, so low-activity customers score lower on it by construction."]
    return c


def explain(df, window_start, window_end, raw, flags, scores, anchors):
    """df: the customer's transactions; raw/flags: compute_raw_features output; scores: score_customer output."""
    flags = list(flags)
    if scores.get('status') != 'ok':
        return {'headline': "Not scored: insufficient data (" + ', '.join(flags or ['unknown']) + ").",
                'pulls_down': [], 'holds_up': [], 'components': {}, 'facts': {},
                'caveats': ["This is synthetic data; the score is not a probability of default."]}
    f = _facts(df, window_start, window_end, raw)
    texts = _component_texts(raw, scores, anchors, f)
    comps, comp_scores = {}, {k: float(scores[k]) for k in C.WEIGHTS}
    for k, w in C.WEIGHTS.items():
        s = comp_scores[k]
        comps[k] = {'name': NAMES[k], 'score': s, 'weight': w, 'contribution': w * s, 'points_lost': 100.0 * w - w * s,
                    'text': texts[k]}
    down, up = rank_drivers(comp_scores)
    pulls = [{'component': k, 'points_lost': v, 'text': f"{NAMES[k]}: -{v:.1f} points."} for k, v in down]
    holds = [{'component': k, 'contribution': v, 'text': f"{NAMES[k]}: +{v:.1f} points (score {comp_scores[k]:.0f}/100)."}
             for k, v in up]
    if not pulls:
        pulls = [{'component': None, 'points_lost': 0.0, 'text': "No component is materially below its maximum."}]
    if not holds:
        holds = [{'component': None, 'contribution': 0.0, 'text': "No component scores above 75."}]
    head = f"UCRII {scores['UCRII']:.1f}/100 - {scores['band']} ({scores['confidence']} confidence)."
    if down:
        head += f" Biggest drag: {NAMES[down[0][0]]} (-{down[0][1]:.1f} pts)."
    return {'headline': head, 'pulls_down': pulls, 'holds_up': holds, 'components': comps,
            'caveats': _caveats(flags, raw, anchors, f), 'facts': f}
