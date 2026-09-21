"""Normalisation (frozen anchors), component scores and the UCRII composite.

Internally components are on 0-1; reported components and UCRII are on 0-100
(UCRII = sum(weight x component) is identical to the paper's "UCRII x 100" with 0-1 components).
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

# indicator -> (component, higher_is_better, natively_bounded_0_1)
INDICATORS = {
    'tfs_raw':                  ('TFS', True,  False),
    'pcs_raw':                  ('PCS', True,  True),
    'L1_balance_cv':            ('LSS', False, False),
    'L2_inflow_cv':             ('LSS', False, False),
    'mds_raw':                  ('MDS', True,  True),
    'F1_obligation_continuity': ('FRS', True,  True),
    'F2_buffer_adequacy':       ('FRS', True,  True),
}
COMPONENTS = list(C.WEIGHTS)


def fit_anchors(raw, percentiles=C.ANCHOR_PERCENTILES, meta=None):
    """Freeze normalisation anchors from a REFERENCE cohort's raw indicators (one row per customer).
    Unbounded indicators: min-max between the reference percentiles. Bounded 0-1 indicators: same, unless the
    reference spread is below NO_STRETCH_MIN_SPREAD, in which case the native 0-1 value is used unchanged."""
    lo_p, hi_p = percentiles
    anchors = {}
    for ind, (_, _, bounded) in INDICATORS.items():
        s = raw[ind].dropna()
        if len(s) < 30:
            raise ValueError(f"reference cohort too small for {ind}: {len(s)} usable customers (need >= 30)")
        a, b = float(np.percentile(s, lo_p)), float(np.percentile(s, hi_p))
        if bounded and (b - a) < C.NO_STRETCH_MIN_SPREAD:
            anchors[ind] = {'mode': 'native', 'reference_p_lo': a, 'reference_p_hi': b, 'reference_n': int(len(s))}
        elif b > a:
            anchors[ind] = {'mode': 'minmax', 'a': a, 'b': b, 'reference_n': int(len(s))}
        else:
            raise ValueError(f"degenerate reference distribution for {ind} (p{lo_p}=p{hi_p}={a})")
    return {'anchors': anchors, 'percentiles': list(percentiles), 'no_stretch_min_spread': C.NO_STRETCH_MIN_SPREAD,
            'window': [C.WINDOW_START, C.WINDOW_END], 'created_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'Anchors are derived from a SYNTHETIC reference cohort; they are not empirical Indian statistics.',
            'meta': meta or {}}


def save_anchors(anchors, path):
    with open(path, 'w') as f:
        json.dump(anchors, f, indent=1, sort_keys=True)


def load_anchors(path):
    with open(path) as f:
        return json.load(f)


def normalise(value, spec):
    """0-1 normalisation of one indicator; NaN stays NaN."""
    if value is None or not np.isfinite(value):
        return np.nan
    if spec['mode'] == 'native':
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip((value - spec['a']) / (spec['b'] - spec['a']), 0.0, 1.0))


def band(score):
    for lower, name in C.BANDS:
        if score >= lower:
            return name
    return C.BANDS[-1][1]


def score_customer(raw, flags, anchors):
    """raw: dict of raw indicators; flags: list of strings; anchors: output of fit_anchors/load_anchors.
    Returns a flat dict: component scores (0-100), UCRII (0-100), band, weighted contributions, flags, status."""
    spec = anchors['anchors']
    n = {}
    for ind, (_, higher_better, _) in INDICATORS.items():
        v = normalise(raw.get(ind, np.nan), spec[ind])
        n[ind] = v if (higher_better or not np.isfinite(v)) else 1.0 - v

    flags = list(flags)
    comp = {}
    if 'no_debits' in flags or 'no_merchant_payments' in flags or not np.isfinite(n['tfs_raw']) \
            or not np.isfinite(n['pcs_raw']) or not np.isfinite(n['mds_raw']):
        return {**{k: np.nan for k in COMPONENTS}, 'UCRII': np.nan, 'band': None,
                'status': 'insufficient_data', 'flags': ';'.join(flags)}

    comp['TFS'], comp['PCS'], comp['MDS'] = n['tfs_raw'], n['pcs_raw'], n['mds_raw']
    # LSS: an undefined CV (no positive balance / no inflow at all) is the worst outcome for that sub-indicator
    l1 = n['L1_balance_cv'] if np.isfinite(n['L1_balance_cv']) else 0.0
    l2 = n['L2_inflow_cv'] if np.isfinite(n['L2_inflow_cv']) else 0.0
    comp['LSS'] = (l1 + l2) / 2.0
    # FRS: obligation continuity (native scale) + buffer adequacy; buffer alone if no obligations were detected
    f2 = n['F2_buffer_adequacy']
    comp['FRS'] = (n['F1_obligation_continuity'] + f2) / 2.0 if np.isfinite(n['F1_obligation_continuity']) else f2

    contrib = {k: 100.0 * C.WEIGHTS[k] * comp[k] for k in COMPONENTS}
    ucrii = float(sum(contrib.values()))
    no_tfs_w = 1.0 - C.WEIGHTS['TFS']
    out = {k: 100.0 * comp[k] for k in COMPONENTS}
    out.update({'UCRII': ucrii, 'band': band(ucrii), 'status': 'ok', 'flags': ';'.join(flags),
                # diagnostic: composite without the volume-driven TFS term, reweighted to 100 (Phase 4 sensitivity)
                'UCRII_excl_TFS': float(sum(v for k, v in contrib.items() if k != 'TFS') / no_tfs_w)})
    out.update({f'contrib_{k}': v for k, v in contrib.items()})
    return out


def score_table(raw_table, flags_by_id, anchors):
    """raw_table: DataFrame indexed by customer_id with raw indicator columns. Returns one row per customer."""
    rows = []
    for cid, r in raw_table.iterrows():
        s = score_customer(r.to_dict(), flags_by_id.get(cid, []), anchors)
        rows.append({'customer_id': cid, **r.to_dict(), **s})
    return pd.DataFrame(rows)
