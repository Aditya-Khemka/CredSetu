"""[EXTENSION] Descriptive credit signals derived from ONE customer's transaction rows. Not part of UCRII: nothing here feeds
the score, has been validated against repayment outcomes, or is a prediction. Pure functions, no I/O, no Streamlit.

Thresholds and category groupings below are engineering conventions, not paper methodology. Wording is neutral: patterns
are behavioural signals, not judgements.
"""
import math

import numpy as np
import pandas as pd

from . import config as C
from .explain import window_slices
from .features import detect_obligations, unexplained_flows

INFLOW_DROP_THRESHOLD = 0.30          # [EXT] a month-over-month inflow fall beyond this is listed as a drop
ESSENTIAL_TYPE = ('Utilities', 'Telecom', 'Grocery', 'Healthcare', 'Education', 'Fuel', 'Transport')
FLEXIBLE_TYPE = ('Food', 'Entertainment', 'Shopping', 'Other')      # [EXT] grouping convention; P2P transfers kept separate
DEFAULT_DUE_DAY = 5


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ratio(a, b):
    return _num(a / b) if b else None


def _change(first, second):
    return None if not first else _num(second / first - 1.0)


def _daily_balance(df, we):
    """End-of-day balance, same construction as compute_raw_features (history carries the balance in)."""
    d = df.groupby(df['Timestamp'].dt.normalize())['Balance_After'].last()
    return d.reindex(pd.date_range(d.index.min(), pd.Timestamp(we).normalize())).ffill().bfill()


def _month_series(rows, col, months):
    return rows.groupby(rows['Timestamp'].dt.to_period('M'))[col].sum().reindex(months, fill_value=0.0)


def _cycles(att_name):
    """Billing cycles for one obligation (same gap rule as detect_obligations): list of dicts."""
    g = att_name.sort_values('Timestamp', kind='mergesort')
    t = g['Timestamp'].to_numpy()
    gap = np.diff(t).astype('timedelta64[s]').astype(float) / 86400.0
    cid = np.cumsum(np.concatenate([[True], np.floor(gap) > C.CYCLE_GAP_DAYS])) - 1
    out = []
    for i in range(int(cid[-1]) + 1):
        c = g[cid == i]
        out.append({'first': c['Timestamp'].iloc[0], 'attempts': int(len(c)), 'ok': bool((c['Payment_Status'] == 'SUCCESS').any())})
    return out


def insights(df, window_start, window_end):
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    months = pd.period_range(ws, we, freq='M')
    n_m, n_days = len(months), int((we.normalize() - ws.normalize()).days + 1)
    df = df.copy()
    df['unexplained'] = unexplained_flows(df)
    w, att, ok = window_slices(df, window_start, window_end)
    cr = w[w['Direction'] == 'CREDIT']
    jumps = w[w['unexplained'] > C.UNEXPLAINED_JUMP_MIN]
    spent = _month_series(ok, 'Amount', months)
    inflow = _month_series(cr, 'Amount', months) + _month_series(jumps, 'unexplained', months)
    events = pd.concat([cr[['Timestamp', 'Amount']], jumps[['Timestamp', 'unexplained']].rename(columns={'unexplained': 'Amount'})],
                       ignore_index=True)
    mean_in, mean_sp = float(inflow.mean()), float(spent.mean())

    # ---- fixed-payment load (recurring obligations detected by the scorer's own rule)
    obl = detect_obligations(att)
    names = [o['name'] for o in obl]
    mean_fixed = float(ok.loc[ok['Counterparty_Name'].isin(names), 'Amount'].sum() / n_m)
    net = inflow - spent
    ratio = (spent / inflow.where(inflow > 0)).dropna()
    afford = {'mean_monthly_inflow': mean_in, 'mean_monthly_spend': mean_sp,
              'median_spend_to_inflow': _num(ratio.median()) if len(ratio) else None,
              'months_spend_exceeds_inflow': int((spent > inflow).sum()), 'median_monthly_net': float(net.median()),
              'months_negative_net': int((net < 0).sum()), 'mean_monthly_fixed_payments': mean_fixed,
              'fixed_payment_load': _ratio(mean_fixed, mean_in), 'mean_inflow_after_fixed': mean_in - mean_fixed,
              'monthly': [{'Month': str(m), 'Inflow': float(inflow[m]), 'Spent': float(spent[m]), 'Net': float(net[m]),
                           'Spend_to_inflow': _ratio(spent[m], inflow[m])} for m in months]}

    # ---- income quality
    if len(events):
        pick = events.groupby(events['Timestamp'].dt.to_period('M'))['Amount'].idxmax()
        main = events.loc[pick.to_numpy()].set_axis(pick.index)
    else:
        main = pd.DataFrame(columns=['Timestamp', 'Amount'])
    days = main['Timestamp'].dt.day if len(main) else pd.Series(dtype=float)
    share = (main['Amount'] / inflow.reindex(main.index)).replace([np.inf], np.nan) if len(main) else pd.Series(dtype=float)
    src = cr.groupby('Counterparty_Name').agg(amount=('Amount', 'sum'), months=('Timestamp', lambda s: s.dt.to_period('M').nunique()))
    income = {'median_monthly_inflow': float(inflow.median()), 'months_without_inflow': int((inflow <= 0).sum()),
              'main_event_median_day': _num(days.median()) if len(days) else None,
              'main_event_day_spread': _num(days.std(ddof=1)) if len(days) > 1 else None,
              'main_event_median_share': _num(share.median()) if len(share) else None,
              'n_credit_sources': int(len(src)),
              'top_source_share': _ratio(src['amount'].max(), src['amount'].sum()) if len(src) else None,
              'recurring_credit_sources': int((src['months'] >= C.OBLIGATION_MIN_MONTHS).sum()) if len(src) else 0}

    # ---- obligation servicing
    per, tot = [], {'cycles': 0, 'ok': 0, 'retry': 0}
    for name in names:
        cy = _cycles(att[att['Counterparty_Name'] == name])
        dom = np.array([c['first'].day for c in cy], dtype=float)
        n_ok, n_retry = sum(c['ok'] for c in cy), sum(c['attempts'] > 1 for c in cy)
        per.append({'name': name, 'cycles': len(cy), 'paid_cycles': n_ok, 'missed_cycles': len(cy) - n_ok,
                    'cycles_with_retry': n_retry, 'typical_day': float(np.median(dom)),
                    'day_spread': _num(dom.std(ddof=1)) if len(dom) > 1 else None})
        tot['cycles'] += len(cy)
        tot['ok'] += n_ok
        tot['retry'] += n_retry
    servicing = {'obligations': per, 'total_cycles': tot['cycles'], 'paid_cycles': tot['ok'],
                 'missed_cycles': tot['cycles'] - tot['ok'], 'cycles_with_retry': tot['retry'],
                 'retry_share': _ratio(tot['retry'], tot['cycles'])}

    # ---- liquidity
    daily = _daily_balance(df, we)
    dw = daily[ws:we]
    thr = C.LOW_BALANCE_FRACTION * float(ok['Amount'].sum() / n_m)
    low = (dw < thr)
    runs = low.groupby((low != low.shift()).cumsum()).agg(['first', 'size'])
    runs = runs[runs['first']]['size']
    before = []
    for ts in main['Timestamp'] if len(main) else []:
        prev = ts.normalize() - pd.Timedelta(days=1)
        if prev in daily.index:
            before.append(float(daily[prev]))
    daily_spend = float(ok['Amount'].sum() / n_days)
    liquidity = {'median_balance': float(dw.median()), 'min_balance': float(dw.min()),
                 'days_of_cover': _ratio(float(dw.median()), daily_spend), 'low_balance_threshold': thr,
                 'low_balance_days': int(low.sum()), 'low_balance_episodes': int(len(runs)),
                 'longest_low_run_days': int(runs.max()) if len(runs) else 0,
                 'median_low_run_days': float(runs.median()) if len(runs) else 0.0,
                 'median_balance_before_main_inflow': float(np.median(before)) if before else None}

    # ---- stress
    chg = inflow.pct_change().replace([np.inf, -np.inf], np.nan)
    drops = chg[chg < -INFLOW_DROP_THRESHOLD]
    worst = drops.idxmin() if len(drops) else None
    isf = att[(att['Payment_Status'] == 'FAILED') & (att['Failure_Reason'] == 'INSUFFICIENT_FUNDS')]
    stress = {'inflow_drop_threshold': INFLOW_DROP_THRESHOLD, 'months_with_inflow_drop': int(len(drops)),
              'largest_inflow_drop': _num(drops.min()) if len(drops) else None, 'largest_drop_month': str(worst) if worst is not None else None,
              'spend_spike_ratio': _ratio(float(spent.max()), float(spent.median())), 'spend_spike_month': str(spent.idxmax()),
              'insufficient_funds_failures': int(len(isf)), 'months_with_insufficient_funds': int(isf['Timestamp'].dt.to_period('M').nunique())}

    # ---- direction: first half vs second half of the window
    h = n_m // 2
    halves = {'first': months[:h], 'second': months[h:]}
    fail_m = isf.groupby(isf['Timestamp'].dt.to_period('M')).size().reindex(months, fill_value=0)
    cnt_m = ok.groupby(ok['Timestamp'].dt.to_period('M')).size().reindex(months, fill_value=0)
    rows = []
    for label, s in (('Estimated inflow per month', inflow), ('Spend per month', spent), ('Successful payments per month', cnt_m),
                     ('Insufficient-funds failures per month', fail_m)):
        a, b = float(s[halves['first']].mean()), float(s[halves['second']].mean())
        rows.append({'Measure': label, 'First half': a, 'Second half': b, 'Change': _change(a, b)})
    direction = {'first_half': [str(m) for m in halves['first']], 'second_half': [str(m) for m in halves['second']], 'rows': rows}

    # ---- spending mix
    sp = ok['Amount'].sum()
    p2p = ok['Transaction_Type'] == 'P2P'
    ess = ok['Merchant_Category'].isin(ESSENTIAL_TYPE) & ~p2p
    flx = ok['Merchant_Category'].isin(FLEXIBLE_TYPE) & ~p2p
    amt = {'Essential-type categories': float(ok.loc[ess, 'Amount'].sum()), 'Flexible-type categories': float(ok.loc[flx, 'Amount'].sum()),
           'Person-to-person transfers': float(ok.loc[p2p, 'Amount'].sum())}
    amt['Other'] = float(sp) - sum(amt.values())
    mix = {'groups': [{'Group': k, 'Amount': v, 'Share': _ratio(v, float(sp))} for k, v in amt.items()],
           'flexible_spend_share_of_inflow': _ratio(amt['Flexible-type categories'] / n_m, mean_in),
           'essential': list(ESSENTIAL_TYPE), 'flexible': list(FLEXIBLE_TYPE)}

    # ---- data sufficiency
    suff = {'active_months': int((cnt_m > 0).sum()), 'n_months': n_m,
            'active_day_share': float(ok['Timestamp'].dt.normalize().nunique() / n_days),
            'months_with_inflow': int((inflow > 0).sum()), 'credit_rows': int(len(cr)),
            'successful_debits': int(len(ok)), 'below_low_history_threshold': bool(len(ok) < C.LOW_HISTORY_MIN_SUCCESSFUL_DEBITS)}

    out = {'affordability': afford, 'income': income, 'servicing': servicing, 'liquidity': liquidity, 'stress': stress,
           'direction': direction, 'mix': mix, 'sufficiency': suff}
    out['readings'] = _readings(out)
    return out


def _pct(x):
    return 'not defined' if x is None else f'{x:.0%}'


def _readings(o):
    a, i, s, q, t, d, m, u = (o[k] for k in ('affordability', 'income', 'servicing', 'liquidity', 'stress', 'direction', 'mix', 'sufficiency'))
    fx = next(r for r in d['rows'] if r['Measure'].startswith('Estimated inflow'))
    return {
        'affordability': (f"Spend was {_pct(a['median_spend_to_inflow'])} of estimated inflow in the median month; spend exceeded inflow "
                          f"in {a['months_spend_exceeds_inflow']} of {u['n_months']} months. Recurring payments averaged INR "
                          f"{a['mean_monthly_fixed_payments']:,.0f} a month ({_pct(a['fixed_payment_load'])} of average inflow)."),
        'income': (f"Estimated inflow averaged INR {a['mean_monthly_inflow']:,.0f} a month; {i['months_without_inflow']} months had none. "
                   + ("The largest monthly inflow arrived around day "
                      f"{i['main_event_median_day']:.0f} of the month (spread {i['main_event_day_spread'] or 0:.1f} days) and made up "
                      f"{_pct(i['main_event_median_share'])} of that month's inflow." if i['main_event_median_day'] is not None
                      else 'No inflow events were observed.')),
        'servicing': (f"{s['paid_cycles']} of {s['total_cycles']} recurring-payment cycles were paid; {s['cycles_with_retry']} needed a "
                      "retry." if s['total_cycles'] else 'No recurring obligations were detected, so there is no servicing record.'),
        'liquidity': (f"Median daily balance was INR {q['median_balance']:,.0f}, about {q['days_of_cover'] or 0:.0f} days of average "
                      f"spending. Balance was below INR {q['low_balance_threshold']:,.0f} on {q['low_balance_days']} days in "
                      f"{q['low_balance_episodes']} stretches (longest {q['longest_low_run_days']} days)."),
        'stress': (f"{t['months_with_inflow_drop']} months saw inflow fall by more than {t['inflow_drop_threshold']:.0%} from the previous "
                   f"month; the busiest month's spend was {t['spend_spike_ratio'] or 0:.1f} times the median month. "
                   f"{t['insufficient_funds_failures']} attempts failed for insufficient funds."),
        'direction': f"Average estimated inflow changed {_pct(fx['Change'])} between the first and second half of the window.",
        'mix': ("Flexible-type categories took "
                f"{_pct(next(g['Share'] for g in m['groups'] if g['Group'].startswith('Flexible')))} of spend."),
        'sufficiency': (f"Spending appeared in {u['active_months']} of {u['n_months']} months and on {u['active_day_share']:.0%} of days "
                        f"({u['successful_debits']} successful debits)."),
    }


def what_if(df, window_start, window_end, instalment, due_day=DEFAULT_DUE_DAY):
    """Replay the customer's own balance history with a hypothetical monthly instalment paid on `due_day`. A backtest of
    past behaviour: spending is not assumed to adapt and failures are not modelled, so it is not a forecast."""
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    months = pd.period_range(ws, we, freq='M')
    _, _, ok = window_slices(df, window_start, window_end)
    dw = _daily_balance(df, we)[ws:we]
    due = [pd.Timestamp(m.year, m.month, min(int(due_day), m.days_in_month)) for m in months]
    paid = np.searchsorted(np.array(due, dtype='datetime64[ns]'), dw.index.to_numpy(), side='right')
    sim = dw - paid * float(instalment)
    thr = C.LOW_BALANCE_FRACTION * float(ok['Amount'].sum() / len(months))
    short = sim < 0
    first = sim.index[short.argmax()] if short.any() else None
    mean_in = float(insights(df, window_start, window_end)['affordability']['mean_monthly_inflow'])
    return {
        'instalment': float(instalment), 'due_day': int(due_day),
        'base_low_days': int((dw < thr).sum()), 'with_instalment_low_days': int((sim < thr).sum()),
        'days_balance_negative': int(short.sum()), 'months_with_shortfall': int(sim[short].index.to_period('M').nunique()),
        'first_shortfall_date': None if first is None else str(first.date()), 'lowest_balance': float(sim.min()),
        'share_of_mean_inflow': _ratio(float(instalment), mean_in),
        'series': pd.DataFrame({'Date': dw.index, 'Base': dw.to_numpy(), 'With instalment': sim.to_numpy()}),
    }
