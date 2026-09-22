"""Streamlit UI only: every computation lives in ui/services.py and ucrii/. Run from ucrii_v1/:  streamlit run ui/app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt                       # noqa: E402  (bundled with streamlit)
import pandas as pd                        # noqa: E402
import streamlit as st                     # noqa: E402

from ui import services as S               # noqa: E402
from ui import settings                    # noqa: E402
from ucrii import config as C              # noqa: E402
from ucrii.insights import DEFAULT_DUE_DAY  # noqa: E402

BAND_COLORS = [(0, 40, 'Very high risk', '#e6a5a5'), (40, 60, 'High risk', '#f0c9a0'),
               (60, 80, 'Moderate risk', '#efe3a0'), (80, 100, 'Low risk', '#b9dcb0')]
FLAG_LABELS = {'low_history': 'Limited history', 'no_obligations': 'No recurring obligations detected',
               'no_inflow': 'No inflow observed', 'no_positive_balance': 'No positive balance observed',
               'no_debits': 'No debits in window', 'no_merchant_payments': 'No merchant payments'}
NOT_USED = ['Age group', 'State', 'Bank', 'Device type', 'Customer name', 'Network type',
            'Latent generator parameters (_latent/)', 'The xlsx Profile sheet and manifest.csv']
FORMULAS = """
- 5 indicators from the transaction rows: payment frequency (TFS), success rate (PCS), balance/inflow stability (LSS),
  category spread (MDS), obligation and buffer health (FRS).
- Each is normalised against a synthetic reference cohort, then combined: **UCRII = 100 x (0.25 TFS + 0.25 PCS + 0.20 LSS
  + 0.15 MDS + 0.15 FRS)**.
- Bands (80+ Low, 60-80 Moderate, 40-60 High, below 40 Very high) and the weights above are illustrative, from an
  unvalidated preprint - not calibrated thresholds.
- Scoring window: 2024-02-01 to 2024-12-31 (January is burn-in, excluded).
"""

st.set_page_config(page_title='UCRII prototype', layout='wide')


# ------------------------------------------------------------------ helpers
def _set_pan(value):
    st.session_state['pan_input'] = value


def _pick_existing():
    if st.session_state.get('existing_pick'):
        _set_pan(st.session_state['existing_pick'])


def _new_pan():
    _set_pan(S.random_new_pan())


def meter(score):
    bands = pd.DataFrame(BAND_COLORS, columns=['lo', 'hi', 'band', 'color'])
    base = alt.Chart(bands).mark_bar(height=28).encode(
        x=alt.X('lo:Q', scale=alt.Scale(domain=[0, 100]), title='UCRII (0-100); band thresholds 40 / 60 / 80'),
        x2='hi:Q', color=alt.Color('color:N', scale=None), tooltip=['band', 'lo', 'hi'])
    mark = alt.Chart(pd.DataFrame({'x': [score]})).mark_rule(size=4, color='#222').encode(x='x:Q')
    return (base + mark).properties(height=70)


@st.cache_data(show_spinner=False)
def _activity(path, mtime_ns):
    return S.activity(path)


# ------------------------------------------------------------------ header + notice
st.title('UCRII: alternative-data reliability score (prototype)')
# ------------------------------------------------------------------ sidebar
existing = S.existing_pans()
with st.sidebar:
    st.header('Customer')
    st.button('Random new PAN', on_click=_new_pan)
    st.selectbox(f'Existing customers ({len(existing)})', [''] + existing, key='existing_pick', on_change=_pick_existing)
    with st.form('pan_form'):
        pan_text = st.text_input('PAN', key='pan_input', max_chars=12,
                                 help="Format: 3 letters, 'P', 1 letter, 4 digits, 1 letter (e.g. ABCPD1234E).")
        st.caption('Used only when a history must be generated:')
        age = st.selectbox('Age group', S.AGE_CHOICES)
        activity = st.selectbox('Activity level', S.ACTIVITY_CHOICES)
        situation = st.selectbox('Situation (demo preset)', list(S.SITUATIONS))
        with st.expander('Advanced'):
            income = st.selectbox('Income type', ['(default)'] + S.income_types())
        regenerate = st.checkbox('Regenerate even if a history exists (overwrites)', value=False)
        submitted = st.form_submit_button('Generate / Score')
    st.caption(f'Analysis window (fixed): {C.WINDOW_START} to {C.WINDOW_END}; history from 2024-01-01 (January is burn-in).')
    try:
        from upi_user_generator import GENERATOR_VERSION      # type: ignore  (path set by services)
        st.caption(f'ucrii {S.UCRII_VERSION} | generator {GENERATOR_VERSION} | anchors {settings.anchors_path().name}')
    except ImportError:
        st.caption(f'ucrii {S.UCRII_VERSION} | anchors {settings.anchors_path().name}')

# ------------------------------------------------------------------ run
if submitted:
    st.session_state.pop('error', None)
    try:
        st.session_state['result'] = S.score_pan(
            pan_text, age, activity, situation, None if income == '(default)' else income, regenerate)
    except S.ServiceError as e:
        st.session_state.pop('result', None)
        st.session_state['error'] = str(e)
    except Exception as e:                                       # never show a traceback
        st.session_state.pop('result', None)
        st.session_state['error'] = f'Could not score this customer ({type(e).__name__}). Try Regenerate.'

if st.session_state.get('error'):
    st.error(st.session_state['error'])
res = st.session_state.get('result')

if res is None:
    st.subheader('How it works')
    st.code('Transactions -> features -> TFS / PCS / LSS / MDS / FRS -> UCRII -> explanation', language=None)
    st.caption('Enter a PAN in the sidebar. If no history exists for it, a synthetic one is generated deterministically '
               'from the PAN; otherwise the existing file is scored as-is.')
    st.stop()

# ------------------------------------------------------------------ result banners
src = res['source']

scores, expl = res['scores'], res['explanation']
tabs = st.tabs(['Score', 'Why this score', 'Activity and balance', 'Spending analytics', 'Credit signals', 'Data and method'])
csv_path = settings.data_dir() / f"{res['pan']}.csv"

# ---- 1. Score
with tabs[0]:
    if res['status'] != 'ok':
        st.subheader('Not scored')
        st.write('Insufficient data for a score. Reason: ' + ', '.join(FLAG_LABELS.get(f, f) for f in res['flags']))
    else:
        if 'low_history' in res['flags']:
            st.warning(f"Limited history ({res['raw']['n_successful_debits']} successful debits in the window; threshold "
                       f"{C.LOW_HISTORY_MIN_SUCCESSFUL_DEBITS}): TFS is floored and PCS/MDS are noisy; treat the band as low-confidence.")
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.metric('UCRII', f"{scores['UCRII']:.1f}")
        c2.metric('Risk band (illustrative)', scores['band'])
        c3.metric('Confidence', res['confidence'])
        for f in res['flags']:
            st.badge(FLAG_LABELS.get(f, f), color='gray')
        st.altair_chart(meter(scores['UCRII']), width='stretch')
        cols = st.columns(5)
        for col, k in zip(cols, C.WEIGHTS):
            col.metric(k, f"{scores[k]:.0f}")
            col.caption(f"weight {C.WEIGHTS[k]:.0%}; contributes {res['contributions'][k]:.1f} points")
        st.metric('Behaviour-only view (excludes TFS)', f"{scores['UCRII_excl_TFS']:.1f}")
        st.caption('TFS reflects activity volume, so low-activity customers score lower on it by construction.')
        if st.toggle('Compare with cohort'):
            with st.spinner('Scoring the cohort folder (cached until it changes)...'):
                cp = S.cohort_percentiles(res['pan'])
            if cp:
                st.dataframe(pd.DataFrame({'Percentile in cohort': {k: f'{v:.0f}' for k, v in cp['percentiles'].items()}}))
                st.caption(f"Share of the {cp['n']} scored customers in the data folder (this one included) at or below this "
                           'score. The cohort is synthetic, so this is not a comparison with real customers.')

# ---- 2. Why
with tabs[1]:
    if not expl or res['status'] != 'ok':
        st.write('No explanation: the customer was not scored.')
    else:
        st.subheader(expl['headline'])
        a, b = st.columns(2)
        a.markdown('**What pulls the score down**')
        for d in expl['pulls_down']:
            a.write(d['text'])
        b.markdown('**What holds it up**')
        for d in expl['holds_up']:
            b.write(d['text'])
        contrib = pd.DataFrame([{'component': k, 'points': v['contribution']} for k, v in expl['components'].items()])
        st.altair_chart(alt.Chart(contrib).mark_bar(height=30).encode(
            x=alt.X('points:Q', stack='zero', scale=alt.Scale(domain=[0, 100]), title=f"Contributions sum to UCRII {scores['UCRII']:.1f}"),
            color=alt.Color('component:N', sort=list(C.WEIGHTS)), tooltip=['component', alt.Tooltip('points:Q', format='.1f')]
        ).properties(height=90), width='stretch')
        table = pd.DataFrame([{'Component': k, 'Score': f"{v['score']:.0f}/100", 'Weight': f"{v['weight']:.0%}",
                               'Points': f"{v['contribution']:.1f}"} for k, v in expl['components'].items()])
        st.dataframe(table, hide_index=True)
        with st.expander('Component details'):
            for k, v in expl['components'].items():
                st.markdown(f"**{k}** ({v['name']}): {v['text']}")
        with st.expander('Caveats'):
            for c in expl['caveats']:
                st.write('- ' + c)

# ---- 3. Activity
with tabs[2]:
    try:
        act = _activity(str(csv_path), csv_path.stat().st_mtime_ns)
        d = act['daily_balance']
        shade = pd.DataFrame([{'a': '2024-01-01', 'b': C.WINDOW_START, 'part': 'Burn-in month'},
                              {'a': C.WINDOW_START, 'b': C.WINDOW_END, 'part': 'Analysis window'}])
        shade[['a', 'b']] = shade[['a', 'b']].apply(pd.to_datetime)
        bg = alt.Chart(shade).mark_rect(opacity=0.15).encode(x='a:T', x2='b:T', color=alt.Color('part:N', title=None))
        line = alt.Chart(d).mark_line().encode(x=alt.X('Date:T', title=None), y=alt.Y('Balance:Q', title='End-of-day balance'))
        st.subheader('Daily balance')
        st.altair_chart(bg + line, width='stretch')
        l, r = st.columns(2)
        l.subheader('Successful debits per month')
        l.bar_chart(act['monthly_successful'], x='Month', y='Successful debits')
        r.subheader('Failed attempts per month, by reason')
        r.bar_chart(act['monthly_failed'], x='Month', y=['INSUFFICIENT_FUNDS', 'TECHNICAL'])
        st.subheader('Estimated monthly inflow')
        st.bar_chart(act['monthly_inflow'], x='Month', y='Estimated inflow')
        st.subheader('Detected recurring obligations')
        obl = (expl or {}).get('facts', {}).get('obligations', [])
        if obl:
            st.dataframe(pd.DataFrame(obl).rename(columns={'name': 'Counterparty', 'cycles': 'Cycles', 'ok_cycles': 'Paid cycles'}),
                         hide_index=True)
        else:
            st.write('None detected.')
    except Exception as e:
        st.warning(f'Charts unavailable ({type(e).__name__}).')

# ---- 4. Spending analytics
@st.cache_data(show_spinner=False)
def _analytics(path, mtime_ns):
    return S.analytics(path)


with tabs[3]:
    try:
        an = _analytics(str(csv_path), csv_path.stat().st_mtime_ns)
        k = an['kpis']
        inr = lambda v: f"INR {v:,.0f}"                                        # noqa: E731
        r1 = st.columns(4)
        r1[0].metric('Total spent (successful debits)', inr(k['total_spent']))
        r1[1].metric('Total received (credits)', inr(k['total_received']))
        r1[2].metric('Median payment', inr(k['median_payment']))
        r1[3].metric('Largest payment', inr(k['largest_payment']))
        r2 = st.columns(4)
        r2[0].metric('Active days', f"{k['active_days']} of {k['n_days']}")
        r2[1].metric('Payments per active day', f"{k['payments_per_active_day']:.1f}")
        r2[2].metric('Weekend share of payments', f"{k['weekend_share']:.0%}")
        r2[3].metric('Top-5 counterparties, share of spend', f"{k['top5_counterparty_share']:.0%}")
        st.caption(f"Window {C.WINDOW_START} to {C.WINDOW_END}. Descriptive only: these figures are not inputs to the score. "
                   'Balance is bank-side; median daily balance ' + inr(k['median_balance']) + ', lowest ' + inr(k['min_balance']) + '.')
        st.subheader('Monthly cash flow')
        st.bar_chart(an['monthly_flow'].melt('Month', ['Spent', 'Received'], 'Type', 'INR'), x='Month', y='INR', color='Type',
                     stack=False)
        st.dataframe(an['monthly_flow'].round(0), hide_index=True)
        a, b = st.columns(2)
        a.subheader('Spend by category')
        a.bar_chart(an['by_category'], x='Merchant_Category', y='Amount', horizontal=True)
        b.subheader('Spend by payment type')
        b.bar_chart(an['by_type'], x='Transaction_Type', y='Amount', horizontal=True)
        a.caption("'P2P' = person-to-person transfers, which have no merchant category.")
        st.dataframe(an['by_category'].assign(Share=lambda t: (t['Share'] * 100).round(1)).rename(
            columns={'Merchant_Category': 'Category', 'Share': 'Share of spend (%)'}), hide_index=True)
        c, d = st.columns(2)
        c.subheader('Payment size distribution')
        c.bar_chart(an['amount_hist'], x='Amount (INR)', y='Payments', sort=False)
        d.subheader('Payments by weekday')
        d.bar_chart(an['by_weekday'], x='Day', y='Payments', sort=False)
        st.subheader('Payments by hour of day')
        st.bar_chart(an['by_hour'], x='Hour', y='Payments')
        st.subheader('Top counterparties by amount')
        st.dataframe(an['top_counterparties'].assign(Share=lambda t: (t['Share'] * 100).round(1)).rename(
            columns={'Counterparty_Name': 'Counterparty', 'Share': 'Share of spend (%)'}), hide_index=True)
    except Exception as e:
        st.warning(f'Analytics unavailable ({type(e).__name__}).')

# ---- 5. Credit signals  [EXTENSION]
@st.cache_data(show_spinner=False)
def _signals(path, mtime_ns):
    return S.credit_signals(path)


with tabs[4]:
    if res['status'] != 'ok':
        st.write('No credit signals: the customer was not scored.')
    else:
        try:
            sg = _signals(str(csv_path), csv_path.stat().st_mtime_ns)
            rd = sg['readings']
            inr = lambda v: 'n/a' if v is None else f"INR {v:,.0f}"                      # noqa: E731
            pc = lambda v: 'n/a' if v is None else f"{v:.0%}"                            # noqa: E731
            af, inc, sv, lq, stx, dr, mx, sf = (sg[k] for k in ('affordability', 'income', 'servicing', 'liquidity', 'stress',
                                                               'direction', 'mix', 'sufficiency'))
            st.subheader('Affordability')
            st.write(rd['affordability'])
            m = st.columns(4)
            m[0].metric('Average monthly inflow (estimated)', inr(af['mean_monthly_inflow']))
            m[1].metric('Median spend-to-inflow', pc(af['median_spend_to_inflow']))
            m[2].metric('Recurring payments, share of inflow', pc(af['fixed_payment_load']))
            m[3].metric('Median monthly net cash flow', inr(af['median_monthly_net']))
            mon = pd.DataFrame(af['monthly'])
            st.line_chart(mon, x='Month', y='Spend_to_inflow')
            st.caption('Monthly spend divided by estimated inflow (1.0 = spend equals inflow).')
            with st.expander('Monthly table'):
                st.dataframe(mon.round(2), hide_index=True)

            st.subheader('Income quality')
            st.write(rd['income'])
            m = st.columns(4)
            m[0].metric('Months without inflow', inc['months_without_inflow'])
            m[1].metric('Main inflow: median day of month', 'n/a' if inc['main_event_median_day'] is None else f"{inc['main_event_median_day']:.0f}")
            m[2].metric('Largest credit source, share', pc(inc['top_source_share']))
            m[3].metric('Recurring credit sources', inc['recurring_credit_sources'])
            st.caption('Inflow = credits plus balance jumps not explained by UPI payments (income that arrives outside UPI).')

            st.subheader('Obligation servicing')
            st.write(rd['servicing'])
            if sv['obligations']:
                st.dataframe(pd.DataFrame(sv['obligations']).rename(columns={
                    'name': 'Counterparty', 'cycles': 'Cycles', 'paid_cycles': 'Paid', 'missed_cycles': 'Missed',
                    'cycles_with_retry': 'Needed retry', 'typical_day': 'Typical day', 'day_spread': 'Day spread (sd)'}), hide_index=True)

            st.subheader('Liquidity')
            st.write(rd['liquidity'])
            m = st.columns(4)
            m[0].metric('Median daily balance', inr(lq['median_balance']))
            m[1].metric('Days of cover', 'n/a' if lq['days_of_cover'] is None else f"{lq['days_of_cover']:.0f}")
            m[2].metric('Low-balance stretches', lq['low_balance_episodes'])
            m[3].metric('Balance before main inflow (median)', inr(lq['median_balance_before_main_inflow']))
            st.caption('Days of cover = median daily balance / average daily spending. Low balance = below '
                       f"{C.LOW_BALANCE_FRACTION:.0%} of a typical month's outflow (same rule as FRS).")

            st.subheader('Stress and recovery')
            st.write(rd['stress'])
            m = st.columns(4)
            m[0].metric('Months with inflow drop', stx['months_with_inflow_drop'])
            m[1].metric('Largest inflow drop', pc(stx['largest_inflow_drop']))
            m[2].metric('Busiest-month spend vs median', 'n/a' if stx['spend_spike_ratio'] is None else f"{stx['spend_spike_ratio']:.1f}x")
            m[3].metric('Insufficient-funds failures', stx['insufficient_funds_failures'])

            st.subheader('Direction: first half vs second half')
            st.write(rd['direction'])
            d = pd.DataFrame(dr['rows'])
            d['Change'] = d['Change'].map(pc)
            st.dataframe(d.round(1), hide_index=True)
            st.caption(f"First half: {dr['first_half'][0]} to {dr['first_half'][-1]}; second half: {dr['second_half'][0]} to {dr['second_half'][-1]}.")

            st.subheader('Spending mix')
            st.write(rd['mix'])
            g = pd.DataFrame(mx['groups'])
            st.bar_chart(g, x='Group', y='Amount', horizontal=True)
            st.caption('Essential-type: ' + ', '.join(mx['essential']) + '. Flexible-type: ' + ', '.join(mx['flexible'])
                       + '. Groupings are a labelling convention, not a judgement about the customer.')

            st.subheader('Data sufficiency')
            st.write(rd['sufficiency'])

            st.subheader('Instalment what-if')
            st.caption('Replays this customer\'s own balance history with a hypothetical monthly instalment added. A backtest of '
                       'past behaviour, not a forecast: spending is assumed not to adapt and payment failures are not modelled.')
            top = max(1000, int(round(af['mean_monthly_inflow'] / 500.0) * 500))
            c1, c2 = st.columns(2)
            inst = c1.slider('Monthly instalment (INR)', 0, top, min(top, int(round(0.2 * af['mean_monthly_inflow'] / 500.0) * 500)), 500)
            due = c2.slider('Due day of month', 1, 28, DEFAULT_DUE_DAY)
            wi = S.instalment_what_if(csv_path, inst, due)
            m = st.columns(4)
            m[0].metric('Instalment, share of average inflow', pc(wi['share_of_mean_inflow']))
            m[1].metric('Days balance would be below zero', wi['days_balance_negative'])
            m[2].metric('Months with a shortfall', wi['months_with_shortfall'])
            m[3].metric('Low-balance days (before -> with)', f"{wi['base_low_days']} -> {wi['with_instalment_low_days']}")
            st.line_chart(wi['series'].melt('Date', ['Base', 'With instalment'], 'Series', 'Balance'), x='Date', y='Balance', color='Series')
            st.caption('First shortfall: ' + (wi['first_shortfall_date'] or 'none in the window')
                       + '. Balances below zero mean the instalment could not have been covered from the balance at that point.')
        except Exception as e:
            st.warning(f'Credit signals unavailable ({type(e).__name__}).')

# ---- 6. Data and method
with tabs[5]:
    st.subheader('Transactions (first 200 rows)')
    try:
        st.dataframe(pd.read_csv(csv_path, nrows=200), hide_index=True)
        st.download_button('Download transactions CSV', csv_path.read_bytes(), file_name=f"{res['pan']}.csv", mime='text/csv')
    except Exception as e:
        st.warning(f'Transactions unavailable ({type(e).__name__}).')
    st.download_button('Download full report (JSON)', json.dumps(res, indent=1), file_name=f"{res['pan']}_report.json",
                       mime='application/json')
    st.markdown('**Columns the score reads:** ' + ', '.join(C.REQUIRED_COLUMNS))
    st.markdown('**Not used by the score:** ' + '; '.join(NOT_USED))
    with st.expander('How the score is calculated'):
        st.markdown(FORMULAS)
    with st.expander('Known limitations'):
        st.markdown(S.limitations_markdown())
