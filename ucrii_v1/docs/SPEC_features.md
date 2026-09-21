# UCRII: how the score works (V1.1)

UCRII (UPI Credit Risk Intelligence Index) turns a customer's UPI transaction history into one number from 0 to 100.
A higher number means the customer's UPI behaviour looks more reliable.

It comes from Ranjan, *UPI Credit Risk Intelligence Index (UCRII)*, SSRN 6973243 (21 Jun 2026). That paper is a
conceptual preprint and calls its own tables "purely hypothetical". It names five dimensions but gives no formulas,
so **every formula on this page is our own engineering choice**.

> **What the score is not.** It is an explainable reliability indicator, not a probability of default. There are no
> repayment outcomes in the data, so we make no claim that it predicts anything.

Status: implemented in `ucrii/` (version 1.1), with 34 unit tests and four validation scripts in `validation/`.

---

## 1. Quick summary

The score is a weighted average of five components, each scored 0–100:

| Component | Weight | Plain-English question | Formula in one line |
|---|---|---|---|
| **TFS** Transaction Frequency | 0.25 | Does the customer use UPI regularly? | Average successful payments per month |
| **PCS** Payment Consistency | 0.25 | Do their payments go through? | Successful debits ÷ debit attempts (technical failures excluded) |
| **LSS** Liquidity Stability | 0.20 | Are balance and income steady? | How little balances and inflows swing month to month |
| **MDS** Merchant Diversity | 0.15 | Do they pay a spread of merchant types? | Entropy of spend across 11 categories |
| **FRS** Financial Resilience | 0.15 | Do recurring bills get paid, and is there a cash buffer? | Bill success rate + days with a healthy balance |

```
UCRII = 0.25·TFS + 0.25·PCS + 0.20·LSS + 0.15·MDS + 0.15·FRS
```

| UCRII | Band |
|---|---|
| 80 and above | Low risk |
| 60 to 80 | Moderate risk |
| 40 to 60 | High risk |
| below 40 | Very high risk |

Bands are stored as lower bounds (`config.BANDS`), so a fractional score such as 79.6 is classified correctly.
The weights and bands come from the paper and are **unvalidated**. Nothing here tunes or learns them.

---

## 2. What comes from where

Keeping these four layers separate matters. Blurring them is the easiest way to overclaim.

| Layer | What it covers |
|---|---|
| **Paper** | The five dimension names and one-line descriptions, the weights, the risk bands, and the min-max normalisation idea. |
| **Synthetic-data assumptions** | Everything inside the data generator: income types, spending ratios, shocks, failure rates, age mixes. These are made up, not real statistics. |
| **Engineering decisions** | Every formula, window and threshold in section 5. Each constant in `config.py` is tagged `[PAPER]` or `[ENG]`. |
| **Extensions** | Diagnostics only (`pcs_incl_technical`, `UCRII_excl_TFS`, merchant counts). They are reported but never enter the score. |

---

## 3. Code layout

| File | Role |
|---|---|
| `ucrii/config.py` | All constants (weights, bands, window, thresholds), each tagged by provenance. |
| `ucrii/features.py` | `load_transactions` (read and validate one CSV) and `compute_raw_features` (raw indicators, diagnostics, flags for one customer). |
| `ucrii/scoring.py` | Anchor fitting, normalisation, component scores, composite, band, confidence. |
| `ucrii/cli.py` | `fit-anchors` and `score` commands; builds the raw-indicator table for a cohort directory. |
| `ucrii/anchors_reference_v1_1.json` | The frozen anchors used for scoring. |
| `validation/` | Scenario, dynamics, fairness and sparse-anchor scripts (section 9). |
| `tests/test_ucrii.py` | 34 unit tests. |

**Data flow:**

```
customer CSV
  → load_transactions()          validate columns, stable sort by Timestamp
  → compute_raw_features()       7 raw indicators + diagnostics + flags
  → normalise() with anchors     each indicator → 0–1
  → score_customer()             5 components → UCRII, band, confidence
```

---

## 4. Inputs

**Input:** one transaction CSV per customer (generator v2 schema). The customer id is the file name.

**Required columns:** `Timestamp`, `Direction`, `Transaction_Type`, `Merchant_Category`, `Counterparty_Name`, `Amount`,
`Payment_Status`, `Failure_Reason`, `Balance_After`. A missing column raises `ValueError`.

**Never read:** subfolders (notably `_latent/`), the xlsx `Profile` sheet, and `manifest.csv` (age group, state, bank,
name). The CLI only opens top-level `*.csv` files other than `manifest.csv`. The score depends on transactions only.

**Scoring window:** 1 Feb 2024 to 31 Dec 2024 (11 months), giving one score as of 31 Dec.
January is skipped as burn-in: the synthetic opening balance is arbitrary, and January's insufficient-funds rate was
about 2.5× that of other months. Rows before the window are used only to carry the balance into it.

**Notation used below**

| Symbol | Meaning |
|---|---|
| D | debit attempts in the window (`Direction == DEBIT`, any status) |
| Ds | successful debits in the window |
| B_d | end-of-day `Balance_After`: the last row of each calendar day, forward-filled on days without transactions |

---

## 5. The five components

Most indicators go through a helper `n(x)` that maps a raw value onto 0–1:

```
n(x) = clip( (x − a) / (b − a), 0, 1 )
```

`a` and `b` are fixed anchors (section 6). When a **higher raw value is worse**, the score uses `1 − n(x)`.

### TFS: Transaction Frequency

- `x` = mean over the 11 calendar months of the number of successful debits in that month (months with none count as 0).
- `TFS = n(x)`, anchors a = 25.17, b = 78.28.
- A failed attempt followed by a successful retry counts once, because only the success is in Ds.
- Credits are not counted. The paper describes frequency as something "performed by a borrower".

### PCS: Payment Consistency

- `x = |Ds| / (|D| − |technical failures|)`. `PCS = n(x)`, anchors a = 0.8922, b = 1.0.
- Technical failures (`Payment_Status == FAILED` and `Failure_Reason == TECHNICAL`, i.e. network or bank downtime) are
  excluded from the denominator, because they are not the customer's doing.
- Insufficient-funds failures still count. A failure plus a successful retry is one failure and one success; a retry
  that fails again is a second failure.
- This relies on `Failure_Reason` being reliably coded.
- Diagnostic: `pcs_incl_technical = |Ds| / |D|` is reported alongside.

### LSS: Liquidity Stability

The paper gives the idea ("stability of available balances and cash-flow behaviour"). The formula is ours. It averages
two sub-indicators, each a coefficient of variation (CV = sample standard deviation ÷ mean, `ddof=1`; lower means
steadier). A CV is undefined (NaN) if the mean is not positive.

- **L1** = CV of the 11 monthly mean balances (mean of B_d within each month). Anchors a = 0.1671, b = 0.6494.
- **L2** = CV of the 11 monthly inflows. Anchors a = 0.0218, b = 0.6214.
  - Monthly inflow = successful and unsuccessful `CREDIT` rows summed by amount, plus *hidden inflows*.
  - A hidden inflow is a row where the balance change from the previous row, minus the row's own signed successful UPI
    amount, exceeds ₹1. Salary or cash deposits that never appear as UPI credits show up this way.
  - The first row of a customer's history has no previous balance, so its unexplained flow is NaN and is ignored.
- `LSS = 0.5 × [ (1 − n(L1)) + (1 − n(L2)) ]`
- If a CV is undefined, that sub-indicator scores 0 and the customer is flagged (`no_inflow` for L2,
  `no_positive_balance` for L1).

### MDS: Merchant Diversity

- **Pool:** successful P2M, Bill Payment and Recharge debits. P2P is excluded because its recipients are people, not merchants.
- **Formula:** normalised Shannon entropy over the 11 categories (Food, Grocery, Fuel, Entertainment, Shopping,
  Healthcare, Education, Transport, Other, Utilities, Telecom):
  `MDS = − Σ p_c · ln(p_c) / ln(11)`, with zero-count categories skipped.
  This is 0 when all spend is in one category and 1 when it is spread evenly.
- It is used as-is (0–1) and **not** stretched, because the reference cohort barely varies (p5–p95 spread 0.077,
  below the 0.10 cut-off).
- Diagnostics only, not scored: `n_unique_merchants` and `effective_merchants` (`exp` of the entropy over counterparties,
  which depends on volume).

### FRS: Financial Resilience

The paper gives one line (ability to maintain stability and meet obligations). Everything else is ours.

**F1, obligation continuity:** of the recurring bills we detect, what share of billing cycles were paid?

Detection (`features.detect_obligations`) runs on **all debit attempts**, failed ones included, so a bill that keeps
failing is still detected. For each counterparty:

1. It must be in Entertainment, Telecom or Utilities, **or** be a P2P payee.
2. Attempts less than or equal to 12 whole days apart form one billing cycle (this groups retries with the original attempt).
3. It needs at least 6 cycles, with cycle starts in at least 6 distinct calendar months.
4. The whole-day gaps between consecutive cycle starts must have a median of 24–35 days and a sample standard deviation under 8.
5. If every attempt is P2P, the amount CV must be under 0.05, which is how rent is recognised.

A cycle is **paid** if any attempt in it succeeded. `F1 = paid cycles ÷ all cycles`, summed over all obligations.
F1 is used as-is (reference spread 0.054 is below 0.10).

**F2, buffer adequacy:** how often was the balance healthy?

- A day is "low balance" if B_d is below 25% of the customer's mean monthly successful outflow
  (`sum of successful debit amounts ÷ 11`).
- `F2 = 1 − (share of days in the window that were low balance)`. Anchors a = 0.7313, b = 1.0.

**Combined:** `FRS = 0.5 × [ F1 + n(F2) ]`.
If no obligations are found, `FRS = n(F2)` alone and the customer is flagged `no_obligations`.

### Composite

Components are combined on the 0–1 scale, then multiplied by 100. Each component's weighted contribution is output as
`contrib_<COMPONENT>` (in UCRII points, so the five sum to UCRII).

`UCRII_excl_TFS = Σ contributions except TFS ÷ 0.75`, a diagnostic that removes the volume-driven term and reweights to 100.

---

## 6. Normalisation (how raw numbers become 0–1)

The paper's min-max uses each indicator's minimum and maximum. We use a more outlier-proof version:

- **Anchors** are the 5th and 95th percentiles of a **frozen reference cohort** (`np.percentile`, linear interpolation).
  Values beyond them are clipped to 0 or 1.
- **Native indicators:** a bounded 0–1 indicator whose reference spread (p95 − p5) is under 0.10 is used unchanged
  (clipped to 0–1). Stretching it would turn near-constant values into fake variation. This applies to MDS and F1.
- `fit_anchors` refuses to run with fewer than 30 usable reference customers for any indicator, and raises an error if an
  unbounded indicator has p5 = p95 (degenerate distribution).
- NaN stays NaN through normalisation; it is never imputed.

**Frozen anchors used for scoring** (`anchors_reference_v1_1.json`, 1,000 customers, window 2024-02-01 to 2024-12-31):

| Indicator | Mode | a (p5) | b (p95) |
|---|---|---|---|
| `tfs_raw` | min-max | 25.1727 | 78.2773 |
| `pcs_raw` | min-max | 0.8922 | 1.0 |
| `L1_balance_cv` | min-max, lower is better | 0.1671 | 0.6494 |
| `L2_inflow_cv` | min-max, lower is better | 0.0218 | 0.6214 |
| `mds_raw` | native | 0.8378 | 0.9150 |
| `F1_obligation_continuity` | native (n = 986) | 0.9457 | 1.0 |
| `F2_buffer_adequacy` | min-max | 0.7313 | 1.0 |

For native indicators the two values are the reference p5 and p95, recorded for information only.

**Reference cohort:** 1,000 synthetic customers, generated with `python driver.py 250 --seed S` for S = 9001…9004 and
merged. It is kept separate from the validation cohorts (seed 2024).
These anchors are artefacts of synthetic data, **not Indian statistics**.

---

## 7. Flags, confidence and missing data

Scores are never silently imputed. The output carries flags, a `status` and a `confidence`:

| Flag | Meaning | Effect |
|---|---|---|
| `low_history` | Fewer than 150 successful debits in the window (about 14 a month) | Still scored; `confidence = low` |
| `no_obligations` | No recurring bills detected | FRS uses F2 only |
| `no_inflow` | No inflow at all | LSS sub-indicator L2 scores 0 |
| `no_positive_balance` | Monthly balance mean not positive | LSS sub-indicator L1 scores 0 |
| `no_merchant_payments` | No merchant-type debits | `status = insufficient_data`, no score |
| `no_debits` | No debit attempts, or no successful debits | `status = insufficient_data`, no score |

A customer is also `insufficient_data` if TFS, PCS or MDS cannot be computed.

| `status` | `confidence` | Meaning |
|---|---|---|
| `ok` | `normal` | Scored, enough history |
| `ok` | `low` | Scored, but `low_history`: treat with caution |
| `insufficient_data` | `none` | No score |

The 150-debit threshold separated the synthetic sparse cohort (median 102 successful debits) from the baseline cohort
(0% below 150). It is tied to the synthetic activity ranges, not an empirical cutoff.

---

## 8. Known limitations

Carry these into any write-up. Figures marked † come from the v1.0 validation runs and should be re-checked with v1.1.

1. **Volume drives the score.** UCRII correlated 0.68† with activity volume; with TFS removed that fell to −0.03†.
   TFS penalises thin-file customers by design (the paper requires it). `UCRII_excl_TFS` is output so you can see the effect.
2. **LSS penalises volatile income.** Gig customers scored about 13† points lower (LSS 34† vs 67–86†). That reflects
   income type, not evidence of worse repayment.
3. **MDS is nearly constant.** Its standard deviation is about 2† on a 0–100 scale and the split-half reliability of the
   raw indicator is only 0.22†. In practice it adds almost the same number to everyone, because the synthetic category
   mix comes from fixed age weights.
4. **FRS hits its ceiling.** The median is 100†. It mostly separates customers who had shocks.
5. **PCS depends on `Failure_Reason`.** Failures mislabelled as TECHNICAL would be excluded and inflate PCS.
6. **L2 is a lower bound.** It only sees inflow that reaches the balance, and balance data is bank-side rather than UPI-native.
7. **Weights and bands are hypothetical** (from the paper). Validation only checks direction and stability.
8. **The low-history threshold and all anchors are synthetic-cohort artefacts.**

---

## 9. Validation

The scripts in `validation/` stress-test the score on synthetic cohorts where the "right" answer is known by
construction. None of them change the score. They read `_latent/`, the manifest and the xlsx Profile sheet, which is
fine for validation but never done in scoring. They need `scipy` in addition to the packages in `requirements.txt`.

| Script | Question | Method |
|---|---|---|
| `run_scenarios.py` (4a) | Does the score move the right way when behaviour changes? | Scores 8 scenario cohorts with the frozen anchors and compares each customer with a reference cohort (paired Wilcoxon test, tolerance 0.5 points). Expected directions are fixed before looking at results. |
| `run_dynamics.py` (4b) | Is the score stable over time, and does it react to a shock? | Recomputes scores with an expanding window at month-ends July to December. Compares income-disruption customers with no-shock customers around the shock date. |
| `run_fairness.py` (4c) | Is the score biased, and how sensitive is it to the weights? | Breaks scores down by age, income type, state, bank and device (Kruskal-Wallis with ε²). Checks correlation with connectivity, activity volume and spend ratio. Perturbs the weights 2,000 times with Dirichlet(50·w) and drops each component in turn, measuring rank correlation and band changes. |
| `sparse_anchor_check.py` | Would anchors fitted with sparse customers included fix thin-file scoring? | Exploratory. Compares the frozen anchors with sparse-aware anchors on baseline and sparse cohorts. |

**Scenarios and expected directions** (−1 down, +1 up, 0 no change):

| Scenario | Reference | Expected |
|---|---|---|
| `high_spend_ratio` | baseline | PCS, FRS, LSS, UCRII −1 |
| `low_spend_ratio` | baseline | PCS, FRS, UCRII +1 |
| `income_disruption` | control_noshock | PCS, LSS, FRS, UCRII −1 |
| `expense_spike` | control_noshock | PCS, FRS, UCRII −1 |
| `irregular_recurring` | baseline | FRS, UCRII −1 |
| `activity_decay` | baseline | TFS, UCRII −1 |
| `technical_flaky` | baseline | PCS, UCRII 0 (technical failures are excluded from PCS) |
| `sparse` | baseline | TFS, UCRII −1 |

Pairing customers across cohorts is approximate (same customer id gives the same master-seed draws, but replay-time
random draws diverge), so `run_scenarios.py` also prints the noise floor: the standard deviation of paired differences
between `irregular_recurring` and `baseline`.

---

## 10. Running it

```bash
pip install numpy pandas pytest scipy     # scipy only for validation/

# run the tests
python -m pytest tests/test_ucrii.py

# score a cohort (one CSV per customer in <cohort_dir>)
python -m ucrii score <cohort_dir> --anchors ucrii/anchors_reference_v1_1.json --out scores.csv

# only if you need to refit the anchors from a new reference cohort
python -m ucrii fit-anchors <reference_cohort_dir> --out anchors.json

# validation (from ucrii_v1/)
python validation/run_scenarios.py <scenarios_root> <anchors.json> <out_dir>
python validation/run_dynamics.py <control_dir> <shock_dir> <anchors.json> <out_dir>
python validation/run_fairness.py <anchors.json> <out.csv> <cohort_dir> [<cohort_dir> ...]
python validation/sparse_anchor_check.py <anchors.json> <ref_baseline_dir> <ref_sparse_dir> <baseline_scn_dir> <sparse_scn_dir>
```

`score` accepts `--window-start` and `--window-end` to change the window. It prints summary statistics, band counts and
flag counts.

**Output columns** (one row per customer): `customer_id`; the seven raw indicators and the diagnostics
(`pcs_incl_technical`, `n_successful_debits`, `n_attempts`, `n_obligations`, `n_unique_merchants`,
`effective_merchants`, `mean_monthly_outflow`); the five components (0–100); `UCRII`; `band`; `status`; `confidence`;
`flags`; `UCRII_excl_TFS`; and `contrib_TFS`, `contrib_PCS`, `contrib_LSS`, `contrib_MDS`, `contrib_FRS`.
