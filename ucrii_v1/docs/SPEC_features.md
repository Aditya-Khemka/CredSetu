# UCRII: how the score works (V1)

UCRII (UPI Credit Risk Intelligence Index) turns a customer's UPI transaction history into one number from 0 to 100.
A higher number means the customer's UPI behaviour looks more reliable.

It comes from Ranjan, *UPI Credit Risk Intelligence Index (UCRII)*, SSRN 6973243 (21 Jun 2026). That paper is a
conceptual preprint and calls its own tables "purely hypothetical". It names five dimensions but gives no formulas,
so **every formula on this page is our own engineering choice**.

> **What the score is not.** It is an explainable reliability indicator, not a probability of default. There are no
> repayment outcomes in the data, so we make no claim that it predicts anything.

Status: implemented in `ucrii/`, with 30 unit tests.V

---

## 1. Quick summary

The score is a weighted average of five components, each scored 0–100:

| Component | Weight | Plain-English question | Formula in one line |
|---|---|---|---|
| **TFS** Transaction Frequency | 0.25 | Does the customer use UPI regularly? | Average successful payments per month |
| **PCS** Payment Consistency | 0.25 | Do their payments go through? | Successful debits ÷ all debit attempts |
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

The weights and bands come from the paper and are **unvalidated**. Nothing here tunes or learns them.

---

## 2. What comes from where

Keeping these four layers separate matters. Blurring them is the easiest way to overclaim.

| Layer | What it covers |
|---|---|
| **Paper** | The five dimension names and one-line descriptions, the weights, the risk bands, and the min-max normalisation idea. |
| **Synthetic-data assumptions** | Everything inside the data generator: income types, spending ratios, shocks, failure rates, age mixes. These are made up, not real statistics. |
| **Engineering decisions** | Every formula, window and threshold in section 4. |
| **Extensions** | Diagnostics only (`pcs_excl_technical`, `UCRII_excl_TFS`, merchant counts). They are reported but never enter the score. |

---

## 3. Inputs

**Input:** one transaction CSV per customer (generator v2 schema). The customer id is the file name.

**Required columns:** `Timestamp`, `Direction`, `Transaction_Type`, `Merchant_Category`, `Counterparty_Name`, `Amount`,
`Payment_Status`, `Failure_Reason`, `Balance_After`.

**Never read:** the `_latent/` folder, the xlsx `Profile` sheet, and the manifest (age group, state, bank, name). The
score must depend on transactions only.

**Scoring window:** 1 Feb 2024 to 31 Dec 2024 (11 months), giving one score as of 31 Dec.
January is skipped as burn-in: the synthetic opening balance is arbitrary, and January's insufficient-funds rate was
about 2.5× that of other months.

**Notation used below**

| Symbol | Meaning |
|---|---|
| D | debit attempts in the window |
| Ds | successful debits in the window |
| B_d | end-of-day `Balance_After` (last row of the day, carried forward on quiet days; history before the window only supplies the starting balance) |

---

## 4. The five components

Most components use a helper `n(x)` that maps a raw value onto 0–1:

```
n(x) = clip( (x − a) / (b − a), 0, 1 )
```

`a` and `b` are fixed "anchor" values (see section 5). When a **higher raw value is worse**, we use `1 − n(x)` instead.

### TFS: Transaction Frequency

- `x` = average number of successful debits per month over the 11 months.
- `TFS = n(x)`, anchors a = 25.17, b = 78.28.
- A failed attempt followed by a successful retry counts once, because only the success is in Ds.
- Credits are not counted. The paper describes frequency as something "performed by a borrower".

### PCS: Payment Consistency

- `x = |Ds| / |D|`, counted per attempt. `PCS = n(x)`, anchors a = 0.8815, b = 0.9969.
- A failure plus a successful retry is one failure and one success. A retry that fails again is a second failure.
- Technical (network) failures **are** counted, following the paper literally.
- The paper says "regular" payments but never defines it, so we do not model regularity here.

### LSS: Liquidity Stability

The paper gives the idea ("stability of available balances and cash-flow behaviour"). The formula is ours. It averages
two sub-indicators, each using the coefficient of variation (CV = standard deviation ÷ mean; lower means steadier).

- **L1** = CV of the 11 monthly average balances. Anchors a = 0.1671, b = 0.6494.
- **L2** = CV of the 11 monthly inflows. Anchors a = 0.0218, b = 0.6214.
  - Monthly inflow = UPI credits plus *hidden inflows*: any balance jump between consecutive rows, after removing the
    row's own successful UPI amount, that exceeds ₹1 (salary or cash deposits show up this way).
- `LSS = 0.5 × [ (1 − n(L1)) + (1 − n(L2)) ]`
- If a CV is undefined (no inflow, or no positive balance), that sub-indicator scores 0 and the customer is flagged.

### MDS: Merchant Diversity

- **Pool:** successful P2M, Bill Payment and Recharge debits. P2P is excluded because its recipients are people, not merchants.
- **Formula:** normalised Shannon entropy over the 11 categories:
  `MDS = − Σ p_c · ln(p_c) / ln(11)`
  This is 0 when all spend is in one category and 1 when it is spread perfectly evenly.
- It is used as-is (0–1) and **not** stretched, because the reference cohort barely varies (p5–p95 spread 0.077,
  below the 0.10 cut-off).
- Diagnostics only, not scored: unique merchant count and effective merchant count. The latter depends on volume.

### FRS: Financial Resilience

The paper gives one line (ability to maintain stability and meet obligations). Everything else is ours.

**F1, obligation continuity:** of the recurring bills we detect, what share of billing cycles were paid?

- A **recurring obligation** is a counterparty that:
  - is in Entertainment, Telecom or Utilities, **or** is a P2P payee;
  - is billed in at least 6 different months;
  - has a median gap between cycles of 24–35 days, with gap standard deviation under 8 days;
  - and, if P2P, has near-constant amounts (CV < 0.05), which is how we recognise rent.
- Attempts within 12 days of each other form one cycle. A cycle counts as paid if **any** attempt in it succeeded.
- `F1 = paid cycles ÷ all cycles`. It is used as-is (reference spread 0.054 is below 0.10).

**F2, buffer adequacy:** how often was the balance healthy?

- A day is "low balance" if `B_d` is below 25% of the customer's mean monthly successful outflow.
- `F2 = 1 − (share of days in the window that were low balance)`. Anchors a = 0.7313, b = 1.0.

**Combined:** `FRS = 0.5 × [ F1 + n(F2) ]`.
If no obligations are found, `FRS = n(F2)` alone and the customer is flagged `no_obligations`.

---

## 5. Normalisation (how raw numbers become 0–1)

The paper's min-max uses each indicator's minimum and maximum. We use a more outlier-proof version:

- **Anchors** are the 5th and 95th percentiles of a **frozen reference cohort**. Values beyond them are clipped to 0 or 1.
- **Native indicators:** any 0–1 indicator whose reference spread is under 0.10 is used unchanged. Stretching it
  would turn near-constant values into fake variation. This applies to MDS and F1.
- **Reference cohort:** 1,000 synthetic customers, from generator commit `af616bc` run as
  `python driver.py 250 --seed S` for S = 9001…9004, then merged. It is kept separate from the validation cohorts (seed 2024).
- **Anchor file:** `ucrii/anchors_reference_v1.json`.
- These anchors are artefacts of synthetic data, **not Indian statistics**.

---

## 6. Flags and missing data

Scores are never silently imputed. Instead the output carries flags:

| Flag | Meaning | Effect |
|---|---|---|
| `low_history` | Fewer than 60 successful debits in the window | Still scored |
| `no_obligations` | No recurring bills detected | FRS uses F2 only |
| `no_inflow` | No inflow at all | LSS sub-indicator L2 scores 0 |
| `no_positive_balance` | Balance never positive | LSS sub-indicator L1 scores 0 |
| `no_merchant_payments` | No merchant-type debits | Status `insufficient_data`, no score |
| `no_debits` | No debits at all | Status `insufficient_data`, no score |

---

## 7. Known limitations

Carry these into any write-up.

1. **Volume drives the score.** UCRII correlates 0.68 with activity volume; with TFS removed that falls to −0.03.
   TFS penalises thin-file customers by design (the paper requires it). `UCRII_excl_TFS` is output so you can see the
   effect.
2. **LSS penalises volatile income.** Gig customers score about 13 points lower (LSS 34 vs 67–86). That reflects
   income type, not evidence of worse repayment.
3. **MDS is nearly constant.** Its standard deviation is about 2 on a 0–100 scale and the split-half reliability of the
   raw indicator is only 0.22. In practice it adds almost the same number to everyone, because the synthetic category
   mix comes from fixed age weights.
4. **FRS hits its ceiling.** The median is 100. It mostly separates customers who had shocks.
5. **PCS counts technical failures**, which may proxy for network quality or region. `pcs_excl_technical` is provided
   as a fairness check.
6. **L2 is a lower bound.** It only sees inflow that reaches the balance, and balance data is bank-side rather than UPI-native.
7. **Weights and bands are hypothetical** (from the paper). Validation in Phase 4 only checks direction and stability.

---

## 8. Running it

```bash
pip install numpy pandas pytest

# run the tests
python -m pytest tests/test_ucrii.py

# score a cohort (one CSV per customer in <cohort_dir>)
python -m ucrii score <cohort_dir> --anchors ucrii/anchors_reference_v1.json --out scores.csv

# only if you need to refit the anchors from a new reference cohort
python -m ucrii fit-anchors <reference_cohort_dir> --out anchors.json
```

The output has one row per customer with: the five components (0–100), `UCRII`, `band`, `status`, `flags`, the raw
indicators, each component's weighted contribution (`contrib_*`), and the diagnostic `UCRII_excl_TFS`.
