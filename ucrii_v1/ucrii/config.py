"""Constants for the UCRII feature/scoring pipeline.

Every constant is tagged with its provenance, so nothing engineering-defined is presented as paper methodology:
  [PAPER]  - taken from the UCRII preprint (Ranjan, SSRN 6973243, 21 Jun 2026). The paper labels its own numbers
             "purely hypothetical"; weights and bands are adopted as an unvalidated baseline.
  [ENG]    - engineering decision made for this prototype (the paper does not define it).
"""

# [PAPER] weights of the five dimensions (Section 9). Not learned, not empirically validated.
WEIGHTS = {'TFS': 0.25, 'PCS': 0.25, 'LSS': 0.20, 'MDS': 0.15, 'FRS': 0.15}

# [PAPER] risk bands (Table 3), expressed as lower bounds so fractional scores (79.6) are classified.
BANDS = [(80.0, 'Low risk'), (60.0, 'Moderate risk'), (40.0, 'High risk'), (float('-inf'), 'Very high risk')]

# [ENG] scoring window. January is dropped as burn-in because the synthetic opening balance is arbitrary
# (insufficient-funds rate in month 1 was ~2.5x other months). One score as of WINDOW_END.
WINDOW_START = '2024-02-01'
WINDOW_END = '2024-12-31'

# [ENG] a customer with fewer successful debits than this in the window is scored but flagged 'low_history'.
LOW_HISTORY_MIN_SUCCESSFUL_DEBITS = 60

# [ENG] FRS / F2: a day is "low balance" if end-of-day balance < this fraction of mean monthly successful outflow.
LOW_BALANCE_FRACTION = 0.25

# [ENG] LSS / L2: a balance change between consecutive rows beyond the row's own successful UPI amount counts as
# a hidden inflow if it exceeds this many rupees.
UNEXPLAINED_JUMP_MIN = 1.0

# [ENG] FRS / F1: recurring-obligation inference from history alone (no isRecurring field exists).
CYCLE_GAP_DAYS = 12                 # attempts closer than this belong to the same billing cycle (retries)
OBLIGATION_MIN_MONTHS = 6           # cycles must fall in at least this many distinct months
OBLIGATION_MEDIAN_GAP_DAYS = (24, 35)
OBLIGATION_GAP_SD_MAX = 8.0
RENT_AMOUNT_CV_MAX = 0.05           # P2P obligations (rent) must have near-constant amounts
OBLIGATION_CATEGORIES = ('Entertainment', 'Telecom', 'Utilities')

# [ENG] MDS: pool = merchant-type debits (P2P excluded: recipients are people, category 'P2P').
MERCHANT_TRANSACTION_TYPES = ('P2M', 'Bill Payment', 'Recharge')
CATEGORIES = ['Food', 'Grocery', 'Fuel', 'Entertainment', 'Shopping', 'Healthcare', 'Education', 'Transport',
              'Other', 'Utilities', 'Telecom']

# [ENG] normalization: robust variant of the paper's min-max (Section 9), anchors frozen from a reference cohort.
ANCHOR_PERCENTILES = (5, 95)
# [ENG] guard: a natively 0-1 indicator whose reference spread (p95 - p5) is below this is NOT stretched, because
# min-max would manufacture variation out of near-constant values (observed for MDS and F1).
NO_STRETCH_MIN_SPREAD = 0.10

REQUIRED_COLUMNS = ['Timestamp', 'Direction', 'Transaction_Type', 'Merchant_Category', 'Counterparty_Name',
                    'Amount', 'Payment_Status', 'Failure_Reason', 'Balance_After']
