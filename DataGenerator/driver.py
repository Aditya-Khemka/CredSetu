"""Driver: generate ~12 months of UPI history for many synthetic users, one CSV per user named by (made-up) PAN.

Run:  python driver.py [n_users]     ->  UPI_Txns/<PAN>.csv
"""
import os
import random
import string
import sys

import numpy as np

from upi_user_generator import generate_user_transactions

N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUTPUT_DIR = "UPI_Txns"
START_DATE, END_DATE = "2024-01-01", "2024-12-31"
MASTER_SEED = 2024

# Same age mix as the original generator's sender_age distribution
AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '56+']
AGE_P = [0.25, 0.35, 0.25, 0.10, 0.05]


def make_pan(rng, used):
    """Made-up PAN in the real format AAAPL1234C: 3 letters, 'P' (individual), surname initial, 4 digits, check letter."""
    while True:
        L = string.ascii_uppercase
        pan = (''.join(rng.choices(L, k=3)) + 'P' + rng.choice(L) +
               ''.join(rng.choices(string.digits, k=4)) + rng.choice(L))
        if pan not in used:
            used.add(pan)
            return pan


def main():
    rng = random.Random(MASTER_SEED)          # PANs, per-user seeds, activity levels
    used = set()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for _ in range(N_USERS):
        pan = make_pan(rng, used)
        age_group = rng.choices(AGE_GROUPS, weights=AGE_P)[0]
        # Users differ in how active they are: skewed around ~550 transactions/year, 200-1500
        n_records = int(np.clip(rng.lognormvariate(np.log(550), 0.35), 200, 1500))
        df = generate_user_transactions(age_group, n_records, START_DATE, END_DATE, seed=rng.randrange(2**31))
        df.to_csv(os.path.join(OUTPUT_DIR, f"{pan}.csv"), index=False)
        rows.append((pan, age_group, len(df), (df['Payment_Status'] == 'FAILED').mean()))

    print(f"Saved {len(rows)} users to {OUTPUT_DIR}/")
    for age in AGE_GROUPS:
        r = [x for x in rows if x[1] == age]
        if r:
            print(f"  {age:6s} {len(r):3d} users, avg {np.mean([x[2] for x in r]):.0f} txns, "
                  f"avg failed {np.mean([x[3] for x in r]):.1%}")
    print(f"Total transactions: {sum(x[2] for x in rows):,}; per-user range {min(x[2] for x in rows)}-{max(x[2] for x in rows)}")
    print("Sample files:", sorted(os.listdir(OUTPUT_DIR))[:3])


if __name__ == "__main__":
    main()
