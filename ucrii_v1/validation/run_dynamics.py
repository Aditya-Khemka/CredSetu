"""Phase 4b: dynamics. Expanding-window month-end scores (window = 2024-02-01 .. as-of date) for as-of dates Jul..Dec.
(a) stability for customers with NO shock; (b) response of income-disruption customers around the shock start,
measured against the same calendar change in no-shock customers. Shock dates come from _latent/ (VALIDATION ONLY).
Usage: python validation/run_dynamics.py <control_dir> <shock_dir> <anchors.json> <out_dir>"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucrii.cli import customer_files                                   # noqa: E402
from ucrii.features import compute_raw_features, load_transactions     # noqa: E402
from ucrii.scoring import load_anchors, score_customer                 # noqa: E402

ASOF = [pd.Timestamp(d) for d in ('2024-07-31', '2024-08-31', '2024-09-30', '2024-10-31', '2024-11-30', '2024-12-31')]


def trajectories(cohort_dir, anchors):
    rows = []
    for p in customer_files(cohort_dir):
        df = load_transactions(p)
        for e in ASOF:
            f, fl = compute_raw_features(df, '2024-02-01', e)
            s = score_customer(f, fl, anchors)
            rows.append({'customer_id': p.stem, 'asof': e, **{k: s.get(k) for k in ('TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII', 'band', 'status')}})
    return pd.DataFrame(rows)


def main(control_dir, shock_dir, anchors_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    A = load_anchors(anchors_path)
    ctl, shk = trajectories(control_dir, A), trajectories(shock_dir, A)
    ctl.to_csv(os.path.join(out_dir, 'traj_control.csv'), index=False)
    shk.to_csv(os.path.join(out_dir, 'traj_income_disruption.csv'), index=False)
    return ctl, shk


if __name__ == '__main__':
    main(*sys.argv[1:5])
