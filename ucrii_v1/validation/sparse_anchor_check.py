"""Exploratory: do sparse-aware reference anchors fix thin-file scoring? Usage:
python validation/sparse_anchor_check.py <anchors.json> <ref_baseline_dir> <ref_sparse_dir> <baseline_scn_dir> <sparse_scn_dir>"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucrii.cli import raw_table                                   # noqa: E402
from ucrii.scoring import fit_anchors, load_anchors, score_table  # noqa: E402

A = load_anchors(sys.argv[1])
A2 = fit_anchors(pd.concat([raw_table(sys.argv[2])[0], raw_table(sys.argv[3])[0]]))
rb, fb = raw_table(sys.argv[4]); rs, fs = raw_table(sys.argv[5])
for name, AA in (('v1 anchors', A), ('sparse-aware anchors', A2)):
    sb, ss = score_table(rb, fb, AA), score_table(rs, fs, AA)
    print(f"{name}: baseline UCRII {sb.UCRII.mean():.1f} (TFS {sb.TFS.mean():.1f}) | sparse UCRII {ss.UCRII.mean():.1f}, "
          f"TFS mean {ss.TFS.mean():.1f}, share TFS==0 {(ss.TFS <= .01).mean():.2f}")
print('sparse successful debits p10/p50/p90:', rs.n_successful_debits.quantile([.1, .5, .9]).round(0).tolist())
