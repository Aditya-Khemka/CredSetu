"""Command line:  python -m ucrii fit-anchors <cohort_dir> --out anchors.json
                 python -m ucrii score <cohort_dir> --anchors anchors.json --out scores.csv

A cohort dir holds one <ID>.csv per customer (the generator's output). ONLY top-level *.csv files are read, except
manifest.csv; sub-directories (notably _latent/) and .xlsx Profile sheets are never opened.
The customer id is the file stem. No age/state/bank/name information reaches the scoring."""
import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from .features import RAW_COLUMNS, DIAGNOSTIC_COLUMNS, compute_raw_features, load_transactions
from .scoring import fit_anchors, load_anchors, save_anchors, score_table


def customer_files(cohort_dir):
    return sorted(p for p in Path(cohort_dir).glob('*.csv') if p.name != 'manifest.csv')


def raw_table(cohort_dir, window_start=C.WINDOW_START, window_end=C.WINDOW_END):
    """(DataFrame of raw indicators indexed by customer_id, {customer_id: flags})"""
    rows, flags = {}, {}
    for p in customer_files(cohort_dir):
        feats, fl = compute_raw_features(load_transactions(p), window_start, window_end)
        rows[p.stem], flags[p.stem] = feats, fl
    df = pd.DataFrame.from_dict(rows, orient='index')[RAW_COLUMNS + DIAGNOSTIC_COLUMNS]
    df.index.name = 'customer_id'
    return df, flags


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ucrii', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fit-anchors', 'score'):
        sp = sub.add_parser(name)
        sp.add_argument('cohort_dir')
        sp.add_argument('--window-start', default=C.WINDOW_START)
        sp.add_argument('--window-end', default=C.WINDOW_END)
        sp.add_argument('--out', required=True)
        if name == 'score':
            sp.add_argument('--anchors', required=True)
    args = ap.parse_args(argv)

    raw, flags = raw_table(args.cohort_dir, args.window_start, args.window_end)
    if args.cmd == 'fit-anchors':
        anchors = fit_anchors(raw, meta={'source_dir': str(args.cohort_dir), 'n_customers': int(len(raw)),
                                         'window': [args.window_start, args.window_end]})
        save_anchors(anchors, args.out)
        print(f"Fitted anchors on {len(raw)} reference customers -> {args.out}")
        for ind, spec in anchors['anchors'].items():
            print(f"  {ind:26s} {spec['mode']:7s}" + (f" a={spec['a']:.4f} b={spec['b']:.4f}" if spec['mode'] == 'minmax'
                  else f" (spread {spec['reference_p_hi'] - spec['reference_p_lo']:.3f} < {C.NO_STRETCH_MIN_SPREAD}: native)"))
    else:
        out = score_table(raw, flags, load_anchors(args.anchors))
        out.to_csv(args.out, index=False)
        ok = out[out['status'] == 'ok']
        print(f"Scored {len(ok)}/{len(out)} customers -> {args.out}")
        print(ok[['TFS', 'PCS', 'LSS', 'MDS', 'FRS', 'UCRII']].describe().round(1).to_string())
        print(ok['band'].value_counts().to_string())
        print('flag counts:', pd.Series(';'.join(out['flags']).split(';')).replace('', pd.NA).dropna().value_counts().to_dict())


if __name__ == '__main__':
    main()
