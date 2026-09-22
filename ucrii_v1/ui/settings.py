"""Paths, resolved at call time so env overrides (and tests) take effect.
UCRII_GENERATOR_DIR (default ../Synthetic_UPI_Txn_Generator), UCRII_DATA_DIR (default <generator>/UPI_Txns),
UCRII_ANCHORS (default ucrii/anchors_reference_v1_1.json)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ucrii_v1/


def generator_dir():
    return Path(os.environ.get('UCRII_GENERATOR_DIR') or ROOT.parent / 'Synthetic_UPI_Txn_Generator')


def data_dir():
    return Path(os.environ.get('UCRII_DATA_DIR') or generator_dir() / 'UPI_Txns')


def anchors_path():
    return Path(os.environ.get('UCRII_ANCHORS') or ROOT / 'ucrii' / 'anchors_reference_v1_1.json')
