# UCRII v1.1

Explainable alternative-data reliability indicator from UPI transaction history. See `docs/SPEC_features.md` for definitions,
provenance and limitations. Not a probability of default; weights and bands come from an unvalidated preprint.

## UI

From this folder:

    pip install -r requirements.txt
    streamlit run ui/app.py

Enter a PAN (format `ABCPD1234E`; synthetic only, never a real one). If `<UPI_Txns>/<PAN>.csv` exists it is scored as-is;
otherwise a synthetic history is generated deterministically from the PAN and saved in the generator's cohort layout.

Environment overrides: `UCRII_GENERATOR_DIR` (default `../Synthetic_UPI_Txn_Generator`), `UCRII_DATA_DIR`
(default `<generator>/UPI_Txns`), `UCRII_ANCHORS` (default `ucrii/anchors_reference_v1_1.json`).
