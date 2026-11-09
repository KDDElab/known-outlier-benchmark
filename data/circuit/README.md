# Harvard Dataverse digital-circuit dataset

Source: [Digital circuits dataset for anomaly detection + ground truth
explanations](https://doi.org/10.7910/DVN/W4FPPN), Harvard Dataverse,
version 3, CC0 1.0.

`harvard_dataset_10.7910_DVN_W4FPPN.zip` is the complete 58,847,433-byte
archive. Its SHA-256 is
`e1495d11e64bbff0a98edee9423f1025c54ec45c3a8b587c911528cbe9c20010`.
The archive passed ZIP CRC validation; all 150 archive member sizes match the
Dataverse file records. The archive expands to about 1.30 GB. Direct Dataverse
MD5 values match the 65 CSV and PDF members. The other 84 members are `.tab`
exports of uploaded CSVs, so their MD5 metadata is for the original uploaded
format and does not match the converted tab-separated bytes.

The four circuits are C17, 74182, 74283, and 74181. The package has 80 truth
tables (64 modified, labeled versions and 16 original versions), 65
ground-truth CSV files, four circuit-tree files, and a circuit-details PDF.
The expected 64 modified-table / ground-truth pairs are present. There is one
extra ground-truth CSV with no matching truth table:
`Ground_Truth/c17/c17_modified_z3_nand2_to_xor2_ground_truth_0noise.csv`.
The inventory records this discrepancy; do not score it until its pairing is
resolved.

## Leakage controls for a future evaluation

- Never provide files under `Ground_Truth/`, the `label` column, modified-gate
  identity, or generator metadata to an evaluated model.
- Keep candidate explanations restricted to circuit input columns (`i*`).
  If circuit outputs (`o*`) are given as observed context, preregister that
  choice and do not treat outputs as candidate explanatory features.
- The ground-truth CSVs contain output-specific explanation sets. For each
  anomalous row, evaluation must identify which output changed relative to the
  matching original truth-table row, then compare only against the applicable
  input-feature explanation set. Preserve multiple valid sets if the source
  defines them.
- This is local input-feature effect truth under a single-gate-negation
  circuit construction. It is not a general causal ground truth for business
  tables.

The archive is downloaded and inventoried, but it is not included in the
ADBench/UCI multi-LLM weak-reference run. See
`../../manifests/circuit_dataset_inventory.csv` and
`../../manifests/circuit_dataset_inventory.json` for file IDs, checksums,
sizes, and pairing status. The original Dataverse metadata is preserved in
`dataverse_metadata.json`.
