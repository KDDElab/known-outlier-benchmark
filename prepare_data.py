#!/usr/bin/env python3
"""Fetch the small public inputs and freeze reproducible query payloads."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["sampling_seed"])
CAP = int(CONFIG["max_anomalies_per_dataset"])
N_NORMAL_EXAMPLES = int(CONFIG["normal_reference_examples_per_case"])

ADBENCH_FILES = [
    "4_breastw.npz",
    "14_glass.npz",
    "15_Hepatitis.npz",
    "21_Lymphography.npz",
    "29_Pima.npz",
    "37_Stamps.npz",
    "39_vertebral.npz",
    "42_WBC.npz",
    "45_wine.npz",
]
ADBENCH_RAW = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/"
    "adbench/datasets/Classical/"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/63/lymphography.zip"

UCI_FEATURES = [
    ("lymphatics", {1: "normal", 2: "arched", 3: "deformed", 4: "displaced"}),
    ("block_of_afferent_channels", {1: "no", 2: "yes"}),
    ("block_of_lymphatic_channels", {1: "no", 2: "yes"}),
    ("block_of_lymphatic_sinuses", {1: "no", 2: "yes"}),
    ("bypass", {1: "no", 2: "yes"}),
    ("extravasates", {1: "no", 2: "yes"}),
    ("regeneration_of_lymphatic_channels", {1: "no", 2: "yes"}),
    ("early_uptake_in_lymphatic_channels", {1: "no", 2: "yes"}),
    ("lymph_nodes_diminished", {1: "0", 2: "1", 3: "2", 4: "3"}),
    ("lymph_nodes_enlarged", {1: "1", 2: "2", 3: "3", 4: "4"}),
    ("changes_in_lymphatics", {1: "bean", 2: "oval", 3: "round"}),
    ("defect_in_node", {1: "no", 2: "lacunar", 3: "lacunar_marginal", 4: "lacunar_central"}),
    ("changes_in_node", {1: "no", 2: "lacunar", 3: "lacunar_marginal", 4: "lacunar_central"}),
    ("changes_in_structure", {
        1: "no", 2: "grainy", 3: "drop_like", 4: "coarse",
        5: "diluted", 6: "reticular", 7: "stripped", 8: "faint",
    }),
    ("special_forms", {1: "no", 2: "chalices", 3: "vesicles"}),
    ("dislocation_of", {1: "no", 2: "yes"}),
    ("exclusion_of_no", {1: "no", 2: "yes"}),
    ("number_of_nodes", {
        1: "0-9", 2: "10-19", 3: "20-29", 4: "30-39",
        5: "40-49", 6: "50-59", 7: "60-69", 8: "70_or_more",
    }),
]
UCI_CLASS_NAMES = {
    1: "normal_find",
    2: "metastases",
    3: "malign_lymph",
    4: "fibrosis",
}
UCI_ANOMALY_CLASSES = {2, 3, 4}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, attempts: int = 6) -> bytes:
    if destination.exists():
        return destination.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ADBench-LLM-reference-materials/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            destination.write_bytes(data)
            return data
        except Exception as exc:  # network proxies can close an idle TLS connection
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Could not fetch {url}: {error}") from error


def safe_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, 6)


def build_summary(matrix: np.ndarray, rows: np.ndarray, uci: bool) -> dict:
    summary = {}
    for j in range(matrix.shape[1]):
        feature_id = f"F{j + 1:03d}"
        values = matrix[rows, j]
        if uci:
            counts = {}
            for value, count in zip(*np.unique(values.astype(int), return_counts=True)):
                counts[str(int(value))] = int(count)
            mode_code = min(
                (int(code) for code in counts),
                key=lambda code: (-counts[str(code)], code),
            )
            summary[feature_id] = {
                "mode_code": mode_code,
                "mode_description": UCI_FEATURES[j][1].get(mode_code, "unknown"),
                "category_counts": counts,
            }
        else:
            values = np.asarray(values, dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            summary[feature_id] = {
                "median": safe_float(median),
                "mad": safe_float(mad),
                "p05": safe_float(np.quantile(values, 0.05)),
                "p95": safe_float(np.quantile(values, 0.95)),
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                "n_unique": int(np.unique(values).size),
            }
    return summary


def feature_schema(matrix: np.ndarray, uci: bool) -> list[dict]:
    schema = []
    for j in range(matrix.shape[1]):
        fid = f"F{j + 1:03d}"
        if uci:
            name, mapping = UCI_FEATURES[j]
            schema.append({
                "feature_id": fid,
                "name": name,
                "type": "categorical numeric code",
                "value_map": {str(k): v for k, v in mapping.items()},
            })
        else:
            schema.append({
                "feature_id": fid,
                "name": "feature name not preserved in the public NPZ",
                "type": "numeric as stored",
            })
    return schema


def present_row(row: np.ndarray, uci: bool) -> dict:
    result = {}
    for j, value in enumerate(row):
        fid = f"F{j + 1:03d}"
        if uci:
            code = int(value)
            name, mapping = UCI_FEATURES[j]
            result[fid] = {
                "name": name,
                "code": code,
                "value": mapping.get(code, "unknown"),
            }
        else:
            result[fid] = safe_float(value)
    return result


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def sample_rows(indices: np.ndarray, size: int, seed: int) -> np.ndarray:
    if len(indices) <= size:
        return np.sort(indices.astype(int))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=size, replace=False).astype(int))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_dir = ROOT / "data"
    adbench_dir = data_dir / "adbench"
    uci_dir = data_dir / "uci"
    runs_dir = ROOT / "runs"
    manifest_dir = ROOT / "manifests"
    prompt_system = (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8").strip()

    local_cache = PROJECT / "a2p_oi" / "adbench_data"
    selected_set = set(ADBENCH_FILES)
    dataset_rows: list[dict] = []
    case_rows: list[dict] = []
    payload_rows: list[dict] = []
    case_counter = 1

    for filename in ADBENCH_FILES:
        destination = adbench_dir / filename
        source_url = ADBENCH_RAW + filename
        source_bytes = fetch(source_url, destination)
        with np.load(io.BytesIO(source_bytes), allow_pickle=False) as dataset:
            matrix = np.asarray(dataset["X"])
            labels = np.asarray(dataset["y"]).astype(int)
        if matrix.ndim != 2 or len(labels) != matrix.shape[0]:
            raise ValueError(f"Unexpected ADBench schema in {filename}")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"Expected binary ADBench y labels in {filename}")

        anomalies = np.flatnonzero(labels == 1)
        normals = np.flatnonzero(labels == 0)
        dataset_key = f"ADB_{filename[:-4]}"
        chosen = sample_rows(anomalies, CAP, SEED + stable_seed(dataset_key))
        schema = feature_schema(matrix, uci=False)
        summary = build_summary(matrix, normals, uci=False)
        dataset_rows.append({
            "dataset_key": dataset_key,
            "source_family": "ADBench Classical",
            "source_dataset": filename,
            "source_url": source_url,
            "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "anomaly_rows": int(len(anomalies)),
            "normal_rows": int(len(normals)),
            "selected_query_rows": int(len(chosen)),
            "label_definition": "official ADBench y: 1 anomaly, 0 normal",
            "feature_semantics": "feature names not preserved in NPZ; opaque IDs",
            "local_sha256": sha256_bytes(source_bytes),
            "included": "yes",
        })

        for row_idx in chosen:
            case_id = f"C{case_counter:04d}"
            case_counter += 1
            ref_idx = sample_rows(
                normals,
                N_NORMAL_EXAMPLES,
                SEED + stable_seed(case_id + dataset_key),
            )
            refs = [
                {
                    "row_id": f"R{i + 1:02d}",
                    "features": present_row(matrix[int(ref_row)], uci=False),
                }
                for i, ref_row in enumerate(ref_idx)
            ]
            user_payload = {
                "task": (
                    "The benchmark flags the target row as anomalous. "
                    "Compare it with the supplied normal reference rows."
                ),
                "feature_schema": schema,
                "normal_reference_population": int(len(normals)),
                "normal_reference_summary": summary,
                "normal_reference_examples": refs,
                "target_row": {
                    "row_id": "X_STAR",
                    "features": present_row(matrix[int(row_idx)], uci=False),
                },
                "response_requirements": (
                    "Return the JSON schema from the system instruction. "
                    "Use only feature IDs shown in the target row."
                ),
            }
            user_text = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
            payload = {
                "case_id": case_id,
                "dataset_key": dataset_key,
                "source_row_index": int(row_idx),
                "normal_reference_source_row_indices": [int(i) for i in ref_idx],
                "input_source": "ADBench Classical official NPZ",
                "system_prompt": prompt_system,
                "user_prompt": user_text,
                "prompt_sha256": sha256_bytes(
                    (prompt_system + "\n" + user_text).encode("utf-8")
                ),
            }
            payload_rows.append(payload)
            case_rows.append({
                "case_id": case_id,
                "dataset_key": dataset_key,
                "source_family": "ADBench Classical",
                "source_row_index": int(row_idx),
                "benchmark_label": 1,
                "n_normal_reference_pool": int(len(normals)),
                "n_normal_examples_in_prompt": int(len(ref_idx)),
                "normal_reference_source_row_indices": json.dumps([int(i) for i in ref_idx]),
                "prompt_sha256": payload["prompt_sha256"],
            })

    # UCI Lymphography keeps its original feature names and category codes.
    uci_archive = fetch(UCI_ZIP_URL, uci_dir / "lymphography.zip")
    uci_zip = zipfile.ZipFile(io.BytesIO(uci_archive))
    raw_data = uci_zip.read("lymphography.data")
    raw_names = uci_zip.read("lymphography.names")
    (uci_dir / "lymphography.data").write_bytes(raw_data)
    (uci_dir / "lymphography.names").write_bytes(raw_names)
    raw_rows = [
        [int(value) for value in line.split(",")]
        for line in raw_data.decode("utf-8").splitlines()
        if line.strip()
    ]
    raw = np.asarray(raw_rows, dtype=int)
    source_classes = raw[:, 0]
    matrix = raw[:, 1:]
    labels = np.isin(source_classes, list(UCI_ANOMALY_CLASSES)).astype(int)
    if matrix.shape != (148, 18) or int(labels.sum()) != 146:
        raise ValueError("UCI Lymphography source schema or anomaly mapping changed.")
    uci_columns = [name for name, _ in UCI_FEATURES]
    csv_path = uci_dir / "lymphography.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_code", *uci_columns])
        writer.writerows(raw.tolist())

    dataset_key = "UCI_LYMPHOGRAPHY"
    anomalies = np.flatnonzero(labels == 1)
    normals = np.flatnonzero(labels == 0)
    chosen = sample_rows(
        anomalies,
        CAP,
        SEED + stable_seed(dataset_key),
    )
    schema = feature_schema(matrix, uci=True)
    summary = build_summary(matrix, normals, uci=True)
    dataset_rows.append({
        "dataset_key": dataset_key,
        "source_family": "UCI",
        "source_dataset": "Lymphography (dataset 63)",
        "source_url": "https://archive.ics.uci.edu/dataset/63/lymphography",
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "anomaly_rows": int(len(anomalies)),
        "normal_rows": int(len(normals)),
        "selected_query_rows": int(len(chosen)),
        "label_definition": "source class 1 (normal find) is normal; disease classes 2-4 are treated as anomalies",
        "feature_semantics": "UCI field names and category codes",
        "local_sha256": sha256_bytes(uci_archive),
        "included": "supplementary; not pooled with ADBench view",
    })

    for row_idx in chosen:
        case_id = f"C{case_counter:04d}"
        case_counter += 1
        ref_idx = sample_rows(
            normals,
            N_NORMAL_EXAMPLES,
            SEED + stable_seed(case_id + dataset_key),
        )
        refs = [
            {
                "row_id": f"R{i + 1:02d}",
                "features": present_row(matrix[int(ref_row)], uci=True),
            }
            for i, ref_row in enumerate(ref_idx)
        ]
        user_payload = {
            "task": (
                "The benchmark flags the target row as anomalous. "
                "Compare it with the supplied normal reference rows."
            ),
            "feature_schema": schema,
            "normal_reference_population": int(len(normals)),
            "normal_reference_summary": summary,
            "normal_reference_examples": refs,
            "target_row": {
                "row_id": "X_STAR",
                "features": present_row(matrix[int(row_idx)], uci=True),
            },
            "response_requirements": (
                "Return the JSON schema from the system instruction. "
                "Use only feature IDs shown in the target row."
            ),
        }
        user_text = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        payload = {
            "case_id": case_id,
            "dataset_key": dataset_key,
            "source_row_index": int(row_idx),
            "source_class_code_hidden_from_models": int(source_classes[int(row_idx)]),
            "normal_reference_source_row_indices": [int(i) for i in ref_idx],
            "input_source": "UCI Lymphography 63",
            "system_prompt": prompt_system,
            "user_prompt": user_text,
            "prompt_sha256": sha256_bytes(
                (prompt_system + "\n" + user_text).encode("utf-8")
            ),
        }
        payload_rows.append(payload)
        case_rows.append({
            "case_id": case_id,
            "dataset_key": dataset_key,
            "source_family": "UCI",
            "source_row_index": int(row_idx),
            "benchmark_label": 1,
            "source_class_code_hidden_from_models": int(source_classes[int(row_idx)]),
            "n_normal_reference_pool": int(len(normals)),
            "n_normal_examples_in_prompt": int(len(ref_idx)),
            "normal_reference_source_row_indices": json.dumps([int(i) for i in ref_idx]),
            "prompt_sha256": payload["prompt_sha256"],
        })

    # Audit all ADBench files already present locally without adding large
    # datasets to this annotation package.
    for path in sorted(local_cache.glob("*.npz")):
        if path.name in selected_set:
            continue
        with np.load(path, allow_pickle=False) as dataset:
            shape = dataset["X"].shape
            labels_local = np.asarray(dataset["y"]).astype(int)
        included = "no"
        reason = "not in the selected small pilot"
        if shape[0] > 800 or shape[1] > 20:
            reason = "exceeds pilot threshold: rows > 800 or features > 20"
        dataset_rows.append({
            "dataset_key": f"ADB_{path.stem}",
            "source_family": "ADBench Classical (local audit)",
            "source_dataset": path.name,
            "source_url": ADBENCH_RAW + path.name,
            "rows": int(shape[0]),
            "features": int(shape[1]),
            "anomaly_rows": int(np.sum(labels_local == 1)),
            "normal_rows": int(np.sum(labels_local == 0)),
            "selected_query_rows": 0,
            "label_definition": "official ADBench y: 1 anomaly, 0 normal",
            "feature_semantics": "not in this annotation package",
            "local_sha256": sha256_file(path),
            "included": f"no; {reason}",
        })

    secom_path = PROJECT / "data" / "uci-secom.csv"
    if secom_path.exists():
        import pandas as pd
        secom = pd.read_csv(secom_path)
        label_col = secom.columns[-1]
        fail_count = int((secom[label_col] == 1).sum())
        dataset_rows.append({
            "dataset_key": "UCI_SECOM",
            "source_family": "UCI local audit",
            "source_dataset": "SECOM (dataset 179)",
            "source_url": "https://archive.ics.uci.edu/dataset/179/secom",
            "rows": int(len(secom)),
            "features": int(len(secom.columns) - 1),
            "anomaly_rows": fail_count,
            "normal_rows": int(len(secom) - fail_count),
            "selected_query_rows": 0,
            "label_definition": "Pass/Fail; Fail treated as anomaly only in the local audit",
            "feature_semantics": "590 sensor columns; names not available in local CSV",
            "local_sha256": sha256_file(secom_path),
            "included": "no; 590 predictors require a separate high-dimensional protocol",
        })

    write_csv(manifest_dir / "datasets.csv", dataset_rows)
    write_csv(manifest_dir / "cases.csv", case_rows)
    with (runs_dir / "input_payloads.jsonl").open("w", encoding="utf-8") as stream:
        for item in payload_rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    (ROOT / "data" / "uci" / "feature_schema.json").write_text(
        json.dumps(
            {
                "source": "UCI Lymphography dataset 63",
                "class_codes_hidden_from_models": UCI_CLASS_NAMES,
                "operational_anomaly_class_codes": sorted(UCI_ANOMALY_CLASSES),
                "operational_normal_class_codes": [1],
                "feature_schema": feature_schema(matrix, uci=True),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (ROOT / "manifests" / "input_provenance.json").write_text(
        json.dumps(
            {
                "protocol_id": CONFIG["protocol_id"],
                "created_date": CONFIG["created_date"],
                "adbench_source": "https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical",
                "adbench_files": [
                    {
                        "name": p.name,
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(adbench_dir.glob("*.npz"))
                ],
                "uci_lymphography": {
                    "url": UCI_ZIP_URL,
                    "sha256": sha256_bytes(uci_archive),
                    "bytes": len(uci_archive),
                    "license": "CC BY 4.0",
                    "doi": "10.24432/C54598",
                    "normal_class_codes": [1],
                    "anomaly_class_codes": sorted(UCI_ANOMALY_CLASSES),
                    "normal_reference_pool_rows": int(np.sum(source_classes == 1)),
                },
                "sample_seed": SEED,
                "adbench_query_cap_per_dataset": CAP,
                "normal_examples_per_query": N_NORMAL_EXAMPLES,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(dataset_rows)} dataset audit rows, {len(case_rows)} cases, "
        f"{len(payload_rows)} exact payloads."
    )
    print(f"ADBench files: {len(ADBENCH_FILES)}; UCI Lymphography cases: {len(chosen)}")


if __name__ == "__main__":
    main()
