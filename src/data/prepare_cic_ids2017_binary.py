from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Repo root not found. Please run this script inside the project folder.")


def clean_text(value: str) -> str:
    value = str(value).replace("\ufeff", "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def clean_column_name(name: str) -> str:
    name = clean_text(name)
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace("-", "_")
    name = re.sub(r"[^\w\s\.]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._").lower()
    return name or "unnamed_column"


def make_unique(names: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique_names: List[str] = []

    for name in names:
        if name not in counts:
            counts[name] = 0
            unique_names.append(name)
            continue

        counts[name] += 1
        unique_names.append(f"{name}__dup_{counts[name]}")

    return unique_names


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "MISSING"

    text = clean_text(str(value))
    text = re.sub(r"\s+", " ", text)
    return text


def binary_from_label(label: str) -> int | None:
    if label == "MISSING":
        return None
    return 0 if label.upper() == "BENIGN" else 1


def read_and_standardize_csv(csv_path: Path) -> Tuple[pd.DataFrame, dict]:
    raw_df = pd.read_csv(csv_path, low_memory=False)

    original_columns = list(raw_df.columns)
    cleaned_columns = [clean_column_name(col) for col in original_columns]
    unique_columns = make_unique(cleaned_columns)

    df = raw_df.copy()
    df.columns = unique_columns

    label_candidates = [col for col in df.columns if col == "label" or col.endswith("_label")]
    if not label_candidates:
        raise ValueError(f"Label column not found in {csv_path.name}")

    label_col = label_candidates[0]

    inf_count = int(np.isinf(raw_df.select_dtypes(include=[np.number]).to_numpy()).sum()) if not raw_df.empty else 0
    df = df.replace([np.inf, -np.inf], np.nan)

    df["source_file"] = csv_path.name
    df["attack_category"] = df[label_col].map(normalize_label)
    df["binary_label"] = df["attack_category"].map(binary_from_label)

    feature_cols = [col for col in df.columns if col not in {label_col, "source_file", "attack_category", "binary_label"}]
    ordered_cols = ["source_file", "attack_category", "binary_label"] + feature_cols
    df = df[ordered_cols]

    meta = {
        "file_name": csv_path.name,
        "rows": int(len(df)),
        "original_column_count": int(len(original_columns)),
        "cleaned_column_count": int(len(df.columns)),
        "label_column_used": label_col,
        "duplicate_column_renames": int(len(cleaned_columns) - len(set(cleaned_columns))),
        "inf_values_replaced_with_nan": inf_count,
        "raw_label_counts": Counter(df["attack_category"].tolist()),
        "binary_label_counts": Counter(df["binary_label"].dropna().astype(int).tolist()),
        "missing_binary_label_rows": int(df["binary_label"].isna().sum()),
    }
    return df, meta


def build_summary(file_metas: List[dict], combined_df: pd.DataFrame, input_files: List[Path]) -> dict:
    raw_label_counts = Counter()

    for meta in file_metas:
        raw_label_counts.update(meta["raw_label_counts"])

    summary = {
        "dataset_name": "cic_ids2017_binary_benchmark",
        "input_dir_file_count": len(input_files),
        "input_files": [p.name for p in input_files],
        "total_rows": int(len(combined_df)),
        "total_columns": int(combined_df.shape[1]),
        "feature_column_count": int(combined_df.shape[1] - 3),
        "binary_label_distribution": {
            "benign_0": int((combined_df["binary_label"] == 0).sum()),
            "attack_1": int((combined_df["binary_label"] == 1).sum()),
            "missing": int(combined_df["binary_label"].isna().sum()),
        },
        "raw_label_distribution": dict(sorted(raw_label_counts.items(), key=lambda x: x[0])),
        "per_file": file_metas,
        "columns": combined_df.columns.tolist(),
        "source_files_seen_in_combined": sorted(combined_df["source_file"].astype(str).unique().tolist()),
    }
    return summary


def save_outputs(
    combined_df: pd.DataFrame,
    file_metas: List[dict],
    summary: dict,
    out_dir: Path,
    report_dir: Path,
    also_save_csv: bool,
) -> None:
    parquet_path = out_dir / "cic_ids2017_binary.parquet"
    csv_path = out_dir / "cic_ids2017_binary.csv"

    parquet_saved = False
    try:
        combined_df.to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception as exc:
        print(f"[WARN] Parquet save failed: {exc}")
        print("[WARN] Devam ediyorum; CSV çıktısı üretilecek.")

    if also_save_csv or not parquet_saved:
        combined_df.to_csv(csv_path, index=False)

    pd.DataFrame(file_metas).drop(columns=["raw_label_counts", "binary_label_counts"]).to_csv(
        report_dir / "cic_ids2017_binary_prep_file_summary.csv",
        index=False,
    )

    (
        combined_df["attack_category"]
        .value_counts(dropna=False)
        .rename_axis("attack_category")
        .reset_index(name="row_count")
        .to_csv(report_dir / "cic_ids2017_raw_label_distribution.csv", index=False)
    )

    (
        combined_df["binary_label"]
        .fillna(-1)
        .astype(int)
        .map({0: "benign_0", 1: "attack_1", -1: "missing"})
        .value_counts(dropna=False)
        .rename_axis("binary_label")
        .reset_index(name="row_count")
        .to_csv(report_dir / "cic_ids2017_binary_label_distribution.csv", index=False)
    )

    with open(report_dir / "cic_ids2017_binary_prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] CIC-IDS2017 binary benchmark hazırlandı.")
    if parquet_saved:
        print(f"[OK] Parquet: {parquet_path}")
    if also_save_csv or not parquet_saved:
        print(f"[OK] CSV: {csv_path}")
    print(f"[OK] Summary JSON: {report_dir / 'cic_ids2017_binary_prep_summary.json'}")
    print(f"[OK] File summary CSV: {report_dir / 'cic_ids2017_binary_prep_file_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIC-IDS2017 binary benchmark dataset.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory containing CIC-IDS2017 machine learning CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save processed CIC-IDS2017 outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save summary tables/reports.",
    )
    parser.add_argument(
        "--also-save-csv",
        action="store_true",
        help="Also save the combined dataset as CSV in addition to parquet.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())

    raw_dir = Path(args.raw_dir) if args.raw_dir else repo_root / "data" / "raw" / "cic_ids2017" / "machine_learning_csv"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / "processed" / "cic_ids2017"
    report_dir = Path(args.report_dir) if args.report_dir else repo_root / "reports" / "tables" / "cic_ids2017"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    frames: List[pd.DataFrame] = []
    file_metas: List[dict] = []

    for csv_path in csv_files:
        print(f"[INFO] Reading: {csv_path.name}")
        df, meta = read_and_standardize_csv(csv_path)
        frames.append(df)
        file_metas.append(meta)

    combined_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    meta_cols = ["source_file", "attack_category", "binary_label"]
    other_cols = [c for c in combined_df.columns if c not in meta_cols]
    combined_df = combined_df[meta_cols + other_cols]

    summary = build_summary(file_metas=file_metas, combined_df=combined_df, input_files=csv_files)

    save_outputs(
        combined_df=combined_df,
        file_metas=file_metas,
        summary=summary,
        out_dir=out_dir,
        report_dir=report_dir,
        also_save_csv=args.also_save_csv,
    )


if __name__ == "__main__":
    main()