from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Repo root not found. Please run this script inside the project folder.")


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, low_memory=False)

    raise ValueError(f"Unsupported file type: {input_path.suffix}")


def summarize_split(df: pd.DataFrame, name: str) -> Dict:
    source_counts = (
        df["source_file"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        if "source_file" in df.columns
        else {}
    )

    attack_counts = (
        df["attack_category"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        if "attack_category" in df.columns
        else {}
    )

    binary_counts = (
        df["binary_label"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        if "binary_label" in df.columns
        else {}
    )

    return {
        "split_name": name,
        "rows": int(len(df)),
        "binary_label_distribution": {str(k): int(v) for k, v in binary_counts.items()},
        "source_file_distribution": {str(k): int(v) for k, v in source_counts.items()},
        "attack_category_distribution": {str(k): int(v) for k, v in attack_counts.items()},
    }


def save_split(df: pd.DataFrame, out_dir: Path, split_name: str, also_save_csv: bool) -> None:
    parquet_path = out_dir / f"{split_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    if also_save_csv:
        csv_path = out_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for CIC-IDS2017 binary dataset.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to prepared CIC binary dataset (.parquet or .csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save split files.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save split summary reports.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
        help="Train ratio. Default: 0.70",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation ratio. Default: 0.15",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test ratio. Default: 0.15",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--also-save-csv",
        action="store_true",
        help="Also save split files as CSV.",
    )
    args = parser.parse_args()

    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total:.6f}"
        )

    repo_root = find_repo_root(Path.cwd())

    input_path = (
        Path(args.input_path)
        if args.input_path
        else repo_root / "data" / "processed" / "cic_ids2017" / "cic_ids2017_binary.parquet"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "data" / "processed" / "cic_ids2017" / "splits" / "binary"
    )
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else repo_root / "reports" / "tables" / "cic_ids2017"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)

    required_cols = {"source_file", "attack_category", "binary_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    before_rows = len(df)

    # Eksik label varsa temizle
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)

    after_rows = len(df)
    dropped_missing = before_rows - after_rows

    if df["binary_label"].nunique() < 2:
        raise ValueError("binary_label does not contain at least 2 classes after cleaning.")

    stratify_y = df["binary_label"]

    # Önce train ve temp
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_size,
        random_state=args.random_state,
        stratify=stratify_y,
    )

    # Temp içinden val/test
    temp_ratio = args.val_size + args.test_size
    val_ratio_within_temp = args.val_size / temp_ratio

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_within_temp,
        random_state=args.random_state,
        stratify=temp_df["binary_label"],
    )

    # Index reset
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Kaydet
    save_split(train_df, out_dir, "train", args.also_save_csv)
    save_split(val_df, out_dir, "val", args.also_save_csv)
    save_split(test_df, out_dir, "test", args.also_save_csv)

    split_summary = {
        "dataset_name": "cic_ids2017_binary",
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "report_dir": str(report_dir),
        "random_state": args.random_state,
        "requested_ratios": {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        "rows_before_drop_missing_binary_label": int(before_rows),
        "rows_after_drop_missing_binary_label": int(after_rows),
        "dropped_missing_binary_label_rows": int(dropped_missing),
        "train": summarize_split(train_df, "train"),
        "val": summarize_split(val_df, "val"),
        "test": summarize_split(test_df, "test"),
    }

    with open(report_dir / "cic_ids2017_binary_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "benign_0": int((train_df["binary_label"] == 0).sum()),
                "attack_1": int((train_df["binary_label"] == 1).sum()),
            },
            {
                "split": "val",
                "rows": len(val_df),
                "benign_0": int((val_df["binary_label"] == 0).sum()),
                "attack_1": int((val_df["binary_label"] == 1).sum()),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "benign_0": int((test_df["binary_label"] == 0).sum()),
                "attack_1": int((test_df["binary_label"] == 1).sum()),
            },
        ]
    ).to_csv(report_dir / "cic_ids2017_binary_split_overview.csv", index=False)

    print("\n[OK] CIC-IDS2017 binary split oluşturuldu.")
    print(f"[OK] Train rows: {len(train_df)}")
    print(f"[OK] Val rows:   {len(val_df)}")
    print(f"[OK] Test rows:  {len(test_df)}")
    print(f"[OK] Split dir:  {out_dir}")
    print(f"[OK] Summary:    {report_dir / 'cic_ids2017_binary_split_summary.json'}")
    print(f"[OK] Overview:   {report_dir / 'cic_ids2017_binary_split_overview.csv'}")


if __name__ == "__main__":
    main()