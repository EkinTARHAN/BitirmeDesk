from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PRESETS = {
    "strict_v1": {
        "description": (
            "Harder file-based split. Entire source files are assigned to train/val/test. "
            "Ratios are not exact because CIC-IDS2017 has only 8 CSV files."
        ),
        "train_files": [
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ],
        "val_files": [
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        ],
        "test_files": [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ],
    }
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Repo root not found. Please run this script inside the project folder.")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def summarize_split(df: pd.DataFrame, split_name: str) -> Dict:
    binary_counts = df["binary_label"].value_counts(dropna=False).sort_index().to_dict()
    source_counts = df["source_file"].value_counts(dropna=False).sort_index().to_dict()
    attack_counts = df["attack_category"].value_counts(dropna=False).sort_index().to_dict()

    return {
        "split_name": split_name,
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


def validate_preset(df: pd.DataFrame, preset_name: str) -> Dict[str, List[str]]:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {sorted(PRESETS)}")

    preset = PRESETS[preset_name]
    available_files = set(df["source_file"].astype(str).unique().tolist())

    required_files = set(preset["train_files"] + preset["val_files"] + preset["test_files"])

    missing_files = sorted(required_files - available_files)
    if missing_files:
        raise ValueError(
            "Some preset files are missing from the dataset:\n"
            + "\n".join(f"- {name}" for name in missing_files)
        )

    overlap_train_val = set(preset["train_files"]) & set(preset["val_files"])
    overlap_train_test = set(preset["train_files"]) & set(preset["test_files"])
    overlap_val_test = set(preset["val_files"]) & set(preset["test_files"])

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError("Preset contains overlapping source files across splits.")

    unassigned_files = sorted(available_files - required_files)
    if unassigned_files:
        print("[WARN] Some dataset files are not assigned by the preset:")
        for name in unassigned_files:
            print(f"  - {name}")

    return preset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create harder CIC-IDS2017 binary splits by assigning entire source files to train/val/test."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Prepared CIC binary dataset path (.parquet or .csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for grouped split files.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Output directory for grouped split reports.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="strict_v1",
        help=f"Preset name. Available: {', '.join(sorted(PRESETS))}",
    )
    parser.add_argument(
        "--also-save-csv",
        action="store_true",
        help="Also save split files as CSV.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())

    input_path = (
        Path(args.input_path)
        if args.input_path
        else repo_root / "data" / "processed" / "cic_ids2017" / "cic_ids2017_binary.parquet"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "data" / "processed" / "cic_ids2017" / "splits" / "binary_grouped"
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
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)
    after_rows = len(df)

    preset = validate_preset(df, args.preset)

    train_df = df[df["source_file"].isin(preset["train_files"])].copy().reset_index(drop=True)
    val_df = df[df["source_file"].isin(preset["val_files"])].copy().reset_index(drop=True)
    test_df = df[df["source_file"].isin(preset["test_files"])].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("At least one split is empty. Preset assignment must be revised.")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df["binary_label"].nunique() < 2:
            print(
                f"[WARN] {split_name} split has fewer than 2 binary classes. "
                f"This may make evaluation unstable."
            )

    save_split(train_df, out_dir, "train", args.also_save_csv)
    save_split(val_df, out_dir, "val", args.also_save_csv)
    save_split(test_df, out_dir, "test", args.also_save_csv)

    assignment_rows = []
    for split_name, file_list in [
        ("train", preset["train_files"]),
        ("val", preset["val_files"]),
        ("test", preset["test_files"]),
    ]:
        for source_file in file_list:
            subset = df[df["source_file"] == source_file]
            assignment_rows.append(
                {
                    "preset": args.preset,
                    "split": split_name,
                    "source_file": source_file,
                    "rows": int(len(subset)),
                    "benign_0": int((subset["binary_label"] == 0).sum()),
                    "attack_1": int((subset["binary_label"] == 1).sum()),
                    "unique_attack_categories": int(subset["attack_category"].nunique()),
                }
            )

    assignment_df = pd.DataFrame(assignment_rows).sort_values(["split", "source_file"]).reset_index(drop=True)
    assignment_df.to_csv(report_dir / f"cic_ids2017_binary_grouped_{args.preset}_file_assignment.csv", index=False)

    overview_df = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "benign_0": int((train_df["binary_label"] == 0).sum()),
                "attack_1": int((train_df["binary_label"] == 1).sum()),
                "source_file_count": int(train_df["source_file"].nunique()),
            },
            {
                "split": "val",
                "rows": len(val_df),
                "benign_0": int((val_df["binary_label"] == 0).sum()),
                "attack_1": int((val_df["binary_label"] == 1).sum()),
                "source_file_count": int(val_df["source_file"].nunique()),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "benign_0": int((test_df["binary_label"] == 0).sum()),
                "attack_1": int((test_df["binary_label"] == 1).sum()),
                "source_file_count": int(test_df["source_file"].nunique()),
            },
        ]
    )
    overview_df.to_csv(report_dir / f"cic_ids2017_binary_grouped_{args.preset}_overview.csv", index=False)

    summary = {
        "dataset_name": "cic_ids2017_binary",
        "split_type": "grouped_by_source_file",
        "preset_name": args.preset,
        "preset_description": preset["description"],
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "rows_before_drop_missing_binary_label": int(before_rows),
        "rows_after_drop_missing_binary_label": int(after_rows),
        "dropped_missing_binary_label_rows": int(before_rows - after_rows),
        "train_files": preset["train_files"],
        "val_files": preset["val_files"],
        "test_files": preset["test_files"],
        "train": summarize_split(train_df, "train"),
        "val": summarize_split(val_df, "val"),
        "test": summarize_split(test_df, "test"),
    }

    summary_path = report_dir / f"cic_ids2017_binary_grouped_{args.preset}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] CIC-IDS2017 grouped binary split oluşturuldu.")
    print(f"[OK] Preset: {args.preset}")
    print(f"[OK] Train rows: {len(train_df)}")
    print(f"[OK] Val rows:   {len(val_df)}")
    print(f"[OK] Test rows:  {len(test_df)}")
    print(f"[OK] Split dir:  {out_dir}")
    print(f"[OK] Summary:    {summary_path}")
    print(
        f"[OK] Assignment: {report_dir / f'cic_ids2017_binary_grouped_{args.preset}_file_assignment.csv'}"
    )
    print(
        f"[OK] Overview:   {report_dir / f'cic_ids2017_binary_grouped_{args.preset}_overview.csv'}"
    )


if __name__ == "__main__":
    main()