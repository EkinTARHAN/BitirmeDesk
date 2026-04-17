from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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

    raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CIC-IDS2017 source_file and label distributions.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to prepared CIC binary dataset.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save source distribution reports.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())

    input_path = (
        Path(args.input_path)
        if args.input_path
        else repo_root / "data" / "processed" / "cic_ids2017" / "cic_ids2017_binary.parquet"
    )
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else repo_root / "reports" / "tables" / "cic_ids2017"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)

    required_cols = {"source_file", "attack_category", "binary_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["binary_label"] = df["binary_label"].fillna(-1).astype(int)

    source_overview = (
        df.groupby("source_file")
        .agg(
            rows=("source_file", "size"),
            benign_0=("binary_label", lambda s: int((s == 0).sum())),
            attack_1=("binary_label", lambda s: int((s == 1).sum())),
            missing_label=("binary_label", lambda s: int((s == -1).sum())),
            unique_attack_categories=("attack_category", "nunique"),
        )
        .reset_index()
        .sort_values("source_file")
    )

    source_attack_matrix = (
        pd.crosstab(df["source_file"], df["attack_category"])
        .reset_index()
        .sort_values("source_file")
    )

    source_binary_matrix = (
        pd.crosstab(df["source_file"], df["binary_label"])
        .rename(columns={-1: "missing", 0: "benign_0", 1: "attack_1"})
        .reset_index()
        .sort_values("source_file")
    )

    attack_source_matrix = (
        pd.crosstab(df["attack_category"], df["source_file"])
        .reset_index()
        .sort_values("attack_category")
    )

    source_overview.to_csv(report_dir / "cic_ids2017_source_overview.csv", index=False)
    source_attack_matrix.to_csv(report_dir / "cic_ids2017_source_attack_matrix.csv", index=False)
    source_binary_matrix.to_csv(report_dir / "cic_ids2017_source_binary_matrix.csv", index=False)
    attack_source_matrix.to_csv(report_dir / "cic_ids2017_attack_source_matrix.csv", index=False)

    print("\n[OK] CIC source/label dağılım analizleri hazırlandı.")
    print(f"[OK] Source overview:       {report_dir / 'cic_ids2017_source_overview.csv'}")
    print(f"[OK] Source attack matrix:  {report_dir / 'cic_ids2017_source_attack_matrix.csv'}")
    print(f"[OK] Source binary matrix:  {report_dir / 'cic_ids2017_source_binary_matrix.csv'}")
    print(f"[OK] Attack source matrix:  {report_dir / 'cic_ids2017_attack_source_matrix.csv'}")


if __name__ == "__main__":
    main()