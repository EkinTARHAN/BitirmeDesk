from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_HARD_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]


META_COLS = {"source_file", "attack_category", "binary_label"}


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


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    numeric_cols = []

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    return numeric_cols


def pooled_std(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)

    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    return float(pooled)


def summarize_file(df: pd.DataFrame, file_name: str) -> dict:
    sub = df[df["source_file"] == file_name].copy()

    return {
        "source_file": file_name,
        "rows": int(len(sub)),
        "benign_0": int((sub["binary_label"] == 0).sum()),
        "attack_1": int((sub["binary_label"] == 1).sum()),
        "attack_ratio": float((sub["binary_label"] == 1).mean()) if len(sub) > 0 else None,
        "unique_attack_categories": int(sub["attack_category"].nunique()),
        "attack_categories": sorted(sub["attack_category"].astype(str).unique().tolist()),
    }


def compute_drift_table(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []

    for col in numeric_cols:
        train_s = pd.to_numeric(train_df[col], errors="coerce")
        test_s = pd.to_numeric(test_df[col], errors="coerce")

        train_non_null = train_s.dropna()
        test_non_null = test_s.dropna()

        train_mean = float(train_non_null.mean()) if len(train_non_null) else np.nan
        test_mean = float(test_non_null.mean()) if len(test_non_null) else np.nan

        train_median = float(train_non_null.median()) if len(train_non_null) else np.nan
        test_median = float(test_non_null.median()) if len(test_non_null) else np.nan

        train_std = float(train_non_null.std(ddof=1)) if len(train_non_null) > 1 else np.nan
        test_std = float(test_non_null.std(ddof=1)) if len(test_non_null) > 1 else np.nan

        pstd = pooled_std(train_s, test_s)
        if pstd is None or np.isnan(pstd) or pstd == 0:
            abs_smd = np.nan
        else:
            abs_smd = abs(train_mean - test_mean) / pstd

        rows.append(
            {
                "feature": col,
                "train_non_null": int(train_non_null.shape[0]),
                "test_non_null": int(test_non_null.shape[0]),
                "train_missing_ratio": float(train_s.isna().mean()),
                "test_missing_ratio": float(test_s.isna().mean()),
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_diff": float(test_mean - train_mean) if not (np.isnan(train_mean) or np.isnan(test_mean)) else np.nan,
                "train_median": train_median,
                "test_median": test_median,
                "train_std": train_std,
                "test_std": test_std,
                "abs_smd": float(abs_smd) if not np.isnan(abs_smd) else np.nan,
            }
        )

    drift_df = pd.DataFrame(rows).sort_values("abs_smd", ascending=False, na_position="last").reset_index(drop=True)
    return drift_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze feature drift for hard CIC-IDS2017 source files.")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
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
        else repo_root / "reports" / "tables" / "cic_ids2017" / "hard_file_drift"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)

    numeric_cols = get_numeric_feature_columns(df)

    source_overview = []
    for source_file in sorted(df["source_file"].astype(str).unique().tolist()):
        source_overview.append(summarize_file(df, source_file))

    source_overview_df = pd.DataFrame(source_overview).sort_values("source_file").reset_index(drop=True)
    source_overview_df.to_csv(report_dir / "source_overview.csv", index=False)

    hard_summary_rows = []

    for hard_file in DEFAULT_HARD_FILES:
        hard_df = df[df["source_file"] == hard_file].copy().reset_index(drop=True)
        train_pool_df = df[df["source_file"] != hard_file].copy().reset_index(drop=True)

        if hard_df.empty:
            print(f"[WARN] Hard file not found or empty: {hard_file}")
            continue

        drift_df = compute_drift_table(train_pool_df, hard_df, numeric_cols)
        drift_df.to_csv(report_dir / f"{hard_file}__feature_drift.csv", index=False)

        top20_df = drift_df.head(20).copy()
        top20_df.to_csv(report_dir / f"{hard_file}__top20_feature_drift.csv", index=False)

        attack_dist_df = (
            hard_df["attack_category"]
            .value_counts(dropna=False)
            .rename_axis("attack_category")
            .reset_index(name="row_count")
        )
        attack_dist_df.to_csv(report_dir / f"{hard_file}__attack_distribution.csv", index=False)

        hard_summary_rows.append(
            {
                "source_file": hard_file,
                "rows": int(len(hard_df)),
                "benign_0": int((hard_df["binary_label"] == 0).sum()),
                "attack_1": int((hard_df["binary_label"] == 1).sum()),
                "attack_ratio": float((hard_df["binary_label"] == 1).mean()),
                "top1_drift_feature": top20_df.iloc[0]["feature"] if len(top20_df) else None,
                "top1_abs_smd": float(top20_df.iloc[0]["abs_smd"]) if len(top20_df) and pd.notna(top20_df.iloc[0]["abs_smd"]) else None,
                "top5_drift_features": top20_df["feature"].head(5).tolist(),
            }
        )

    hard_summary_df = pd.DataFrame(hard_summary_rows)
    hard_summary_df.to_csv(report_dir / "hard_files_summary.csv", index=False)

    summary = {
        "dataset": "CIC-IDS2017",
        "hard_files": DEFAULT_HARD_FILES,
        "numeric_feature_count": len(numeric_cols),
        "report_dir": str(report_dir),
    }

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] CIC hard file drift analizi tamamlandı.")
    print(f"[OK] Source overview: {report_dir / 'source_overview.csv'}")
    print(f"[OK] Hard summary:    {report_dir / 'hard_files_summary.csv'}")
    for hard_file in DEFAULT_HARD_FILES:
        print(f"[OK] Drift CSV:       {report_dir / f'{hard_file}__feature_drift.csv'}")
        print(f"[OK] Top20 Drift:    {report_dir / f'{hard_file}__top20_feature_drift.csv'}")
        print(f"[OK] Attack Dist:    {report_dir / f'{hard_file}__attack_distribution.csv'}")


if __name__ == "__main__":
    main()