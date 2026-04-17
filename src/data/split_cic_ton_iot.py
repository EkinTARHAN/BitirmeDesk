from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Repo root bulunamadı.")


def find_latest_prep_dir(processed_root: Path) -> Path:
    prep_dirs = sorted(
        [p for p in processed_root.glob("prep_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not prep_dirs:
        raise FileNotFoundError(f"Prep klasörü bulunamadı: {processed_root}")
    return prep_dirs[-1]


def find_dataset_file(prep_dir: Path, stem: str) -> Path:
    parquet_path = prep_dir / f"{stem}.parquet"
    csv_gz_path = prep_dir / f"{stem}.csv.gz"
    csv_path = prep_dir / f"{stem}.csv"

    for path in [parquet_path, csv_gz_path, csv_path]:
        if path.exists():
            return path

    raise FileNotFoundError(f"Dataset dosyası bulunamadı: {stem}")


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False, compression="gzip")
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Desteklenmeyen dosya türü: {path}")


def save_dataframe(df: pd.DataFrame, path_without_ext: Path) -> dict[str, Any]:
    parquet_path = path_without_ext.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return {
            "saved_as": "parquet",
            "path": str(parquet_path),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        }
    except Exception as e:
        csv_gz_path = path_without_ext.with_suffix(".csv.gz")
        df.to_csv(csv_gz_path, index=False, compression="gzip", encoding="utf-8-sig")
        return {
            "saved_as": "csv.gz",
            "path": str(csv_gz_path),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "parquet_error": str(e),
        }


def stratified_indices(labels: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    temp_labels = labels.iloc[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    return train_idx, val_idx, test_idx


def distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def split_and_save(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = stratified_indices(df[target_col])

    split_infos = {}

    for split_name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        split_df = df.iloc[idx].copy()
        save_info = save_dataframe(split_df, output_dir / f"{prefix}_{split_name}")
        save_info["target_distribution"] = distribution(split_df[target_col])
        split_infos[split_name] = save_info

    return {
        "target_column": target_col,
        "total_rows": int(len(df)),
        "total_distribution": distribution(df[target_col]),
        "splits": split_infos,
    }


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    processed_root = repo_root / "data" / "processed" / "cic_ton_iot"
    latest_prep_dir = find_latest_prep_dir(processed_root)

    binary_path = find_dataset_file(latest_prep_dir, "cic_ton_iot_binary")
    multiclass_path = find_dataset_file(
        latest_prep_dir,
        "cic_ton_iot_attack_only_multiclass",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_output_dir = processed_root / "splits" / f"stratified_split_{timestamp}"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    report_output_dir = (
        repo_root
        / "reports"
        / "notes"
        / "cic_ton_iot_split"
        / f"run_{timestamp}"
    )
    report_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CIC-ToN-IoT STRATIFIED SPLIT BAŞLADI")
    print("=" * 100)
    print(f"Repo root        : {repo_root}")
    print(f"Latest prep dir  : {latest_prep_dir}")
    print(f"Binary path      : {binary_path}")
    print(f"Multiclass path  : {multiclass_path}")
    print(f"Split output dir : {split_output_dir}")
    print(f"Report output dir: {report_output_dir}")
    print("-" * 100)

    feature_columns_path = latest_prep_dir / "feature_columns.csv"
    if feature_columns_path.exists():
        shutil.copy2(feature_columns_path, split_output_dir / "feature_columns.csv")

    print("Binary dataset okunuyor...")
    binary_df = read_dataset(binary_path)
    if "target_binary" not in binary_df.columns:
        raise ValueError("Binary dataset içinde target_binary kolonu yok.")

    print(f"Binary satır sayısı : {len(binary_df):,}")
    print("Binary split üretiliyor...")
    binary_summary = split_and_save(
        df=binary_df,
        target_col="target_binary",
        output_dir=split_output_dir,
        prefix="binary",
    )

    del binary_df

    print("-" * 100)
    print("Attack-only multiclass dataset okunuyor...")
    multiclass_df = read_dataset(multiclass_path)
    if "target_attack" not in multiclass_df.columns:
        raise ValueError("Multiclass dataset içinde target_attack kolonu yok.")

    print(f"Attack-only satır sayısı : {len(multiclass_df):,}")
    print("Attack-only multiclass split üretiliyor...")
    multiclass_summary = split_and_save(
        df=multiclass_df,
        target_col="target_attack",
        output_dir=split_output_dir,
        prefix="attack_only_multiclass",
    )

    del multiclass_df

    summary = {
        "dataset_name": "CIC-ToN-IoT",
        "split_type": "stratified_baseline_split",
        "random_state": RANDOM_STATE,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "latest_prep_dir": str(latest_prep_dir),
        "split_output_dir": str(split_output_dir),
        "report_output_dir": str(report_output_dir),
        "binary": binary_summary,
        "attack_only_multiclass": multiclass_summary,
    }

    summary_path = report_output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    shutil.copy2(summary_path, split_output_dir / "split_summary.json")

    print("-" * 100)
    print("ÖNEMLİ ÖZET")
    print("-" * 100)

    print("Binary split dağılımı:")
    for split_name, info in binary_summary["splits"].items():
        print(f"  {split_name}: {info['row_count']:,} satır")
        print(f"    distribution: {info['target_distribution']}")

    print("-" * 100)
    print("Attack-only multiclass split dağılımı:")
    for split_name, info in multiclass_summary["splits"].items():
        print(f"  {split_name}: {info['row_count']:,} satır")
        print(f"    distribution: {info['target_distribution']}")

    print("-" * 100)
    print(f"Split summary JSON: {summary_path}")
    print("=" * 100)
    print("CIC-ToN-IoT STRATIFIED SPLIT TAMAMLANDI")
    print("=" * 100)


if __name__ == "__main__":
    main()