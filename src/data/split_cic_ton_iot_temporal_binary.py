from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def standardize_attack_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def parse_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series.astype(str).str.strip(),
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce",
        dayfirst=True,
    )

    if parsed.isna().sum() > 0:
        fallback = pd.to_datetime(
            series.astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
        parsed = parsed.fillna(fallback)

    return parsed


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


def distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def clean_features_with_train_median(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inf_before = {}

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        arr = split_df[feature_cols].to_numpy(dtype=float, copy=False)
        inf_before[split_name] = int(np.isinf(arr).sum())

    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = split_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    nan_after_inf_replace = {
        "train": int(train_df[feature_cols].isna().sum().sum()),
        "val": int(val_df[feature_cols].isna().sum().sum()),
        "test": int(test_df[feature_cols].isna().sum().sum()),
    }

    train_medians = train_df[feature_cols].median(numeric_only=True)

    train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)
    val_df[feature_cols] = val_df[feature_cols].fillna(train_medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)

    nan_after_fill = {
        "train": int(train_df[feature_cols].isna().sum().sum()),
        "val": int(val_df[feature_cols].isna().sum().sum()),
        "test": int(test_df[feature_cols].isna().sum().sum()),
    }

    inf_after = {}
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        arr = split_df[feature_cols].to_numpy(dtype=float, copy=False)
        inf_after[split_name] = int(np.isinf(arr).sum())

    cleaning_summary = {
        "inf_before_cleaning": inf_before,
        "nan_after_inf_replace": nan_after_inf_replace,
        "nan_after_train_median_fill": nan_after_fill,
        "inf_after_cleaning": inf_after,
        "imputation_strategy": "train_median_only",
    }

    if sum(nan_after_fill.values()) != 0:
        raise ValueError(f"Cleaning sonrası NaN kaldı: {nan_after_fill}")

    if sum(inf_after.values()) != 0:
        raise ValueError(f"Cleaning sonrası inf kaldı: {inf_after}")

    return train_df, val_df, test_df, cleaning_summary


def split_time_range(df: pd.DataFrame) -> dict[str, str]:
    return {
        "start": str(df["_timestamp_parsed"].min()),
        "end": str(df["_timestamp_parsed"].max()),
    }


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    raw_csv_path = repo_root / "data" / "raw" / "cic_ton_iot" / "data" / "CIC-ToN-IoT.csv"
    processed_root = repo_root / "data" / "processed" / "cic_ton_iot"
    latest_prep_dir = find_latest_prep_dir(processed_root)

    feature_columns_path = latest_prep_dir / "feature_columns.csv"
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"feature_columns.csv bulunamadı: {feature_columns_path}")

    feature_cols = pd.read_csv(feature_columns_path)["feature_name"].tolist()

    required_cols = ["Timestamp", "Label", "Attack"]
    usecols = feature_cols + required_cols

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_output_dir = processed_root / "splits" / f"temporal_binary_split_{timestamp}"
    report_output_dir = repo_root / "reports" / "notes" / "cic_ton_iot_temporal_split" / f"run_{timestamp}"

    split_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CIC-ToN-IoT TEMPORAL BINARY SPLIT BAŞLADI")
    print("=" * 100)
    print(f"Raw CSV path      : {raw_csv_path}")
    print(f"Latest prep dir   : {latest_prep_dir}")
    print(f"Feature count     : {len(feature_cols)}")
    print(f"Split output dir  : {split_output_dir}")
    print(f"Report output dir : {report_output_dir}")
    print("-" * 100)

    print("Ham veri okunuyor...")
    df = pd.read_csv(raw_csv_path, usecols=usecols, low_memory=False)

    print(f"Okunan satır sayısı : {len(df):,}")
    print(f"Okunan kolon sayısı : {len(df.columns)}")

    df["target_binary"] = pd.to_numeric(df["Label"], errors="coerce").astype("Int64")
    df["attack_name"] = standardize_attack_labels(df["Attack"])
    df["_timestamp_parsed"] = parse_timestamps(df["Timestamp"])

    timestamp_nan = int(df["_timestamp_parsed"].isna().sum())
    if timestamp_nan > 0:
        raise ValueError(f"Parse edilemeyen timestamp sayısı: {timestamp_nan}")

    invalid_binary = int(df["target_binary"].isna().sum())
    if invalid_binary > 0:
        raise ValueError(f"Geçersiz binary label sayısı: {invalid_binary}")

    unique_binary = sorted(df["target_binary"].dropna().unique().tolist())
    if unique_binary != [0, 1]:
        raise ValueError(f"Beklenmeyen binary değerleri: {unique_binary}")

    print("Timestamp sırasına göre sıralanıyor...")
    df = df.sort_values("_timestamp_parsed").reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_SIZE)
    n_val = int(n * VAL_SIZE)
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print("-" * 100)
    print("Temporal split satır sayıları:")
    print(f"  train: {len(train_df):,}")
    print(f"  val  : {len(val_df):,}")
    print(f"  test : {len(test_df):,}")

    print("-" * 100)
    print("Temporal zaman aralıkları:")
    print(f"  train: {split_time_range(train_df)}")
    print(f"  val  : {split_time_range(val_df)}")
    print(f"  test : {split_time_range(test_df)}")

    print("-" * 100)
    print("Feature cleaning başlıyor...")
    train_df, val_df, test_df, cleaning_summary = clean_features_with_train_median(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    output_cols = feature_cols + ["target_binary", "attack_name"]

    train_out = train_df[output_cols].copy()
    val_out = val_df[output_cols].copy()
    test_out = test_df[output_cols].copy()

    print("-" * 100)
    print("Split dosyaları kaydediliyor...")
    train_save = save_dataframe(train_out, split_output_dir / "binary_train")
    val_save = save_dataframe(val_out, split_output_dir / "binary_val")
    test_save = save_dataframe(test_out, split_output_dir / "binary_test")

    shutil.copy2(feature_columns_path, split_output_dir / "feature_columns.csv")

    summary = {
        "dataset_name": "CIC-ToN-IoT",
        "split_type": "temporal_binary_split",
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "raw_csv_path": str(raw_csv_path),
        "latest_prep_dir": str(latest_prep_dir),
        "split_output_dir": str(split_output_dir),
        "report_output_dir": str(report_output_dir),
        "feature_count": len(feature_cols),
        "row_counts": {
            "total": int(n),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "time_ranges": {
            "train": split_time_range(train_df),
            "val": split_time_range(val_df),
            "test": split_time_range(test_df),
        },
        "binary_distribution": {
            "total": distribution(df["target_binary"]),
            "train": distribution(train_df["target_binary"]),
            "val": distribution(val_df["target_binary"]),
            "test": distribution(test_df["target_binary"]),
        },
        "attack_distribution": {
            "total": distribution(df["attack_name"]),
            "train": distribution(train_df["attack_name"]),
            "val": distribution(val_df["attack_name"]),
            "test": distribution(test_df["attack_name"]),
        },
        "cleaning_summary": cleaning_summary,
        "outputs": {
            "train": train_save,
            "val": val_save,
            "test": test_save,
            "feature_columns": str(split_output_dir / "feature_columns.csv"),
        },
    }

    summary_path = report_output_dir / "temporal_binary_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    shutil.copy2(summary_path, split_output_dir / "split_summary.json")

    pd.DataFrame(
        [{"split": k, **v} for k, v in summary["time_ranges"].items()]
    ).to_csv(report_output_dir / "time_ranges.csv", index=False, encoding="utf-8-sig")

    for name, dist in summary["binary_distribution"].items():
        pd.DataFrame(
            [{"label": k, "count": v} for k, v in dist.items()]
        ).to_csv(report_output_dir / f"binary_distribution_{name}.csv", index=False, encoding="utf-8-sig")

    for name, dist in summary["attack_distribution"].items():
        pd.DataFrame(
            [{"attack_name": k, "count": v} for k, v in dist.items()]
        ).to_csv(report_output_dir / f"attack_distribution_{name}.csv", index=False, encoding="utf-8-sig")

    print("-" * 100)
    print("ÖNEMLİ ÖZET")
    print("-" * 100)

    print("Binary dağılım:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {summary['binary_distribution'][split_name]}")

    print("-" * 100)
    print("Attack dağılımı:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {summary['attack_distribution'][split_name]}")

    print("-" * 100)
    print("Cleaning summary:")
    print(json.dumps(cleaning_summary, indent=2, ensure_ascii=False))

    print("-" * 100)
    print(f"Summary JSON: {summary_path}")
    print("=" * 100)
    print("CIC-ToN-IoT TEMPORAL BINARY SPLIT TAMAMLANDI")
    print("=" * 100)


if __name__ == "__main__":
    main()