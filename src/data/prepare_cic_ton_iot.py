from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Repo root bulunamadı. Script'i proje dizini içinde çalıştırdığından emin ol."
    )


def standardize_attack_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "benign": "benign",
        "normal": "benign",
        "xss": "xss",
        "password": "password",
        "injection": "injection",
        "scanning": "scanning",
        "backdoor": "backdoor",
        "ransomware": "ransomware",
        "mitm": "mitm",
        "ddos": "ddos",
        "dos": "dos",
    }

    return s.map(lambda x: mapping.get(x, x))


def save_dataframe_with_fallback(df: pd.DataFrame, base_path_without_ext: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "saved_as": None,
        "path": None,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }

    parquet_path = base_path_without_ext.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        result["saved_as"] = "parquet"
        result["path"] = str(parquet_path)
        return result
    except Exception as e:
        csv_gz_path = base_path_without_ext.with_suffix(".csv.gz")
        df.to_csv(csv_gz_path, index=False, compression="gzip", encoding="utf-8-sig")
        result["saved_as"] = "csv.gz"
        result["path"] = str(csv_gz_path)
        result["parquet_error"] = str(e)
        return result


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    raw_csv_path = repo_root / "data" / "raw" / "cic_ton_iot" / "data" / "CIC-ToN-IoT.csv"
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"Ana CSV bulunamadı: {raw_csv_path}\n"
            "Beklenen yapı: data/raw/cic_ton_iot/data/CIC-ToN-IoT.csv"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    processed_output_dir = (
        repo_root / "data" / "processed" / "cic_ton_iot" / f"prep_{timestamp}"
    )
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    report_output_dir = (
        repo_root / "reports" / "notes" / "cic_ton_iot_prep" / f"run_{timestamp}"
    )
    report_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CIC-ToN-IoT PREP BAŞLADI")
    print("=" * 100)
    print(f"Repo root            : {repo_root}")
    print(f"Raw CSV path         : {raw_csv_path}")
    print(f"Processed output dir : {processed_output_dir}")
    print(f"Report output dir    : {report_output_dir}")
    print("-" * 100)

    df = pd.read_csv(raw_csv_path, low_memory=False)

    print(f"Ham satır sayısı : {len(df):,}")
    print(f"Ham kolon sayısı : {len(df.columns)}")
    print(f"Ham bellek (MB)  : {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f}")
    print("-" * 100)

    required_cols = ["Label", "Attack"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Gerekli kolon bulunamadı: {col}")

    df["Attack"] = standardize_attack_labels(df["Attack"])
    df["target_binary"] = pd.to_numeric(df["Label"], errors="coerce").astype("Int64")

    invalid_binary_rows = int(df["target_binary"].isna().sum())
    if invalid_binary_rows > 0:
        raise ValueError(
            f"Binary target üretiminde NaN oluştu. Sorunlu satır sayısı: {invalid_binary_rows}"
        )

    unique_binary_values = sorted(df["target_binary"].dropna().unique().tolist())
    if unique_binary_values != [0, 1]:
        raise ValueError(
            f"Beklenmeyen binary target değerleri bulundu: {unique_binary_values}"
        )

    leakage_cols = [col for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp"] if col in df.columns]
    target_cols = ["Label", "Attack", "target_binary"]

    feature_cols = [col for col in df.columns if col not in leakage_cols + target_cols]

    non_numeric_feature_cols = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_feature_cols:
        raise ValueError(
            "Feature set içinde numeric olmayan kolon(lar) kaldı: "
            + ", ".join(non_numeric_feature_cols)
        )

    print("Çıkarılacak leakage kolonları:")
    for col in leakage_cols:
        print(f"  - {col}")

    print(f"Feature kolon sayısı : {len(feature_cols)}")
    print("-" * 100)

    numeric_feature_df = df[feature_cols]

    inf_counts_by_col: dict[str, int] = {}
    arr = numeric_feature_df.to_numpy(dtype=float, copy=False)
    inf_mask = np.isinf(arr)
    total_inf_before = int(inf_mask.sum())

    for idx, col in enumerate(feature_cols):
        inf_counts_by_col[col] = int(inf_mask[:, idx].sum())

    print(f"Toplam inf sayısı (cleaning öncesi): {total_inf_before:,}")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    nan_after_inf_replace = int(df[feature_cols].isna().sum().sum())
    print(f"Inf -> NaN sonrası toplam eksik numeric değer: {nan_after_inf_replace:,}")

    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)

    remaining_nan_after_fill = int(df[feature_cols].isna().sum().sum())
    if remaining_nan_after_fill != 0:
        raise ValueError(
            f"Median imputasyon sonrası hâlâ NaN kaldı: {remaining_nan_after_fill}"
        )

    arr_after = df[feature_cols].to_numpy(dtype=float, copy=False)
    total_inf_after = int(np.isinf(arr_after).sum())

    if total_inf_after != 0:
        raise ValueError(
            f"Cleaning sonrası hâlâ inf kaldı: {total_inf_after}"
        )

    print(f"Cleaning sonrası toplam inf sayısı: {total_inf_after:,}")
    print(f"Cleaning sonrası toplam NaN sayısı: {remaining_nan_after_fill:,}")
    print("-" * 100)

    binary_distribution = (
        df["target_binary"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    binary_distribution = {str(k): int(v) for k, v in binary_distribution.items()}

    attack_distribution_all = (
        df["Attack"]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .to_dict()
    )
    attack_distribution_all = {str(k): int(v) for k, v in attack_distribution_all.items()}

    attack_only_df = df.loc[df["Attack"] != "benign", feature_cols + ["Attack"]].copy()
    attack_only_df = attack_only_df.rename(columns={"Attack": "target_attack"})

    attack_only_distribution = (
        attack_only_df["target_attack"]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .to_dict()
    )
    attack_only_distribution = {
        str(k): int(v) for k, v in attack_only_distribution.items()
    }

    binary_df = df[feature_cols + ["target_binary", "Attack"]].copy()
    binary_df = binary_df.rename(columns={"Attack": "attack_name"})

    binary_save_info = save_dataframe_with_fallback(
        binary_df,
        processed_output_dir / "cic_ton_iot_binary",
    )

    attack_only_save_info = save_dataframe_with_fallback(
        attack_only_df,
        processed_output_dir / "cic_ton_iot_attack_only_multiclass",
    )

    pd.DataFrame({"feature_name": feature_cols}).to_csv(
        processed_output_dir / "feature_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{"column_name": k, "inf_count_before_cleaning": v} for k, v in inf_counts_by_col.items()]
    ).to_csv(
        report_output_dir / "inf_counts_by_feature_before_cleaning.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{"target_binary": k, "count": v} for k, v in binary_distribution.items()]
    ).to_csv(
        report_output_dir / "binary_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{"attack_name": k, "count": v} for k, v in attack_distribution_all.items()]
    ).to_csv(
        report_output_dir / "attack_distribution_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{"target_attack": k, "count": v} for k, v in attack_only_distribution.items()]
    ).to_csv(
        report_output_dir / "attack_only_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset_name": "CIC-ToN-IoT",
        "raw_csv_path": str(raw_csv_path),
        "processed_output_dir": str(processed_output_dir),
        "report_output_dir": str(report_output_dir),
        "raw_row_count": int(len(df)),
        "raw_column_count": int(len(df.columns)),
        "leakage_columns_removed": leakage_cols,
        "feature_column_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "binary_target_column": "target_binary",
        "multiclass_target_column": "target_attack",
        "all_attack_name_column_in_binary_output": "attack_name",
        "total_inf_before_cleaning": total_inf_before,
        "total_inf_after_cleaning": total_inf_after,
        "total_nan_after_cleaning": remaining_nan_after_fill,
        "binary_distribution": binary_distribution,
        "attack_distribution_all": attack_distribution_all,
        "attack_only_distribution": attack_only_distribution,
        "binary_output": binary_save_info,
        "attack_only_multiclass_output": attack_only_save_info,
    }

    with open(report_output_dir / "prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("ÖNEMLİ ÖZET")
    print("-" * 100)
    print(f"Binary output saved as           : {binary_save_info['saved_as']}")
    print(f"Binary output path               : {binary_save_info['path']}")
    print(f"Attack-only output saved as      : {attack_only_save_info['saved_as']}")
    print(f"Attack-only output path          : {attack_only_save_info['path']}")
    print(f"Feature kolon sayısı             : {len(feature_cols)}")
    print(f"Toplam inf (önce)                : {total_inf_before:,}")
    print(f"Toplam inf (sonra)               : {total_inf_after:,}")
    print(f"Toplam NaN (sonra)               : {remaining_nan_after_fill:,}")
    print("-" * 100)

    print("Binary dağılım:")
    for k, v in binary_distribution.items():
        print(f"  {k}: {v:,}")

    print("-" * 100)
    print("Attack-only multiclass dağılımı:")
    for k, v in attack_only_distribution.items():
        print(f"  {k}: {v:,}")

    print("-" * 100)
    print(f"Prep summary JSON : {report_output_dir / 'prep_summary.json'}")
    print("=" * 100)
    print("CIC-ToN-IoT PREP TAMAMLANDI")
    print("=" * 100)


if __name__ == "__main__":
    main()