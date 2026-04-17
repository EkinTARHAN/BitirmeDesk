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


def load_feature_reference(feature_csv_path: Path) -> dict[str, Any]:
    if not feature_csv_path.exists():
        return {
            "feature_reference_found": False,
            "feature_reference_path": str(feature_csv_path),
            "feature_reference_columns": [],
            "feature_reference_row_count": 0,
            "feature_reference_names": [],
        }

    ref_df = pd.read_csv(feature_csv_path, low_memory=False)

    feature_name_candidates = [
        "Feature Name",
        "feature_name",
        "Feature",
        "feature",
        "Name",
        "name",
    ]

    feature_name_col = None
    for col in feature_name_candidates:
        if col in ref_df.columns:
            feature_name_col = col
            break

    if feature_name_col is not None:
        feature_names = (
            ref_df[feature_name_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .tolist()
        )
    else:
        feature_names = []

    return {
        "feature_reference_found": True,
        "feature_reference_path": str(feature_csv_path),
        "feature_reference_columns": ref_df.columns.tolist(),
        "feature_reference_row_count": int(len(ref_df)),
        "feature_reference_names": feature_names,
    }


def choose_label_columns(df: pd.DataFrame) -> dict[str, Any]:
    cols = df.columns.tolist()

    multiclass_candidates = [
        "Label",
        "label",
        "Attack",
        "attack",
        "Class",
        "class",
        "Category",
        "category",
    ]

    binary_candidates = [
        "BinaryLabel",
        "binary_label",
        "Binary Label",
        "binary",
        "is_attack",
        "IsAttack",
        "Target",
        "target",
    ]

    multiclass_col = None
    binary_col = None

    for col in multiclass_candidates:
        if col in cols:
            multiclass_col = col
            break

    for col in binary_candidates:
        if col in cols:
            binary_col = col
            break

    return {
        "multiclass_label_column": multiclass_col,
        "binary_label_column": binary_col,
    }


def build_binary_from_multiclass(series: pd.Series) -> pd.Series:
    benign_tokens = {
        "benign",
        "normal",
        "normality",
        "background",
        "legitimate",
        "non-attack",
        "nonattack",
        "0",
        "false",
    }

    s = series.astype(str).str.strip().str.lower()
    binary = (~s.isin(benign_tokens)).astype(int)
    return binary


def summarize_numeric(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts: dict[str, int] = {}

    if numeric_cols:
        arr = df[numeric_cols].to_numpy(dtype=float, copy=False)
        inf_mask = np.isinf(arr)
        for idx, col in enumerate(numeric_cols):
            inf_counts[col] = int(inf_mask[:, idx].sum())

    numeric_summary = {
        "numeric_column_count": len(numeric_cols),
        "numeric_columns": numeric_cols,
        "numeric_inf_counts_by_column": inf_counts,
        "total_inf_values": int(sum(inf_counts.values())),
    }
    return numeric_summary


def summarize_categorical(df: pd.DataFrame) -> dict[str, Any]:
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    top_uniques = {}

    for col in categorical_cols:
        try:
            top_uniques[col] = int(df[col].nunique(dropna=True))
        except Exception:
            top_uniques[col] = -1

    return {
        "categorical_column_count": len(categorical_cols),
        "categorical_columns": categorical_cols,
        "categorical_unique_counts": top_uniques,
    }


def compare_with_reference(
    df_columns: list[str],
    reference_names: list[str],
    label_cols: list[str],
) -> dict[str, Any]:
    cleaned_df_cols = [c for c in df_columns if c not in label_cols]

    df_set = set(cleaned_df_cols)
    ref_set = set(reference_names)

    if not reference_names:
        return {
            "feature_reference_comparison_available": False,
            "dataset_non_label_column_count": len(cleaned_df_cols),
            "dataset_non_label_columns": cleaned_df_cols,
            "reference_feature_count": 0,
            "common_feature_count": 0,
            "common_features": [],
            "dataset_only_feature_count": len(cleaned_df_cols),
            "dataset_only_features": cleaned_df_cols,
            "reference_only_feature_count": 0,
            "reference_only_features": [],
        }

    common = sorted(df_set.intersection(ref_set))
    dataset_only = sorted(df_set - ref_set)
    reference_only = sorted(ref_set - df_set)

    return {
        "feature_reference_comparison_available": True,
        "dataset_non_label_column_count": len(cleaned_df_cols),
        "dataset_non_label_columns": cleaned_df_cols,
        "reference_feature_count": len(reference_names),
        "common_feature_count": len(common),
        "common_features": common,
        "dataset_only_feature_count": len(dataset_only),
        "dataset_only_features": dataset_only,
        "reference_only_feature_count": len(reference_only),
        "reference_only_features": reference_only,
    }


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    raw_root = repo_root / "data" / "raw" / "cic_ton_iot" / "data"

    csv_path = raw_root / "CIC-ToN-IoT.csv"
    feature_ref_path = raw_root / "CICFlowMeter_Features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ana CSV bulunamadı: {csv_path}\n"
            "Beklenen yapı: data/raw/cic_ton_iot/data/CIC-ToN-IoT.csv"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / "reports" / "notes" / "cic_ton_iot_qa" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("CIC-ToN-IoT QA BAŞLADI")
    print("=" * 90)
    print(f"Repo root : {repo_root}")
    print(f"CSV path   : {csv_path}")
    print(f"Output dir : {output_dir}")
    print("-" * 90)

    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Satır sayısı   : {len(df):,}")
    print(f"Kolon sayısı   : {len(df.columns)}")
    print(f"Bellek (MB)    : {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f}")
    print("-" * 90)

    label_info = choose_label_columns(df)
    multiclass_col = label_info["multiclass_label_column"]
    binary_col = label_info["binary_label_column"]

    total_nan = int(df.isna().sum().sum())
    nan_by_col = df.isna().sum().sort_values(ascending=False)
    nan_by_col = {k: int(v) for k, v in nan_by_col.items() if int(v) > 0}

    duplicate_rows = int(df.duplicated().sum())

    numeric_summary = summarize_numeric(df)
    categorical_summary = summarize_categorical(df)

    feature_ref_info = load_feature_reference(feature_ref_path)

    label_cols = [c for c in [multiclass_col, binary_col] if c is not None]
    feature_compare = compare_with_reference(
        df_columns=df.columns.tolist(),
        reference_names=feature_ref_info["feature_reference_names"],
        label_cols=label_cols,
    )

    multiclass_distribution = {}
    binary_distribution = {}

    if multiclass_col is not None:
        multiclass_distribution = (
            df[multiclass_col]
            .astype(str)
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .to_dict()
        )
        multiclass_distribution = {str(k): int(v) for k, v in multiclass_distribution.items()}

        if binary_col is None:
            derived_binary = build_binary_from_multiclass(df[multiclass_col])
            binary_distribution = (
                derived_binary.value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )
            binary_distribution = {str(k): int(v) for k, v in binary_distribution.items()}
    elif binary_col is not None:
        binary_distribution = (
            df[binary_col]
            .astype(str)
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .to_dict()
        )
        binary_distribution = {str(k): int(v) for k, v in binary_distribution.items()}

    summary = {
        "dataset_name": "CIC-ToN-IoT",
        "csv_path": str(csv_path),
        "feature_reference_path": str(feature_ref_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024 ** 2)), 4),
        "label_detection": {
            "multiclass_label_column": multiclass_col,
            "binary_label_column": binary_col,
        },
        "missing_values": {
            "total_nan": total_nan,
            "nan_by_column": nan_by_col,
        },
        "duplicates": {
            "duplicate_row_count": duplicate_rows,
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "multiclass_distribution": multiclass_distribution,
        "binary_distribution": binary_distribution,
        "feature_reference_info": feature_ref_info,
        "feature_reference_comparison": feature_compare,
    }

    summary_json_path = output_dir / "qa_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    columns_csv_path = output_dir / "columns.csv"
    pd.DataFrame({"column_name": df.columns.tolist()}).to_csv(columns_csv_path, index=False, encoding="utf-8-sig")

    dtypes_csv_path = output_dir / "dtypes.csv"
    pd.DataFrame({
        "column_name": df.columns.tolist(),
        "dtype": [str(dt) for dt in df.dtypes.tolist()],
    }).to_csv(dtypes_csv_path, index=False, encoding="utf-8-sig")

    nan_csv_path = output_dir / "nan_by_column.csv"
    pd.DataFrame(
        [{"column_name": k, "nan_count": v} for k, v in nan_by_col.items()]
    ).to_csv(nan_csv_path, index=False, encoding="utf-8-sig")

    if multiclass_distribution:
        multiclass_csv_path = output_dir / "multiclass_distribution.csv"
        pd.DataFrame(
            [{"label": k, "count": v} for k, v in multiclass_distribution.items()]
        ).to_csv(multiclass_csv_path, index=False, encoding="utf-8-sig")

    if binary_distribution:
        binary_csv_path = output_dir / "binary_distribution.csv"
        pd.DataFrame(
            [{"label": k, "count": v} for k, v in binary_distribution.items()]
        ).to_csv(binary_csv_path, index=False, encoding="utf-8-sig")

    inf_csv_path = output_dir / "inf_by_numeric_column.csv"
    pd.DataFrame(
        [
            {"column_name": k, "inf_count": v}
            for k, v in numeric_summary["numeric_inf_counts_by_column"].items()
        ]
    ).to_csv(inf_csv_path, index=False, encoding="utf-8-sig")

    feature_compare_csv_path = output_dir / "feature_reference_comparison.csv"
    feature_compare_rows = []

    for col in feature_compare.get("common_features", []):
        feature_compare_rows.append({"feature_name": col, "status": "common"})
    for col in feature_compare.get("dataset_only_features", []):
        feature_compare_rows.append({"feature_name": col, "status": "dataset_only"})
    for col in feature_compare.get("reference_only_features", []):
        feature_compare_rows.append({"feature_name": col, "status": "reference_only"})

    pd.DataFrame(feature_compare_rows).to_csv(
        feature_compare_csv_path, index=False, encoding="utf-8-sig"
    )

    print("ÖNEMLİ ÖZET")
    print("-" * 90)
    print(f"Multiclass label column : {multiclass_col}")
    print(f"Binary label column     : {binary_col}")
    print(f"Toplam NaN              : {total_nan}")
    print(f"Toplam duplicate row    : {duplicate_rows}")
    print(f"Toplam numeric inf      : {numeric_summary['total_inf_values']}")
    print(f"Numeric kolon sayısı    : {numeric_summary['numeric_column_count']}")
    print(f"Kategorik kolon sayısı  : {categorical_summary['categorical_column_count']}")
    print(f"Summary JSON            : {summary_json_path}")

    if multiclass_distribution:
        print("-" * 90)
        print("Multiclass dağılımı:")
        for k, v in list(multiclass_distribution.items())[:20]:
            print(f"  {k}: {v:,}")

    if binary_distribution:
        print("-" * 90)
        print("Binary dağılımı:")
        for k, v in binary_distribution.items():
            print(f"  {k}: {v:,}")

    print("=" * 90)
    print("CIC-ToN-IoT QA TAMAMLANDI")
    print("=" * 90)


if __name__ == "__main__":
    main()