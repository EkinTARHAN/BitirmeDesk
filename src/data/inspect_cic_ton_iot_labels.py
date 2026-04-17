from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Repo root bulunamadı. Script'i proje dizini içinde çalıştırdığından emin ol."
    )


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    csv_path = repo_root / "data" / "raw" / "cic_ton_iot" / "data" / "CIC-ToN-IoT.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV bulunamadı: {csv_path}\n"
            "Beklenen yapı: data/raw/cic_ton_iot/data/CIC-ToN-IoT.csv"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        repo_root
        / "reports"
        / "notes"
        / "cic_ton_iot_label_inspection"
        / f"run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CIC-ToN-IoT LABEL / CATEGORICAL INSPECTION BAŞLADI")
    print("=" * 100)
    print(f"CSV path   : {csv_path}")
    print(f"Output dir : {output_dir}")
    print("-" * 100)

    # Sadece kolon isimlerini almak için küçük örnek
    sample_df = pd.read_csv(csv_path, nrows=2000, low_memory=False)
    columns = sample_df.columns.tolist()

    print(f"Toplam kolon sayısı: {len(columns)}")
    print("-" * 100)
    print("Kolonlar:")
    for col in columns:
        print(f"  - {col}")

    # İsim olarak label/attack/class/category/type/target benzeri kolonları yakala
    keyword_candidates = [
        "label",
        "attack",
        "class",
        "category",
        "type",
        "target",
        "malicious",
        "benign",
    ]

    candidate_columns = []
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(keyword in col_lower for keyword in keyword_candidates):
            candidate_columns.append(col)

    # Sample üstünden object/categorical kolonları bul
    object_like_columns = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Tekrarları kaldır
    interesting_columns = []
    for col in candidate_columns + object_like_columns:
        if col not in interesting_columns:
            interesting_columns.append(col)

    if not interesting_columns:
        print("İlginç kolon bulunamadı. Sadece tüm kolon listesi export edilecek.")
        interesting_columns = []

    print("-" * 100)
    print("İncelenecek kolonlar:")
    if interesting_columns:
        for col in interesting_columns:
            print(f"  - {col}")
    else:
        print("  (yok)")

    inspection_results: dict[str, Any] = {
        "csv_path": str(csv_path),
        "all_columns": columns,
        "candidate_columns_by_name": candidate_columns,
        "object_like_columns_from_sample": object_like_columns,
        "inspected_columns": {},
    }

    # Full dataset'ten sadece ilginç kolonları oku
    if interesting_columns:
        full_df = pd.read_csv(csv_path, usecols=interesting_columns, low_memory=False)

        for col in interesting_columns:
            series = full_df[col]

            value_counts = (
                series.astype(str)
                .fillna("NaN")
                .value_counts(dropna=False)
            )

            top_values = value_counts.head(100).to_dict()
            top_values = {str(k): int(v) for k, v in top_values.items()}

            inspection_results["inspected_columns"][col] = {
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "nunique_including_nan_as_string": int(series.astype(str).nunique(dropna=False)),
                "top_values": top_values,
            }

    # Kolon listesini export et
    pd.DataFrame({"column_name": columns}).to_csv(
        output_dir / "all_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # İncelenen kolonların özetini export et
    rows = []
    for col, info in inspection_results["inspected_columns"].items():
        rows.append(
            {
                "column_name": col,
                "dtype": info["dtype"],
                "non_null_count": info["non_null_count"],
                "null_count": info["null_count"],
                "nunique": info["nunique_including_nan_as_string"],
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "inspected_columns_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Her kolon için top value export
    top_value_rows = []
    for col, info in inspection_results["inspected_columns"].items():
        for value, count in info["top_values"].items():
            top_value_rows.append(
                {
                    "column_name": col,
                    "value": value,
                    "count": count,
                }
            )

    pd.DataFrame(top_value_rows).to_csv(
        output_dir / "inspected_columns_top_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(output_dir / "inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(inspection_results, f, indent=2, ensure_ascii=False)

    print("-" * 100)
    print("ÖNEMLİ ÖZET")
    print("-" * 100)

    if inspection_results["inspected_columns"]:
        for col, info in inspection_results["inspected_columns"].items():
            print(f"[{col}]")
            print(f"  dtype      : {info['dtype']}")
            print(f"  non-null   : {info['non_null_count']:,}")
            print(f"  null       : {info['null_count']:,}")
            print(f"  nunique    : {info['nunique_including_nan_as_string']:,}")
            print("  top values :")
            for value, count in list(info["top_values"].items())[:20]:
                print(f"    {value}: {count:,}")
            print("-" * 100)
    else:
        print("İncelenecek ek kolon bulunamadı.")

    print(f"JSON summary : {output_dir / 'inspection_summary.json'}")
    print("=" * 100)
    print("CIC-ToN-IoT LABEL / CATEGORICAL INSPECTION TAMAMLANDI")
    print("=" * 100)


if __name__ == "__main__":
    main()