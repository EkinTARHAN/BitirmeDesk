from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Repo root bulunamadı.")


def find_latest_dir(root: Path, pattern: str) -> Path:
    dirs = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"Klasör bulunamadı: {root / pattern}")
    return dirs[-1]


def format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    return f"{float(value):.{digits}f}"


def load_cic_ton_iot_latest_package(repo_root: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = repo_root / "reports" / "final" / "cic_ton_iot"
    latest_pkg = find_latest_dir(root, "package_*")

    main_summary = pd.read_csv(latest_pkg / "cic_ton_iot_main_summary.csv")
    seen_unseen = pd.read_csv(latest_pkg / "cic_ton_iot_seen_unseen_summary.csv")
    best_attack = pd.read_csv(latest_pkg / "cic_ton_iot_best_val_f1_attack_family.csv")

    return latest_pkg, main_summary, seen_unseen, best_attack


def build_master_binary_table(cic_ton_main: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "dataset": "UNSW-NB15",
            "benchmark_scope": "main_binary_final",
            "split_type": "official_binary_selection",
            "model": "XGBoost Tuned v2",
            "selection_method": "target_fpr_val_requested_0.005",
            "selected_threshold": 0.94,
            "val_pr_auc": 0.997249,
            "val_roc_auc": 0.994208,
            "test_pr_auc": 0.989072,
            "test_roc_auc": 0.985146,
            "test_f1": 0.932448,
            "test_fpr": 0.017946,
            "note": "Official main binary model.",
        },
        {
            "dataset": "UNSW-NB15",
            "benchmark_scope": "main_binary_backup",
            "split_type": "official_binary_backup",
            "model": "LightGBM Main",
            "selection_method": "selected_operating_point",
            "selected_threshold": 0.950258,
            "val_pr_auc": None,
            "val_roc_auc": None,
            "test_pr_auc": 0.988060,
            "test_roc_auc": 0.983764,
            "test_f1": 0.932044,
            "test_fpr": 0.029486,
            "note": "Official backup binary model.",
        },
        {
            "dataset": "CIC-IDS2017",
            "benchmark_scope": "binary_baseline",
            "split_type": "row_wise_stratified",
            "model": "LightGBM baseline",
            "selection_method": "best_val_f1",
            "selected_threshold": 0.40,
            "val_pr_auc": 0.999889,
            "val_roc_auc": 0.999968,
            "test_pr_auc": 0.999915,
            "test_roc_auc": 0.999979,
            "test_f1": 0.998203,
            "test_fpr": 0.000680,
            "note": "Optimistic baseline later judged too easy.",
        },
        {
            "dataset": "CIC-IDS2017",
            "benchmark_scope": "binary_generalization_check",
            "split_type": "grouped_split",
            "model": "LightGBM",
            "selection_method": "best_val_f1",
            "selected_threshold": None,
            "val_pr_auc": 0.000219,
            "val_roc_auc": 0.715913,
            "test_pr_auc": 0.878776,
            "test_roc_auc": 0.966141,
            "test_f1": 0.771796,
            "test_fpr": 0.002309,
            "note": "Grouped benchmark exposed split sensitivity.",
        },
        {
            "dataset": "CIC-IDS2017",
            "benchmark_scope": "binary_generalization_check",
            "split_type": "file_based_val_test",
            "model": "LightGBM",
            "selection_method": "best_val_f1",
            "selected_threshold": None,
            "val_pr_auc": None,
            "val_roc_auc": None,
            "test_pr_auc": 0.468943,
            "test_roc_auc": 0.834459,
            "test_f1": 0.129533,
            "test_fpr": None,
            "note": "More honest cross-file benchmark.",
        },
    ]

    for _, row in cic_ton_main.iterrows():
        rows.append(
            {
                "dataset": row["benchmark_name"],
                "benchmark_scope": "binary_benchmark",
                "split_type": row["split_type"],
                "model": "LightGBM",
                "selection_method": row["selection_method"],
                "selected_threshold": row["selected_threshold"],
                "val_pr_auc": row["val_pr_auc"],
                "val_roc_auc": row["val_roc_auc"],
                "test_pr_auc": row["test_pr_auc"],
                "test_roc_auc": row["test_roc_auc"],
                "test_f1": row["test_f1"],
                "test_fpr": row["test_fpr"],
                "note": row["note"],
            }
        )

    return pd.DataFrame(rows)


def build_master_multiclass_table() -> pd.DataFrame:
    rows = [
        {
            "dataset": "UNSW-NB15",
            "benchmark_scope": "attack_only_multiclass_final",
            "model": "Random Forest",
            "role": "main",
            "macro_f1": 0.539542,
            "weighted_f1": 0.790918,
            "accuracy": None,
            "note": "Official main multiclass model.",
        },
        {
            "dataset": "UNSW-NB15",
            "benchmark_scope": "attack_only_multiclass_final",
            "model": "LightGBM",
            "role": "backup",
            "macro_f1": 0.531506,
            "weighted_f1": 0.786532,
            "accuracy": None,
            "note": "Official backup multiclass model.",
        },
        {
            "dataset": "UNSW-NB15",
            "benchmark_scope": "attack_only_multiclass_reference",
            "model": "XGBoost",
            "role": "reference",
            "macro_f1": 0.528704,
            "weighted_f1": 0.785278,
            "accuracy": 0.795972,
            "note": "Reference multiclass result kept for comparison.",
        },
    ]
    return pd.DataFrame(rows)


def build_official_decisions_table() -> pd.DataFrame:
    rows = [
        {
            "project_component": "UNSW binary",
            "official_choice": "XGBoost Tuned v2",
            "status": "main",
            "note": "Official final binary model.",
        },
        {
            "project_component": "UNSW binary",
            "official_choice": "LightGBM Main",
            "status": "backup",
            "note": "Official binary backup model.",
        },
        {
            "project_component": "UNSW multiclass",
            "official_choice": "Random Forest",
            "status": "main",
            "note": "Official final attack-only multiclass model.",
        },
        {
            "project_component": "UNSW multiclass",
            "official_choice": "LightGBM",
            "status": "backup",
            "note": "Official multiclass backup model.",
        },
        {
            "project_component": "CIC-IDS2017 benchmark line",
            "official_choice": "LightGBM + file-based val/test + target-FPR",
            "status": "recommended",
            "note": "Working evaluation line after rejecting overly optimistic row-wise view.",
        },
        {
            "project_component": "CIC-ToN-IoT benchmark line",
            "official_choice": "LightGBM binary baseline + temporal seen/unseen analysis",
            "status": "packaged",
            "note": "Third independent benchmark packaged and ready for thesis use.",
        },
    ]
    return pd.DataFrame(rows)


def build_markdown_report(
    binary_df: pd.DataFrame,
    multiclass_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    cic_ton_seen_unseen: pd.DataFrame,
    cic_ton_best_attack: pd.DataFrame,
) -> str:
    unsw_main = binary_df[
        (binary_df["dataset"] == "UNSW-NB15") & (binary_df["benchmark_scope"] == "main_binary_final")
    ].iloc[0]

    cic17_file_based = binary_df[
        (binary_df["dataset"] == "CIC-IDS2017") & (binary_df["split_type"] == "file_based_val_test")
    ].iloc[0]

    cic_ton_temporal = binary_df[
        (binary_df["dataset"] == "CIC-ToN-IoT") & (binary_df["split_type"] == "temporal_binary")
    ].iloc[0]

    cic_ton_seen_unseen_best = cic_ton_seen_unseen[
        cic_ton_seen_unseen["selection_method"] == "best_val_f1"
    ].iloc[0]

    unseen_rows = cic_ton_best_attack[cic_ton_best_attack["seen_status"] == "unseen_in_train"].copy()
    unseen_rows = unseen_rows.sort_values("detection_rate_or_fpr", ascending=False)

    lines: list[str] = []

    lines.append("# Project Master Results Package")
    lines.append("")
    lines.append("## 1. Project structure")
    lines.append("")
    lines.append(
        "The project consists of an explainable hybrid IDS built primarily on UNSW-NB15, "
        "supported by two additional independent benchmarks: CIC-IDS2017 and CIC-ToN-IoT."
    )
    lines.append("")
    lines.append("## 2. Official UNSW decisions")
    lines.append("")
    lines.append(
        f"- Official binary main model: **{unsw_main['model']}** "
        f"(test PR-AUC = **{format_metric(unsw_main['test_pr_auc'])}**, "
        f"test ROC-AUC = **{format_metric(unsw_main['test_roc_auc'])}**, "
        f"test F1 = **{format_metric(unsw_main['test_f1'])}**, "
        f"test FPR = **{format_metric(unsw_main['test_fpr'])}**)."
    )
    lines.append("- Official binary backup model: **LightGBM Main**.")
    lines.append("- Official multiclass main model: **Random Forest**.")
    lines.append("- Official multiclass backup model: **LightGBM**.")
    lines.append("")
    lines.append("## 3. CIC-IDS2017 methodological takeaway")
    lines.append("")
    lines.append(
        "CIC-IDS2017 showed that row-wise results can be overly optimistic. The file-based benchmark "
        f"was much harder (test PR-AUC = **{format_metric(cic17_file_based['test_pr_auc'])}**, "
        f"test ROC-AUC = **{format_metric(cic17_file_based['test_roc_auc'])}**, "
        f"test F1 = **{format_metric(cic17_file_based['test_f1'])}**), supporting the thesis that "
        "split strategy strongly affects IDS conclusions."
    )
    lines.append("")
    lines.append("## 4. CIC-ToN-IoT benchmark takeaway")
    lines.append("")
    lines.append(
        f"- Temporal benchmark test PR-AUC: **{format_metric(cic_ton_temporal['test_pr_auc'])}**"
    )
    lines.append(
        f"- Temporal benchmark test ROC-AUC: **{format_metric(cic_ton_temporal['test_roc_auc'])}**"
    )
    lines.append(
        f"- Temporal benchmark test F1: **{format_metric(cic_ton_temporal['test_f1'])}**"
    )
    lines.append(
        f"- Temporal benchmark test FPR: **{format_metric(cic_ton_temporal['test_fpr'])}**"
    )
    lines.append(
        f"- Seen attack detection rate: **{format_metric(cic_ton_seen_unseen_best['seen_attack_detection_rate'])}**"
    )
    lines.append(
        f"- Unseen attack detection rate: **{format_metric(cic_ton_seen_unseen_best['unseen_attack_detection_rate'])}**"
    )
    lines.append("")
    lines.append(
        "This benchmark supports the same high-level story observed on CIC-IDS2017: ranking/AUC can "
        "remain strong while operational behavior becomes harder under temporal drift and unseen attacks."
    )
    lines.append("")
    lines.append("## 5. CIC-ToN-IoT unseen attack-family notes")
    lines.append("")
    for _, row in unseen_rows.iterrows():
        lines.append(
            f"- **{row['attack_name']}**: detection rate = "
            f"**{format_metric(row['detection_rate_or_fpr'])}** "
            f"(n = {int(row['row_count']):,})"
        )
    lines.append("")
    lines.append(
        "Unseen-family generalization is not uniform: some families such as ddos/dos remain highly "
        "detectable, whereas backdoor and ransomware are harder."
    )
    lines.append("")
    lines.append("## 6. Final thesis positioning")
    lines.append("")
    lines.append(
        "The final thesis can therefore position the work as a hybrid IDS project that combines "
        "model selection, explainability, operating-point analysis, and benchmark methodology. "
        "UNSW-NB15 remains the main dataset, while CIC-IDS2017 and CIC-ToN-IoT serve as complementary "
        "independent benchmarks that validate the methodological conclusions."
    )

    return "\n".join(lines)


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    cic_ton_pkg_dir, cic_ton_main, cic_ton_seen_unseen, cic_ton_best_attack = load_cic_ton_iot_latest_package(repo_root)

    binary_df = build_master_binary_table(cic_ton_main)
    multiclass_df = build_master_multiclass_table()
    decisions_df = build_official_decisions_table()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / "reports" / "final" / "project_master" / f"package_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_path = output_dir / "project_master_binary_summary.csv"
    multiclass_path = output_dir / "project_master_multiclass_summary.csv"
    decisions_path = output_dir / "project_official_decisions.csv"
    report_md_path = output_dir / "project_master_report.md"

    binary_df.to_csv(binary_path, index=False, encoding="utf-8-sig")
    multiclass_df.to_csv(multiclass_path, index=False, encoding="utf-8-sig")
    decisions_df.to_csv(decisions_path, index=False, encoding="utf-8-sig")

    report_md = build_markdown_report(
        binary_df=binary_df,
        multiclass_df=multiclass_df,
        decisions_df=decisions_df,
        cic_ton_seen_unseen=cic_ton_seen_unseen,
        cic_ton_best_attack=cic_ton_best_attack,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    support_dir = output_dir / "supporting_files"
    support_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cic_ton_pkg_dir / "cic_ton_iot_main_summary.csv", support_dir / "cic_ton_iot_main_summary.csv")
    shutil.copy2(cic_ton_pkg_dir / "cic_ton_iot_seen_unseen_summary.csv", support_dir / "cic_ton_iot_seen_unseen_summary.csv")
    shutil.copy2(cic_ton_pkg_dir / "cic_ton_iot_best_val_f1_attack_family.csv", support_dir / "cic_ton_iot_best_val_f1_attack_family.csv")
    shutil.copy2(cic_ton_pkg_dir / "cic_ton_iot_final_report.md", support_dir / "cic_ton_iot_final_report.md")

    metadata = {
        "output_dir": str(output_dir),
        "source_cic_ton_iot_package": str(cic_ton_pkg_dir),
        "generated_files": {
            "binary_summary_csv": str(binary_path),
            "multiclass_summary_csv": str(multiclass_path),
            "official_decisions_csv": str(decisions_path),
            "project_master_report_md": str(report_md_path),
        },
    }

    with open(output_dir / "package_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("PROJECT MASTER PACKAGE OLUŞTURULDU")
    print("=" * 100)
    print(f"Output dir : {output_dir}")
    print("-" * 100)
    print("Üretilen ana dosyalar:")
    print(f"  - {binary_path}")
    print(f"  - {multiclass_path}")
    print(f"  - {decisions_path}")
    print(f"  - {report_md_path}")
    print("-" * 100)
    print("Binary summary preview:")
    print(binary_df.to_string(index=False))
    print("-" * 100)
    print("Official decisions preview:")
    print(decisions_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()