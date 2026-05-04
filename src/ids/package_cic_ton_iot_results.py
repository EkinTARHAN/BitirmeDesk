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


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value)


def format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    return f"{float(value):.{digits}f}"


def build_main_summary(
    strat_summary: dict[str, Any],
    temporal_summary: dict[str, Any],
    seen_unseen_df: pd.DataFrame,
) -> pd.DataFrame:
    strat_ops = pd.DataFrame(strat_summary["operating_points"])
    temp_ops = pd.DataFrame(temporal_summary["operating_points"])

    strat_best = strat_ops[strat_ops["selection_method"] == "best_val_f1"].iloc[0]
    temp_best = temp_ops[temp_ops["selection_method"] == "best_val_f1"].iloc[0]

    temp_seen_unseen_best = seen_unseen_df[
        seen_unseen_df["selection_method"] == "best_val_f1"
    ].iloc[0]

    rows = [
        {
            "benchmark_name": "CIC-ToN-IoT",
            "split_type": "stratified_baseline",
            "selection_method": "best_val_f1",
            "selected_threshold": float(strat_best["selected_threshold"]),
            "val_pr_auc": float(strat_best["val_pr_auc"]),
            "val_roc_auc": float(strat_best["val_roc_auc"]),
            "val_f1": float(strat_best["val_f1"]),
            "val_fpr": float(strat_best["val_fpr"]),
            "test_pr_auc": float(strat_best["test_pr_auc"]),
            "test_roc_auc": float(strat_best["test_roc_auc"]),
            "test_f1": float(strat_best["test_f1"]),
            "test_fpr": float(strat_best["test_fpr"]),
            "seen_attack_detection_rate": None,
            "unseen_attack_detection_rate": None,
            "note": "Stratified row-wise baseline; optimistic reference benchmark.",
        },
        {
            "benchmark_name": "CIC-ToN-IoT",
            "split_type": "temporal_binary",
            "selection_method": "best_val_f1",
            "selected_threshold": float(temp_best["selected_threshold"]),
            "val_pr_auc": float(temp_best["val_pr_auc"]),
            "val_roc_auc": float(temp_best["val_roc_auc"]),
            "val_f1": float(temp_best["val_f1"]),
            "val_fpr": float(temp_best["val_fpr"]),
            "test_pr_auc": float(temp_best["test_pr_auc"]),
            "test_roc_auc": float(temp_best["test_roc_auc"]),
            "test_f1": float(temp_best["test_f1"]),
            "test_fpr": float(temp_best["test_fpr"]),
            "seen_attack_detection_rate": float(temp_seen_unseen_best["seen_attack_detection_rate"]),
            "unseen_attack_detection_rate": float(temp_seen_unseen_best["unseen_attack_detection_rate"]),
            "note": "More realistic temporal benchmark with partial unseen-attack stress.",
        },
    ]

    return pd.DataFrame(rows)


def build_target_fpr_summary(temporal_summary: dict[str, Any]) -> pd.DataFrame:
    temp_ops = pd.DataFrame(temporal_summary["operating_points"]).copy()

    keep_cols = [
        "selection_method",
        "target_fpr",
        "selected_threshold",
        "val_pr_auc",
        "val_roc_auc",
        "val_f1",
        "val_fpr",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_fpr",
    ]
    return temp_ops[keep_cols].copy()


def build_best_val_attack_family_table(attack_family_df: pd.DataFrame) -> pd.DataFrame:
    best_df = attack_family_df[
        attack_family_df["selection_method"] == "best_val_f1"
    ].copy()

    keep_cols = [
        "attack_name",
        "group_type",
        "seen_status",
        "row_count",
        "predicted_attack_count",
        "detection_rate_or_fpr",
        "mean_score",
        "median_score",
        "min_score",
        "max_score",
    ]

    return best_df[keep_cols].sort_values(
        by=["group_type", "seen_status", "attack_name"],
        ascending=[True, True, True],
    )


def build_seen_unseen_summary(seen_unseen_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "selection_method",
        "target_fpr",
        "threshold",
        "fpr",
        "recall",
        "precision",
        "f1",
        "benign_fp_rate",
        "seen_attack_detection_rate",
        "unseen_attack_detection_rate",
        "benign_count",
        "seen_attack_count",
        "unseen_attack_count",
    ]
    return seen_unseen_df[keep_cols].copy()


def build_markdown_report(
    qa_summary: dict[str, Any],
    prep_summary: dict[str, Any],
    strat_summary: dict[str, Any],
    temporal_model_summary: dict[str, Any],
    temporal_split_summary: dict[str, Any],
    main_summary_df: pd.DataFrame,
    seen_unseen_df: pd.DataFrame,
    best_attack_df: pd.DataFrame,
) -> str:
    main_strat = main_summary_df[main_summary_df["split_type"] == "stratified_baseline"].iloc[0]
    main_temp = main_summary_df[main_summary_df["split_type"] == "temporal_binary"].iloc[0]
    best_seen_unseen = seen_unseen_df[seen_unseen_df["selection_method"] == "best_val_f1"].iloc[0]

    unseen_rows = best_attack_df[best_attack_df["seen_status"] == "unseen_in_train"].copy()
    seen_rows = best_attack_df[best_attack_df["seen_status"] == "seen_in_train"].copy()

    unseen_rows = unseen_rows.sort_values("detection_rate_or_fpr", ascending=False)
    seen_rows = seen_rows.sort_values("detection_rate_or_fpr", ascending=False)

    lines: list[str] = []

    lines.append("# CIC-ToN-IoT Final Results Package")
    lines.append("")
    lines.append("## 1. Dataset snapshot")
    lines.append("")
    lines.append(f"- Row count: **{qa_summary['row_count']:,}**")
    lines.append(f"- Column count: **{qa_summary['column_count']}**")
    lines.append(f"- Feature count after prep: **{prep_summary['feature_column_count']}**")
    lines.append(f"- Total inf cleaned during prep: **{prep_summary['total_inf_before_cleaning']:,}**")
    lines.append(f"- Binary label column: **{qa_summary['label_detection']['multiclass_label_column']}** was used together with derived binary target.")
    lines.append("")
    lines.append("## 2. Binary benchmark summary")
    lines.append("")
    lines.append(
        f"- Stratified baseline test PR-AUC: **{format_metric(main_strat['test_pr_auc'])}**, "
        f"test ROC-AUC: **{format_metric(main_strat['test_roc_auc'])}**, "
        f"test F1: **{format_metric(main_strat['test_f1'])}**, "
        f"test FPR: **{format_metric(main_strat['test_fpr'])}**."
    )
    lines.append(
        f"- Temporal benchmark test PR-AUC: **{format_metric(main_temp['test_pr_auc'])}**, "
        f"test ROC-AUC: **{format_metric(main_temp['test_roc_auc'])}**, "
        f"test F1: **{format_metric(main_temp['test_f1'])}**, "
        f"test FPR: **{format_metric(main_temp['test_fpr'])}**."
    )
    lines.append("")
    lines.append("## 3. Main methodological interpretation")
    lines.append("")
    lines.append(
        "The stratified split produced near-perfect results and therefore should be interpreted "
        "as an optimistic baseline. The temporal split is more realistic because the data are "
        "ordered chronologically and the test period includes attack families that are not present "
        "in the training set."
    )
    lines.append("")
    lines.append(
        f"In the temporal benchmark, the model still preserved strong ranking performance "
        f"(test PR-AUC = **{format_metric(main_temp['test_pr_auc'])}**, "
        f"test ROC-AUC = **{format_metric(main_temp['test_roc_auc'])}**), "
        f"but the performance was lower than the stratified baseline, indicating temporal/domain shift."
    )
    lines.append("")
    lines.append("## 4. Seen vs unseen attack behavior")
    lines.append("")
    lines.append(
        f"- Seen attack detection rate (best-val-F1 threshold): "
        f"**{format_metric(best_seen_unseen['seen_attack_detection_rate'])}**"
    )
    lines.append(
        f"- Unseen attack detection rate (best-val-F1 threshold): "
        f"**{format_metric(best_seen_unseen['unseen_attack_detection_rate'])}**"
    )
    lines.append(
        f"- Benign false-positive rate (best-val-F1 threshold): "
        f"**{format_metric(best_seen_unseen['benign_fp_rate'])}**"
    )
    lines.append("")
    lines.append(
        "This shows that the temporal test set behaves not only as a chronological split, but also "
        "as a partial unseen-attack stress scenario. The model generalizes almost perfectly to seen "
        "attack families, while performance drops on unseen families."
    )
    lines.append("")
    lines.append("## 5. Attack-family observations at best-val-F1 threshold")
    lines.append("")
    lines.append("### Seen attacks")
    lines.append("")
    for _, row in seen_rows.iterrows():
        lines.append(
            f"- **{row['attack_name']}**: detection rate = "
            f"**{format_metric(row['detection_rate_or_fpr'])}** "
            f"(n = {int(row['row_count']):,})"
        )

    lines.append("")
    lines.append("### Unseen attacks")
    lines.append("")
    for _, row in unseen_rows.iterrows():
        lines.append(
            f"- **{row['attack_name']}**: detection rate = "
            f"**{format_metric(row['detection_rate_or_fpr'])}** "
            f"(n = {int(row['row_count']):,})"
        )

    lines.append("")
    lines.append(
        "Among unseen families, **ddos** and **dos** remained highly detectable, while "
        "**backdoor** and **ransomware** were harder. This indicates that unseen-attack "
        "generalization is family-dependent rather than uniformly strong or weak."
    )
    lines.append("")
    lines.append("## 6. Final positioning in the thesis")
    lines.append("")
    lines.append(
        "CIC-ToN-IoT should be positioned as a third independent benchmark. It supports the same "
        "main conclusion observed on CIC-IDS2017: strong AUC/ranking performance does not guarantee "
        "stable operational behavior under more realistic split conditions, especially when temporal "
        "shift and unseen attack families are present."
    )

    return "\n".join(lines)


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    qa_root = repo_root / "reports" / "notes" / "cic_ton_iot_qa"
    prep_root = repo_root / "reports" / "notes" / "cic_ton_iot_prep"
    split_root = repo_root / "reports" / "notes" / "cic_ton_iot_temporal_split"
    binary_report_root = repo_root / "reports" / "tables" / "cic_ton_iot_binary"

    qa_dir = find_latest_dir(qa_root, "run_*")
    prep_dir = find_latest_dir(prep_root, "run_*")
    temporal_split_dir = find_latest_dir(split_root, "run_*")
    strat_dir = find_latest_dir(binary_report_root, "lgbm_baseline_*")
    temporal_dir = find_latest_dir(binary_report_root, "lgbm_temporal_*")
    attack_eval_dir = find_latest_dir(binary_report_root, "temporal_attack_family_eval_*")

    qa_summary = load_json(qa_dir / "qa_summary.json")
    prep_summary = load_json(prep_dir / "prep_summary.json")
    temporal_split_summary = load_json(temporal_split_dir / "temporal_binary_split_summary.json")
    strat_summary = load_json(strat_dir / "summary.json")
    temporal_summary = load_json(temporal_dir / "summary.json")

    seen_unseen_df = pd.read_csv(attack_eval_dir / "temporal_seen_unseen_summary.csv")
    attack_family_df = pd.read_csv(attack_eval_dir / "attack_family_detection_by_threshold.csv")

    main_summary_df = build_main_summary(
        strat_summary=strat_summary,
        temporal_summary=temporal_summary,
        seen_unseen_df=seen_unseen_df,
    )
    target_fpr_df = build_target_fpr_summary(temporal_summary)
    best_attack_df = build_best_val_attack_family_table(attack_family_df)
    seen_unseen_out_df = build_seen_unseen_summary(seen_unseen_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        repo_root
        / "reports"
        / "final"
        / "cic_ton_iot"
        / f"package_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    main_summary_path = output_dir / "cic_ton_iot_main_summary.csv"
    target_fpr_path = output_dir / "cic_ton_iot_temporal_operating_points.csv"
    seen_unseen_path = output_dir / "cic_ton_iot_seen_unseen_summary.csv"
    best_attack_path = output_dir / "cic_ton_iot_best_val_f1_attack_family.csv"

    main_summary_df.to_csv(main_summary_path, index=False, encoding="utf-8-sig")
    target_fpr_df.to_csv(target_fpr_path, index=False, encoding="utf-8-sig")
    seen_unseen_out_df.to_csv(seen_unseen_path, index=False, encoding="utf-8-sig")
    best_attack_df.to_csv(best_attack_path, index=False, encoding="utf-8-sig")

    report_md = build_markdown_report(
        qa_summary=qa_summary,
        prep_summary=prep_summary,
        strat_summary=strat_summary,
        temporal_model_summary=temporal_summary,
        temporal_split_summary=temporal_split_summary,
        main_summary_df=main_summary_df,
        seen_unseen_df=seen_unseen_df,
        best_attack_df=best_attack_df,
    )

    report_md_path = output_dir / "cic_ton_iot_final_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")

    supporting_files = [
        qa_dir / "qa_summary.json",
        prep_dir / "prep_summary.json",
        temporal_split_dir / "temporal_binary_split_summary.json",
        strat_dir / "summary.json",
        strat_dir / "operating_points.csv",
        temporal_dir / "summary.json",
        temporal_dir / "operating_points.csv",
        attack_eval_dir / "temporal_seen_unseen_summary.csv",
        attack_eval_dir / "attack_family_detection_by_threshold.csv",
    ]

    support_dir = output_dir / "supporting_files"
    support_dir.mkdir(parents=True, exist_ok=True)

    for src in supporting_files:
        if src.exists():
            shutil.copy2(src, support_dir / src.name)

    metadata = {
        "package_output_dir": str(output_dir),
        "qa_dir": str(qa_dir),
        "prep_dir": str(prep_dir),
        "temporal_split_dir": str(temporal_split_dir),
        "stratified_report_dir": str(strat_dir),
        "temporal_report_dir": str(temporal_dir),
        "attack_eval_dir": str(attack_eval_dir),
        "generated_files": {
            "main_summary_csv": str(main_summary_path),
            "temporal_operating_points_csv": str(target_fpr_path),
            "seen_unseen_summary_csv": str(seen_unseen_path),
            "best_attack_family_csv": str(best_attack_path),
            "final_report_md": str(report_md_path),
        },
    }

    metadata_path = output_dir / "package_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("CIC-ToN-IoT FINAL PACKAGE OLUŞTURULDU")
    print("=" * 100)
    print(f"Output dir : {output_dir}")
    print("-" * 100)
    print("Üretilen ana dosyalar:")
    print(f"  - {main_summary_path}")
    print(f"  - {target_fpr_path}")
    print(f"  - {seen_unseen_path}")
    print(f"  - {best_attack_path}")
    print(f"  - {report_md_path}")
    print("-" * 100)
    print("Main summary preview:")
    print(main_summary_df.to_string(index=False))
    print("-" * 100)
    print("Seen/unseen summary preview:")
    print(seen_unseen_out_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()