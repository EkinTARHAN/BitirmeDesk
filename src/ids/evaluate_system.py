from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)

from src.ids.score import IDSSystem

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def binary_eval(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "binary_threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _summarize_system(df: pd.DataFrame, scored: pd.DataFrame,
                      label_col: str, attack_cat_col: str, unknown_col: str,
                      ids: IDSSystem) -> dict:

    y_label = df[label_col].to_numpy().astype(int)
    is_unknown_target = df[unknown_col].to_numpy().astype(int) if unknown_col in df.columns else np.zeros(len(df), dtype=int)

    pred_unknown = scored["unknown_pred"].to_numpy().astype(int)
    bin_pred = scored["binary_pred"].to_numpy().astype(int)

    benign_mask = (y_label == 0)
    attack_mask = (y_label == 1)
    worms_mask = (is_unknown_target == 1)
    known_attack_mask = attack_mask & (~worms_mask)

    benign_unknown_rate = float(pred_unknown[benign_mask].mean()) if benign_mask.any() else 0.0
    worms_detection_rate = float(pred_unknown[worms_mask].mean()) if worms_mask.any() else 0.0
    known_attack_unknown_rate = float(pred_unknown[known_attack_mask].mean()) if known_attack_mask.any() else 0.0

    # Sistem multi-class performansı: known attacks & unknown değil & binary attack dediğimiz örnekler
    sys_mc_mask = known_attack_mask & (pred_unknown == 0) & (bin_pred == 1)

    known_attack_count = int(known_attack_mask.sum())
    sys_mc_count = int(sys_mc_mask.sum())

    if sys_mc_mask.any():
        y_true_cat = df.loc[sys_mc_mask, attack_cat_col].astype(str).str.strip().to_numpy()
        y_pred_cat = scored.loc[sys_mc_mask, "mc_pred"].astype(str).to_numpy()
        labels = sorted(list(set(y_true_cat.tolist()) | set(y_pred_cat.tolist())))
        mc_macro_f1 = float(f1_score(y_true_cat, y_pred_cat, average="macro", labels=labels))
        mc_weighted_f1 = float(f1_score(y_true_cat, y_pred_cat, average="weighted", labels=labels))
    else:
        mc_macro_f1 = 0.0
        mc_weighted_f1 = 0.0

    # coverage:
    # - tüm test içinde oran
    coverage_all = float(sys_mc_mask.mean())
    # - known attacks içinde oran (sunumda daha anlaşılır)
    coverage_known_attacks = _safe_div(sys_mc_count, known_attack_count)

    return {
        "mode": str(scored["mode"].iloc[0]),
        "mc_conf_min": float(scored["mc_conf_min"].iloc[0]),
        "binary_thr_used": float(ids.cfg.binary_thr),
        "anomaly_thr_used": float(ids.cfg.anomaly_thr),

        "benign_unknown_rate": benign_unknown_rate,
        "worms_detection_rate": worms_detection_rate,
        "known_attack_unknown_rate": known_attack_unknown_rate,

        "sys_mc_count": sys_mc_count,
        "known_attack_count": known_attack_count,
        "sys_coverage_all_rows": coverage_all,
        "sys_coverage_among_known_attacks": coverage_known_attacks,

        "sys_mc_macro_f1": mc_macro_f1,
        "sys_mc_weighted_f1": mc_weighted_f1,
    }


def main():
    cfg = yaml.safe_load(Path("configs/ids_system.yaml").read_text(encoding="utf-8"))

    ids = IDSSystem.from_yaml("configs/ids_system.yaml")

    test_path = Path(cfg["paths"]["test_csv"])
    df = pd.read_csv(test_path)

    label_col = cfg["data"]["label_col"]
    attack_cat_col = cfg["data"]["attack_cat_col"]
    unknown_col = cfg["data"]["unknown_flag_col"]

    y = df[label_col].to_numpy().astype(int)

    # Binary (threshold-independent + operating point)
    bin_prob = ids.predict_binary_prob(df)
    bin_pr_auc = float(average_precision_score(y, bin_prob))
    bin_roc_auc = float(roc_auc_score(y, bin_prob))
    bin_eval = binary_eval(y, bin_prob, ids.cfg.binary_thr)

    modes = list(cfg["evaluation"]["modes"])
    mc_conf_mins = list(cfg["thresholds"]["multiclass_conf_mins"])

    rows = []
    for mode in modes:
        if mode == "anomaly_only":
            scored = ids.score(df, mode=mode, multiclass_conf_min=0.6)
            rows.append(_summarize_system(df, scored, label_col, attack_cat_col, unknown_col, ids))
        else:
            for cm in mc_conf_mins:
                scored = ids.score(df, mode=mode, multiclass_conf_min=float(cm))
                rows.append(_summarize_system(df, scored, label_col, attack_cat_col, unknown_col, ids))

    out_dir = Path(cfg["outputs"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_df = pd.DataFrame(rows)
    sys_df.to_csv(out_dir / "ids_system_open_set_eval.csv", index=False)

    summary = {
        "binary_pr_auc_test": bin_pr_auc,
        "binary_roc_auc_test": bin_roc_auc,
        "binary_operating_point": bin_eval,
        "binary_thr_fpr_target": float(cfg["thresholds"]["binary_fpr_target"]),
        "binary_thr_used": float(ids.cfg.binary_thr),
        "anomaly_thr_fpr_target": float(cfg["thresholds"]["anomaly_fpr_target"]),
        "anomaly_thr_used": float(ids.cfg.anomaly_thr),
        "evaluated_modes": modes,
        "mc_conf_mins": mc_conf_mins,
    }

    (out_dir / "ids_system_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== IDS SYSTEM SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(" -", out_dir / "ids_system_open_set_eval.csv")
    print(" -", out_dir / "ids_system_summary.json")


if __name__ == "__main__":
    main()