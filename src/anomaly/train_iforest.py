from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def make_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """
    IF için hafif preprocess:
    - Categoricals: OrdinalEncoder (unknown -> -1)
    - Numerics: StandardScaler
    """
    cat_tf = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num_tf = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
    )


def anomaly_score_from_decision(decision_scores: np.ndarray) -> np.ndarray:
    """
    IsolationForest.decision_function:
      daha yüksek => daha 'normal' (inlier)
    Bizim anomaly_score:
      daha yüksek => daha 'anomali'
    """
    return (-decision_scores).astype(float)


def pick_threshold_by_benign_fpr(benign_scores: np.ndarray, fpr_target: float) -> float:
    """
    Benign set üzerinde FPR hedefi için threshold:
    benign_scores'in (1 - fpr_target) quantile'ı
    """
    benign_scores = np.asarray(benign_scores).astype(float)
    q = 1.0 - float(fpr_target)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(benign_scores, q))


def summarize_open_set(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    label_col: str,
    unknown_flag_col: str,
    thr: float,
) -> dict:
    """
    Testte:
    - benign_fpr: benign içinde unknown flag oranı
    - worms_detection: unknown_target==1 içinde unknown flag oranı
    - known_attack_unknown_rate: known attack içinde unknown flag oranı
    """
    scores = np.asarray(scores).astype(float)
    is_unknown_target = df_test[unknown_flag_col].to_numpy().astype(int) if unknown_flag_col in df_test.columns else np.zeros(len(df_test), dtype=int)
    y_label = df_test[label_col].to_numpy().astype(int)

    pred_unknown = (scores > thr).astype(int)

    benign_mask = (y_label == 0)
    attack_mask = (y_label == 1)
    unknown_mask = (is_unknown_target == 1)
    known_attack_mask = attack_mask & (~unknown_mask)

    benign_fpr = float(pred_unknown[benign_mask].mean()) if benign_mask.any() else 0.0
    worms_det = float(pred_unknown[unknown_mask].mean()) if unknown_mask.any() else 0.0
    known_attack_unk = float(pred_unknown[known_attack_mask].mean()) if known_attack_mask.any() else 0.0

    return {
        "threshold": float(thr),
        "benign_fpr_test": benign_fpr,
        "worms_detection_rate_test": worms_det,
        "known_attack_unknown_rate_test": known_attack_unk,
        "test_unknown_count": int(unknown_mask.sum()),
        "test_benign_count": int(benign_mask.sum()),
        "test_attack_count": int(attack_mask.sum()),
        "test_known_attack_count": int(known_attack_mask.sum()),
    }


def main():
    cfg = yaml.safe_load(Path("configs/anomaly_if.yaml").read_text(encoding="utf-8"))

    cat_cols = list(cfg["data"]["cat_cols"])
    label_col = str(cfg["data"]["target_label"])
    unknown_flag_col = str(cfg["data"]["unknown_flag_col"])
    drop_feature_cols = list(cfg["data"]["drop_feature_cols"])

    p_train = Path(cfg["paths"]["binary_train"])
    p_val = Path(cfg["paths"]["binary_val"])
    p_test = Path(cfg["paths"]["binary_test"])

    model_dir = Path(cfg["outputs"]["model_dir"])
    report_dir = Path(cfg["outputs"]["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_tr = pd.read_csv(p_train)
    df_va = pd.read_csv(p_val)
    df_te = pd.read_csv(p_test)

    # Benign-only train/val
    tr_ben = df_tr[df_tr[label_col] == 0].copy()
    va_ben = df_va[df_va[label_col] == 0].copy()

    # Features: leakage kolonları düş
    X_tr = tr_ben.drop(columns=[c for c in drop_feature_cols if c in tr_ben.columns])
    X_va = va_ben.drop(columns=[c for c in drop_feature_cols if c in va_ben.columns])
    X_te = df_te.drop(columns=[c for c in drop_feature_cols if c in df_te.columns])

    # Kolon listeleri
    cat_cols = [c for c in cat_cols if c in X_tr.columns]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    pre = make_preprocessor(cat_cols, num_cols)

    if_params = dict(cfg["iforest"])
    iso = IsolationForest(**if_params)

    # Preprocess fit (benign train)
    pre.fit(X_tr)
    Xtr = pre.transform(X_tr)
    Xva = pre.transform(X_va)
    Xte = pre.transform(X_te)

    # Train IF (benign-only)
    iso.fit(Xtr)

    # Scores
    va_dec = iso.decision_function(Xva)
    te_dec = iso.decision_function(Xte)

    va_scores = anomaly_score_from_decision(va_dec)
    te_scores = anomaly_score_from_decision(te_dec)

    # Thresholds by FPR targets (on VAL BENIGN)
    fpr_targets = list(cfg["threshold"]["fpr_targets"])
    thr_map = {}
    rows = []

    # Open-set AUC (unknown vs benign) - test üzerinden raporlayalım
    # (Worms = is_unknown_target==1, benign = label==0)
    if unknown_flag_col in df_te.columns:
        y_unknown = df_te[unknown_flag_col].to_numpy().astype(int)
        y_benign = (df_te[label_col].to_numpy().astype(int) == 0).astype(int)
        # AUC: worms(1) vs benign(0) için maske
        mask = (y_unknown == 1) | (y_benign == 1)
        if mask.sum() > 0 and len(np.unique(y_unknown[mask])) > 1:
            y_bin = y_unknown[mask]  # worms=1, benign=0
            s_bin = te_scores[mask]
            open_roc = float(roc_auc_score(y_bin, s_bin))
            open_pr = float(average_precision_score(y_bin, s_bin))
        else:
            open_roc, open_pr = None, None
    else:
        open_roc, open_pr = None, None

    for fpr_t in fpr_targets:
        fpr_t = float(fpr_t)
        thr = pick_threshold_by_benign_fpr(va_scores, fpr_t)
        thr_map[str(fpr_t)] = thr

        summ = summarize_open_set(df_te, te_scores, label_col, unknown_flag_col, thr)
        summ.update({"fpr_target": fpr_t})
        rows.append(summ)

    summary = {
        "val_benign_count": int(len(tr_ben)),
        "val_benign_for_threshold_count": int(len(va_ben)),
        "test_shape": list(df_te.shape),
        "cat_cols_used": cat_cols,
        "num_cols_count": int(len(num_cols)),
        "thresholds_by_fpr_target": thr_map,
        "open_set_auc_worms_vs_benign_test_roc": open_roc,
        "open_set_auc_worms_vs_benign_test_pr": open_pr,
    }

    # Save bundle
    bundle = {"preprocess": pre, "model": iso, "summary": summary}
    joblib.dump(bundle, model_dir / "iforest_bundle.joblib")
    (model_dir / "iforest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save eval table
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(report_dir / "anomaly_iforest_open_set_eval.csv", index=False)

    print("\n=== IFORREST SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(" -", model_dir / "iforest_bundle.joblib")
    print(" -", model_dir / "iforest_summary.json")
    print(" -", report_dir / "anomaly_iforest_open_set_eval.csv")


if __name__ == "__main__":
    main()