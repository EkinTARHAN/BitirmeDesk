from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve


def safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols) if cols else df


def get_X_y(df: pd.DataFrame, label_col: str = "label") -> Tuple[pd.DataFrame, np.ndarray]:
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in dataframe columns.")
    y = df[label_col].to_numpy().astype(int)

    # IDS pipeline'ında feature olmayan kolonları düş
    drop_cols = [label_col, "attack_cat", "is_unknown_target", "id"]
    X = safe_drop(df.copy(), drop_cols)
    return X, y


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) - basit binning yaklaşımı.
    """
    y_true = y_true.astype(int)
    p = np.clip(p.astype(float), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def pick_threshold_by_fpr(y_true: np.ndarray, p: np.ndarray, fpr_target: float) -> float:
    """
    FPR hedefi: benign (y=0) dağılımı üzerinden threshold seç.
    Threshold'ü, benign p değerlerinin (1-fpr_target) quantile'ı gibi düşünebilirsin.
    """
    p0 = p[y_true == 0]
    if len(p0) == 0:
        raise ValueError("No benign samples found for threshold selection.")
    # FPR target => benignlerin fpr_target kadarı threshold üstünde kalsın
    thr = float(np.quantile(p0, 1.0 - float(fpr_target)))
    return thr


def eval_operating_point(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def plot_reliability(y_true: np.ndarray, p: np.ndarray, out_png: Path, title: str, n_bins: int = 15):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ROOT = Path(".")

    # Girdi dosyaları
    pipeline_path = ROOT / "models/binary/lgbm/binary_lgbm_pipeline.joblib"
    val_path = ROOT / "data/processed/splits/binary_val.csv"
    test_path = ROOT / "data/processed/splits/binary_test.csv"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing: {pipeline_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}")

    pipe = joblib.load(pipeline_path)

    # Veriyi oku
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_val, y_val = get_X_y(df_val, label_col="label")
    X_test, y_test = get_X_y(df_test, label_col="label")

    # Ham olasılıklar
    p_val_raw = pipe.predict_proba(X_val)[:, 1].astype(float)
    p_test_raw = pipe.predict_proba(X_test)[:, 1].astype(float)

    # --- Kalibrasyon: Isotonic ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_val_iso = iso.transform(p_val_raw)
    p_test_iso = iso.transform(p_test_raw)

    # --- Kalibrasyon: Platt (logit(p) -> logistic regression) ---
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(prob_to_logit(p_val_raw).reshape(-1, 1), y_val)
    p_val_platt = lr.predict_proba(prob_to_logit(p_val_raw).reshape(-1, 1))[:, 1]
    p_test_platt = lr.predict_proba(prob_to_logit(p_test_raw).reshape(-1, 1))[:, 1]

    # Metrikler (Brier, LogLoss, ECE)
    n_bins = 15
    metrics_rows = []

    def add_metrics(split: str, method: str, y: np.ndarray, p: np.ndarray):
        metrics_rows.append({
            "split": split,
            "method": method,
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "ece": float(ece_score(y, p, n_bins=n_bins))
        })

    add_metrics("val", "raw", y_val, p_val_raw)
    add_metrics("val", "isotonic", y_val, p_val_iso)
    add_metrics("val", "platt", y_val, p_val_platt)

    add_metrics("test", "raw", y_test, p_test_raw)
    add_metrics("test", "isotonic", y_test, p_test_iso)
    add_metrics("test", "platt", y_test, p_test_platt)

    metrics_df = pd.DataFrame(metrics_rows)

    # Threshold kıyası: val'de target FPR ile threshold seç -> testte ne oluyor?
    fpr_targets = [0.005, 0.01, 0.02]
    thr_rows = []

    preds = {
        "raw": (p_val_raw, p_test_raw),
        "isotonic": (p_val_iso, p_test_iso),
        "platt": (p_val_platt, p_test_platt),
    }

    for method, (pv, pt) in preds.items():
        for fpr_t in fpr_targets:
            thr = pick_threshold_by_fpr(y_val, pv, fpr_t)
            val_op = eval_operating_point(y_val, pv, thr)
            test_op = eval_operating_point(y_test, pt, thr)

            thr_rows.append({
                "method": method,
                "fpr_target_val": float(fpr_t),
                "threshold": float(thr),

                "val_fpr": val_op["fpr"],
                "val_fnr": val_op["fnr"],
                "val_f1": val_op["f1"],

                "test_fpr": test_op["fpr"],
                "test_fnr": test_op["fnr"],
                "test_f1": test_op["f1"],
            })

    thr_df = pd.DataFrame(thr_rows)

    # Kayıt: calibrator modelleri
    out_models = ROOT / "models/binary/calibration"
    out_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, out_models / "isotonic.joblib")
    joblib.dump(lr, out_models / "platt.joblib")

    # Kayıt: tablolar
    out_tables = ROOT / "reports/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_tables / "binary_calibration_metrics.csv", index=False)
    thr_df.to_csv(out_tables / "binary_calibration_threshold_eval.csv", index=False)

    # Kayıt: görseller
    out_fig = ROOT / "reports/figures/calibration"
    out_fig.mkdir(parents=True, exist_ok=True)

    # reliability plots
    plot_reliability(y_val, p_val_raw, out_fig / "reliability_val_raw.png", "Reliability (VAL) - RAW", n_bins=n_bins)
    plot_reliability(y_val, p_val_iso, out_fig / "reliability_val_isotonic.png", "Reliability (VAL) - Isotonic", n_bins=n_bins)
    plot_reliability(y_val, p_val_platt, out_fig / "reliability_val_platt.png", "Reliability (VAL) - Platt", n_bins=n_bins)

    plot_reliability(y_test, p_test_raw, out_fig / "reliability_test_raw.png", "Reliability (TEST) - RAW", n_bins=n_bins)
    plot_reliability(y_test, p_test_iso, out_fig / "reliability_test_isotonic.png", "Reliability (TEST) - Isotonic", n_bins=n_bins)
    plot_reliability(y_test, p_test_platt, out_fig / "reliability_test_platt.png", "Reliability (TEST) - Platt", n_bins=n_bins)

    # kısa özet
    print("\nSaved calibration artifacts:")
    print(" -", out_models / "isotonic.joblib")
    print(" -", out_models / "platt.joblib")
    print(" -", out_tables / "binary_calibration_metrics.csv")
    print(" -", out_tables / "binary_calibration_threshold_eval.csv")
    print(" -", out_fig)

    print("\nCalibration metrics (lower is better for brier/logloss/ece):")
    print(metrics_df.to_string(index=False))

    print("\nThreshold stability (val target -> test FPR):")
    print(thr_df.sort_values(["method", "fpr_target_val"]).to_string(index=False))


if __name__ == "__main__":
    main()