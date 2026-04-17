from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from src.ids.score import IDSSystem
from src.explain.shap_binary import build_shap_assets_from_pipeline, explain_one


def plot_topk_shap(shap_df: pd.DataFrame, out_png: Path, title: str):
    """
    Top-k SHAP (abs) bar chart.
    """
    df = shap_df.copy().sort_values("abs_shap", ascending=True)  # yatay bar için
    plt.figure()
    plt.barh(df["feature"].astype(str), df["abs_shap"].astype(float))
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def get_feature_cols_from_test(root: Path) -> list[str]:
    """
    Feature kolonlarını demo dosyasından değil,
    gerçek test csv'den alıyoruz ki skor kolonları karışmasın.
    """
    test_path = root / "data/processed/splits/binary_test.csv"
    df0 = pd.read_csv(test_path, nrows=1)
    drop = {"label", "attack_cat", "is_unknown_target"}
    return [c for c in df0.columns if c not in drop]


def main():
    ROOT = Path(".")
    demo_cases_path = ROOT / "reports/demo/demo_cases.csv"
    out_dir = ROOT / "reports/figures/shap_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = pd.read_csv(demo_cases_path)

    # IDS + binary pipeline
    ids = IDSSystem.from_yaml(ROOT / "configs/ids_system.yaml")

    binary_pipe = joblib.load(ROOT / "models/binary/lgbm/binary_lgbm_pipeline.joblib")
    assets = build_shap_assets_from_pipeline(binary_pipe)

    feature_cols = get_feature_cols_from_test(ROOT)

    rows_summary = []

    for _, row in demo.iterrows():
        case = str(row["demo_case"])

        # Tek satır DataFrame (index sorununu garanti çözelim)
        row_df = pd.DataFrame([row]).reset_index(drop=True)

        # Skor kolonları olsa bile sadece feature kolonları ile ilerle
        X = row_df[feature_cols].copy()

        # IDS skor (bu vakada nihai karar rapora girsin)
        scored = ids.score(X, mode="anomaly_only", multiclass_conf_min=0.6, topk=3)

        # İNDEKS GÜVENLİ okuma: .iloc[0]
        final_decision = str(scored["final_decision"].iloc[0])
        binary_prob = float(scored["binary_prob"].iloc[0])
        anomaly_score = float(scored["anomaly_score"].iloc[0])
        unknown_pred = int(scored["unknown_pred"].iloc[0])
        mc_pred = str(scored["mc_pred"].iloc[0])

        # SHAP (binary model)
        shap_df, base_val = explain_one(binary_pipe, assets, X, top_k=10)

        # export CSV
        csv_path = out_dir / f"{case}_shap_top10.csv"
        shap_df.to_csv(csv_path, index=False)

        # export PNG
        title = f"SHAP Top-10 (Binary) — {case}"
        png_path = out_dir / f"{case}_shap_top10.png"
        plot_topk_shap(shap_df, png_path, title)

        rows_summary.append(
            {
                "demo_case": case,
                "ground_truth_label": int(row_df.get("label", pd.Series([-1])).iloc[0]),
                "ground_truth_attack_cat": str(row_df.get("attack_cat", pd.Series([""])).iloc[0]),
                "is_unknown_target": int(row_df.get("is_unknown_target", pd.Series([0])).iloc[0]),
                "final_decision": final_decision,
                "binary_prob": binary_prob,
                "anomaly_score": anomaly_score,
                "unknown_pred": unknown_pred,
                "mc_pred": mc_pred,
                "base_value": None if base_val is None else float(base_val),
                "shap_csv": str(csv_path),
                "shap_png": str(png_path),
            }
        )

    summary_df = pd.DataFrame(rows_summary)
    summary_path = out_dir / "demo_shap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved SHAP demo outputs to:", out_dir)
    print("Summary:", summary_path)
    print("\nSummary preview:")
    print(summary_df[["demo_case", "final_decision", "binary_prob", "anomaly_score", "unknown_pred", "mc_pred"]])


if __name__ == "__main__":
    main()