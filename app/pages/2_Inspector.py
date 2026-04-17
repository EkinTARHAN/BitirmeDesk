import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common import (  # noqa: E402
    get_feature_cols,
    load_demo_cases,
    load_ids_system,
    load_binary_pipeline,
    load_shap_binary_assets,
)

st.set_page_config(page_title="Inspector — IDS Studio", layout="wide")
st.title("🔎 Tek Kayıt Denetçisi")

ids = load_ids_system()
demo_cases = load_demo_cases()
feature_cols = get_feature_cols()

left, right = st.columns([1, 2])

with left:
    st.subheader("Örnek vaka seçin")
    case_names = demo_cases["demo_case"].astype(str).tolist()
    selected = st.selectbox("Demo örneği", case_names, index=0)

    mode = st.selectbox(
        "Bilinmeyen karar modu",
        ["anomaly_only", "mc_conf_only", "union_or", "hybrid_and"],
        index=0,
    )
    conf_min = st.slider("mc_conf_min (saldırı güven eşiği)", 0.1, 0.95, 0.6, 0.05)
    st.caption("Not: multiclass confidence sadece binary_pred==1 (saldırı) için anlamlıdır.")

row = demo_cases[demo_cases["demo_case"].astype(str) == selected].iloc[0]
row_df = pd.DataFrame([row])

# sadece feature'larla skorla (demo_cases içinde eski skor kolonları olsa bile sorun olmaz)
X_input = row_df[feature_cols].copy()
scored = ids.score(X_input, mode=mode, multiclass_conf_min=float(conf_min), topk=3)

with right:
    st.subheader("Sonuç")
    st.write("**Nihai karar:**", scored.loc[0, "final_decision"])

    st.write(
        {
            "binary_prob": float(scored.loc[0, "binary_prob"]),
            "binary_pred": int(scored.loc[0, "binary_pred"]),
            "anomaly_score": float(scored.loc[0, "anomaly_score"]),
            "unknown_pred": int(scored.loc[0, "unknown_pred"]),
            "mc_pred": str(scored.loc[0, "mc_pred"]),
            "mc_max_prob": None if pd.isna(scored.loc[0, "mc_max_prob"]) else float(scored.loc[0, "mc_max_prob"]),
        }
    )

    st.divider()
    st.subheader("Ground truth (test etiketi)")
    truth = {
        "label": int(row.get("label", -1)),
        "attack_cat": str(row.get("attack_cat", "")),
        "is_unknown_target": int(row.get("is_unknown_target", 0)),
    }
    st.write(truth)

    st.divider()
    st.subheader("Top-3 attack class olasılıkları (attack ise)")
    cols = ["mc_top1_class", "mc_top1_prob", "mc_top2_class", "mc_top2_prob", "mc_top3_class", "mc_top3_prob"]
    show = {c: scored.loc[0, c] for c in cols if c in scored.columns}
    st.write(show)

    st.divider()
    st.subheader("SHAP açıklaması (Binary model: attack kararını neden verdi?)")

    assets, err = load_shap_binary_assets()
    if err is not None or assets is None:
        st.warning(
            "SHAP yüklenemedi. Muhtemelen `shap` kurulmadı veya import hatası var.\n"
            f"Hata: {err}\n\n"
            "Çözüm: `pip install shap` veya `pip install -r requirements.txt`"
        )
    else:
        from src.explain.shap_binary import explain_one  # noqa: E402

        binary_pipe = load_binary_pipeline()
        shap_df, base_value = explain_one(binary_pipe, assets, X_input, top_k=10)

        st.caption(
            "Not: SHAP değerleri genelde **log-odds** ölçeğindedir. "
            "Pozitif SHAP = attack yönüne iter, negatif SHAP = benign yönüne iter."
        )
        if base_value is not None:
            st.write({"base_value (expected)": float(base_value)})

        st.dataframe(shap_df, use_container_width=True)

    st.divider()
    st.subheader("SHAP görsel (Top-10 bar chart)")

    png_map = {
        "benign": "benign_shap_top10.png",
        "known_attack": "known_attack_shap_top10.png",
        "worms_found_as_unknown": "worms_found_as_unknown_shap_top10.png",
    }
    case_key = str(row.get("demo_case", ""))
    png_name = png_map.get(case_key)

    if png_name:
        img_path = ROOT / "reports/figures/shap_demo" / png_name
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info(f"PNG bulunamadı: {img_path}")
    else:
        st.info("Bu demo_case için hazır SHAP PNG eşlemesi yok.")

    st.divider()
    st.subheader("Feature snapshot")
    st.dataframe(X_input.T, use_container_width=True)