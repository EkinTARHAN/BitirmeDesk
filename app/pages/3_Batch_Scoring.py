import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common import get_feature_cols, load_demo_batch, load_ids_system  # noqa: E402

st.set_page_config(page_title="Batch Scoring — IDS Studio", layout="wide")
st.title("📦 Toplu Puanlama")

ids = load_ids_system()
feature_cols = get_feature_cols()

st.subheader("1) Demo toplu işlem veya CSV yükleme")
use_demo = st.checkbox("Demo batch dosyasını kullanın", value=True)

uploaded = None
if not use_demo:
    uploaded = st.file_uploader("CSV yükle", type=["csv"])

mode = st.selectbox("Bilinmeyen karar modu", ["anomaly_only", "mc_conf_only", "union_or", "hybrid_and"], index=0)
conf_min = st.slider("mc_conf_min", 0.1, 0.95, 0.6, 0.05)

# Eski skor kolonları (varsa) bunları temizleyeceğiz
SCORE_COLS = {
    "binary_prob", "binary_pred",
    "anomaly_score", "unknown_by_anom", "unknown_pred",
    "mc_pred", "mc_max_prob",
    "mc_top1_class", "mc_top1_prob",
    "mc_top2_class", "mc_top2_prob",
    "mc_top3_class", "mc_top3_prob",
    "final_decision", "mode", "mc_conf_min",
    "risk_rank_key",
}

if st.button("Skorla", type="primary"):
    if use_demo:
        df = load_demo_batch()
    else:
        if uploaded is None:
            st.error("Lütfen bir CSV yükle veya demo batch’i seç.")
            st.stop()
        df = pd.read_csv(uploaded)

    st.subheader("2) Sonuçlar")
    st.write("Toplam satır:", len(df))

    # Eğer input dosyasında daha önce skor kolonları varsa temizle
    dup_cols = [c for c in df.columns if c in SCORE_COLS]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    # Feature kontrolü
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Yüklenen dosyada eksik kolonlar var: {missing[:10]}{'...' if len(missing)>10 else ''}")
        st.stop()

    # Skorla
    X = df[feature_cols].copy()
    scored = ids.score(X, mode=mode, multiclass_conf_min=float(conf_min), topk=3)

    out = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    # Risk sıralama (unknown + anomaly)
    out_rank = out.copy()
    out_rank["risk_rank_key"] = out_rank["unknown_pred"].astype(int) * 10_000 + out_rank["anomaly_score"].astype(float)
    out_rank = out_rank.sort_values(by="risk_rank_key", ascending=False)

    st.write("**En şüpheli 30 kayıt**")
    st.dataframe(
        out_rank.head(30)[
            [c for c in ["final_decision", "unknown_pred", "anomaly_score", "binary_prob", "binary_pred", "mc_pred", "mc_max_prob"] if c in out_rank.columns]
        ],
        use_container_width=True,
    )

    st.divider()
    st.subheader("3) İndir")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Sonuç CSV indir",
        data=csv_bytes,
        file_name="batch_scored.csv",
        mime="text/csv",
    )