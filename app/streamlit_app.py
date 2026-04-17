import streamlit as st

st.set_page_config(
    page_title="IDS Studio — UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ IDS Studio — Explainable Hybrid IDS (UNSW-NB15)")
st.write(
    """
Bu dashboard, bitirme projen için geliştirdiğimiz **hibrit IDS** sistemini demo amaçlı sunar:

- **Binary IDS:** benign vs attack (LightGBM)
- **Multi-class IDS:** attack türü (attack-only LightGBM)
- **Open-set / Unknown:** IsolationForest + confidence tabanlı gating stratejileri
- **Sunum modu:** önceden üretilmiş demo vakaları + batch dosyası ile “takılmadan” demo

Soldan sayfalara geçebilirsin:
- **Overview:** trade-off tabloları ve sistem metrikleri
- **Inspector:** 3 demo vaka (benign / known attack / Worms unknown)
- **Batch Scoring:** demo batch veya CSV yükleyip skor üretme
"""
)

st.info("Not: SHAP açıklanabilirlik sayfasını bir sonraki adımda ekleyeceğiz (sunum için Inspector içine entegre edeceğiz).")