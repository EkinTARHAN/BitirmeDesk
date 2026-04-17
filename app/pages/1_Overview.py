import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common import load_open_set_table, load_system_summary  # noqa: E402

st.set_page_config(page_title="Overview — IDS Studio", layout="wide")
st.title("📊 Genel Bakış")

summary = load_system_summary()
df = load_open_set_table()

c1, c2, c3 = st.columns(3)
c1.metric("İkili PR-AUC (test)", f"{summary['binary_pr_auc_test']:.4f}")
c2.metric("İkili ROC-AUC (test)", f"{summary['binary_roc_auc_test']:.4f}")
c3.metric("İkili eşik (kullanılan)", f"{summary['binary_thr_used']:.4f}")

with st.expander("İkili çalışma noktası (karışıklık + FPR/FNR)", expanded=True):
    op = summary["binary_operating_point"]
    st.write(op)

st.subheader("Açık uçlu / Bilinmeyen ödünleşmeler (mod + güven)")
st.caption("Farklı unknown karar stratejilerinin benign yanlış alarmı ve Worms yakalama trade-off’u.")
st.dataframe(df, use_container_width=True)

st.subheader("Hızlı görselleştirme")
plot_df = df.copy()
plot_df["mc_conf_min"] = plot_df["mc_conf_min"].fillna(-1)

st.write("**Benign Unknown Rate (x) vs Worms Detection Rate (y)**")
st.scatter_chart(plot_df, x="benign_unknown_rate", y="worms_detection_rate")

st.write("**Worms Detection Rate (x) vs Known-Attack Unknown Rate (y)**")
st.scatter_chart(plot_df, x="worms_detection_rate", y="known_attack_unknown_rate")

st.divider()
st.header("🔍 Güvenilirlik (Calibration) Analizi")

metrics_path = ROOT / "reports/tables/binary_calibration_metrics.csv"
thr_path = ROOT / "reports/tables/binary_calibration_threshold_eval.csv"
fig_dir = ROOT / "reports/figures/calibration"

if not metrics_path.exists() or not thr_path.exists():
    st.info("Calibration raporları bulunamadı. Önce: `python -m src.calibration.calibrate_binary` çalıştırın.")
else:
    metrics_df = pd.read_csv(metrics_path)
    thr_df = pd.read_csv(thr_path)

    st.subheader("Calibration metrikleri (Brier / LogLoss / ECE)")
    st.caption("Bu metriklerde daha düşük değer daha iyidir. ECE, olasılıkların güvenilirliğini (kalibrasyon) ölçer.")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Threshold stabilitesi (VAL hedef FPR → TEST FPR)")
    st.caption("VAL’de seçilen eşik, TEST’te aynı FPR’ı her zaman korumayabilir. Bu drift/dağılım farkı bulgusudur.")
    st.dataframe(thr_df, use_container_width=True)

    st.subheader("Reliability plot (kalibrasyon eğrisi)")
    st.caption("Kesikli diyagonal çizgi ‘mükemmel kalibrasyon’ demektir. Eğri ne kadar yakında ise o kadar güvenilir olasılık.")

    split = st.selectbox("Split seç", ["val", "test"], index=1)
    method = st.selectbox("Yöntem seç", ["raw", "platt", "isotonic"], index=0)

    img_path = fig_dir / f"reliability_{split}_{method}.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info(f"Görsel bulunamadı: {img_path}")

st.subheader("Sunum için kısa mesaj (hazır)")
st.markdown(
    """
- *“Modelim güçlü ama IDS’te operasyonel hedef yanlış alarm bütçesidir (FPR). Bu yüzden threshold’u val’de hedef FPR ile seçtim.”*
- *“Olasılıkların güvenilirliğini calibration (Platt/Isotonic) ve reliability plot ile analiz ettim.”*
- *“Val’de çok iyi görünen kalibrasyon, testte her zaman iyileşmeyebilir; bu da drift/overfit ve sağlam calibration protokolü ihtiyacını gösterir.”*
"""
)