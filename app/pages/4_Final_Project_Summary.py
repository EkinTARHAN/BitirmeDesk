from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Final Proje Özeti",
    page_icon="📌",
    layout="wide",
)


def latest_path(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern))
    return matches[-1] if matches else None


@st.cache_data
def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("📌 Final Proje Özeti")
st.caption("Bu sayfa, projenin final seçimlerini, güncel benchmark sonuçlarını ve demo sisteminde aktif kullanılan resmi modelleri özetler.")

# ---------------------------------------------------------
# 1) Aktif demo sistemi
# ---------------------------------------------------------
st.subheader("1. Aktif Demo Sistemi")

cfg_path = ROOT / "configs" / "ids_system.yaml"
summary_path = ROOT / "reports" / "tables" / "ids_system_summary.json"

if cfg_path.exists():
    cfg = load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {})
    thr_cfg = cfg.get("thresholds", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Aktif model yolları**")
        st.write(
            {
                "binary_pipeline": paths_cfg.get("binary_pipeline"),
                "multiclass_bundle": paths_cfg.get("multiclass_bundle"),
                "iforest_bundle": paths_cfg.get("iforest_bundle"),
            }
        )
    with c2:
        st.markdown("**Aktif eşikler / ayarlar**")
        st.write(
            {
                "binary_fpr_target": thr_cfg.get("binary_fpr_target"),
                "binary_threshold_override": thr_cfg.get("binary_threshold_override"),
                "anomaly_fpr_target": thr_cfg.get("anomaly_fpr_target"),
                "multiclass_conf_mins": thr_cfg.get("multiclass_conf_mins"),
            }
        )
else:
    st.warning("configs/ids_system.yaml bulunamadı.")

if summary_path.exists():
    system_summary = load_json(summary_path)

    st.markdown("**Aktif sistem özeti**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Binary PR-AUC", f"{system_summary.get('binary_pr_auc_test', float('nan')):.6f}")
    m2.metric("Binary ROC-AUC", f"{system_summary.get('binary_roc_auc_test', float('nan')):.6f}")
    m3.metric("Binary eşik", f"{system_summary.get('binary_thr_used', float('nan')):.6f}")
    m4.metric("Anomaly eşik", f"{system_summary.get('anomaly_thr_used', float('nan')):.6f}")

    bop = system_summary.get("binary_operating_point", {})
    if bop:
        st.markdown("**Aktif binary operating point**")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("F1", f"{bop.get('f1', float('nan')):.6f}")
        o2.metric("FPR", f"{bop.get('fpr', float('nan')):.6f}")
        o3.metric("Precision", f"{bop.get('precision', float('nan')):.6f}")
        o4.metric("Recall", f"{bop.get('recall', float('nan')):.6f}")
else:
    st.warning("reports/tables/ids_system_summary.json bulunamadı.")

st.markdown("---")

# ---------------------------------------------------------
# 2) Resmi model kararları
# ---------------------------------------------------------
st.subheader("2. Resmî Model Kararları")

official_decisions_path = latest_path("reports/final/project_master/package_*/project_official_decisions.csv")

if official_decisions_path is not None and official_decisions_path.exists():
    official_df = load_csv(official_decisions_path)
    st.dataframe(official_df, use_container_width=True, hide_index=True)
else:
    st.warning("project_official_decisions.csv bulunamadı.")

st.markdown("---")

# ---------------------------------------------------------
# 3) Resmî binary sonuç özeti
# ---------------------------------------------------------
st.subheader("3. Resmî Binary Sonuç Özeti")

binary_summary_path = latest_path("reports/final/project_master/package_*/project_master_binary_summary.csv")

if binary_summary_path is not None and binary_summary_path.exists():
    binary_df = load_csv(binary_summary_path)

    show_cols = [
        "dataset",
        "benchmark_scope",
        "split_type",
        "model",
        "selection_method",
        "selected_threshold",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_fpr",
        "note",
    ]
    show_cols = [c for c in show_cols if c in binary_df.columns]
    st.dataframe(binary_df[show_cols], use_container_width=True, hide_index=True)
else:
    st.warning("project_master_binary_summary.csv bulunamadı.")

st.markdown("---")

# ---------------------------------------------------------
# 4) Multiclass recheck özeti
# ---------------------------------------------------------
st.subheader("4. Multiclass Yeniden Değerlendirme Özeti")

multiclass_recheck_path = latest_path("reports/tables/multiclass_recheck/run_*/multiclass_recheck_summary.csv")

if multiclass_recheck_path is not None and multiclass_recheck_path.exists():
    mc_df = load_csv(multiclass_recheck_path)

    show_cols = [
        "model",
        "known_test_row_count",
        "excluded_unknown_row_count",
        "excluded_unknown_labels",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "macro_precision",
        "weighted_precision",
        "roc_auc_ovr_macro",
        "roc_auc_ovr_weighted",
    ]
    show_cols = [c for c in show_cols if c in mc_df.columns]
    st.dataframe(mc_df[show_cols], use_container_width=True, hide_index=True)

    st.info(
        "Not: Bu tablo, known-only multiclass yeniden değerlendirme sonuçlarını gösterir. "
        "Worms sınıfı testte bilinmeyen olarak dışarıda bırakılmıştır."
    )
else:
    st.warning("multiclass_recheck_summary.csv bulunamadı.")

st.markdown("---")

# ---------------------------------------------------------
# 5) CIC-ToN-IoT paket özeti
# ---------------------------------------------------------
st.subheader("5. CIC-ToN-IoT Paket Özeti")

cic_ton_path = latest_path("reports/final/cic_ton_iot/package_*/cic_ton_iot_main_summary.csv")

if cic_ton_path is not None and cic_ton_path.exists():
    cic_df = load_csv(cic_ton_path)

    show_cols = [
        "benchmark_name",
        "split_type",
        "selection_method",
        "selected_threshold",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_fpr",
        "seen_attack_detection_rate",
        "unseen_attack_detection_rate",
        "note",
    ]
    show_cols = [c for c in show_cols if c in cic_df.columns]
    st.dataframe(cic_df[show_cols], use_container_width=True, hide_index=True)
else:
    st.warning("cic_ton_iot_main_summary.csv bulunamadı.")

st.markdown("---")

# ---------------------------------------------------------
# 6) Müfettiş sayfası karar mantığı açıklaması
# ---------------------------------------------------------
st.subheader("6. Müfettiş Sayfası Karar Mantığı")

st.markdown(
    """
- **Temel skorlar**: Binary olasılık, anomaly score, multiclass tahmin ve top-k adaylar aynı kayıt için modelin temel çıktılarıdır.
- **Karar modu** değişince asıl değişebilen kısım: `unknown_pred` ve buna bağlı **nihai karar**dır.
- **mc_conf_min** sadece saldırı olarak görülen kayıtlarda ve güven temelli modlarda etkilidir.
- Demo sayfası, eğitilmiş modeli değiştirmez; mevcut skorların farklı karar kuralları altında nasıl yorumlandığını gösterir.
"""
)

st.success("Bu sayfa, proje finalinde kullanılan resmî seçimler ile Streamlit demo sistemini aynı ekranda özetlemek için eklendi.")