from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ids.score import IDSSystem


st.set_page_config(page_title="Tek Kayıt Denetçisi", page_icon="🔎", layout="wide")


@st.cache_resource
def load_ids_system() -> IDSSystem:
    return IDSSystem.from_yaml(ROOT / "configs" / "ids_system.yaml")


@st.cache_data
def load_test_df() -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "configs" / "ids_system.yaml").read_text(encoding="utf-8"))
    test_path = ROOT / cfg["paths"]["test_csv"]
    return pd.read_csv(test_path)


def build_demo_cases(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cases: dict[str, pd.DataFrame] = {}

    label_col = "label"
    attack_cat_col = "attack_cat"
    unknown_col = "is_unknown_target"

    if label_col not in df.columns:
        raise ValueError("Test verisinde 'label' kolonu bulunamadı.")

    benign_df = df[df[label_col] == 0]
    if not benign_df.empty:
        cases["iyi_huylu_örnek"] = benign_df.head(1).copy()

    known_attack_df = df[df[label_col] == 1]
    if unknown_col in df.columns:
        known_attack_df = known_attack_df[known_attack_df[unknown_col] == 0]
    if not known_attack_df.empty:
        cases["bilinen_saldırı"] = known_attack_df.head(1).copy()

    worms_df = pd.DataFrame()
    if attack_cat_col in df.columns:
        worms_df = df[df[attack_cat_col].astype(str).str.lower() == "worms"]
    if worms_df.empty and unknown_col in df.columns:
        worms_df = df[df[unknown_col] == 1]
    if not worms_df.empty:
        cases["solucanlar_bilinmeyen_olarak_bulundu"] = worms_df.head(1).copy()

    if not cases:
        cases["ilk_kayıt"] = df.head(1).copy()

    return cases


MODE_LABELS = {
    "anomaly_only": "sadece anormallik",
    "mc_conf_only": "sadece çoklu sınıf güveni",
    "union_or": "birleşim (anomallik VEYA düşük güven)",
    "hybrid_and": "hibrit (anomallik VE düşük güven)",
}
MODE_LABELS_REVERSE = {v: k for k, v in MODE_LABELS.items()}


st.title("🔎 Tek Kayıt Denetçisi")

st.markdown(
    """
    <style>
    .result-box {
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-box h4 {
        margin: 0 0 8px 0;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .result-box .decision {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 6px;
        line-height: 1.35;
    }
    .result-box .sub {
        font-size: 0.92rem;
        opacity: 0.92;
    }

    .result-benign {
        background: rgba(34, 197, 94, 0.16);
        border-color: rgba(34, 197, 94, 0.35);
    }
    .result-benign h4,
    .result-benign .decision,
    .result-benign .sub {
        color: #d1fae5;
    }

    .result-attack {
        background: rgba(245, 158, 11, 0.16);
        border-color: rgba(245, 158, 11, 0.35);
    }
    .result-attack h4,
    .result-attack .decision,
    .result-attack .sub {
        color: #fef3c7;
    }

    .result-unknown {
        background: rgba(239, 68, 68, 0.16);
        border-color: rgba(239, 68, 68, 0.35);
    }
    .result-unknown h4,
    .result-unknown .decision,
    .result-unknown .sub {
        color: #fee2e2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ids = load_ids_system()
test_df = load_test_df()
demo_cases = build_demo_cases(test_df)

if "inspector_result" not in st.session_state:
    st.session_state.inspector_result = None
if "inspector_input_df" not in st.session_state:
    st.session_state.inspector_input_df = None
if "inspector_meta" not in st.session_state:
    st.session_state.inspector_meta = None

left, right = st.columns([1.05, 1.0])

with left:
    st.subheader("Örnek vaka?")

    with st.form("inspector_form", clear_on_submit=False):
        case_name = st.selectbox(
            "Demo örneği",
            list(demo_cases.keys()),
            index=0,
        )

        mode_label = st.selectbox(
            "Bilinmeyen karar modu",
            list(MODE_LABELS.values()),
            index=0,
        )
        mode = MODE_LABELS_REVERSE[mode_label]

        conf_min = st.slider(
            "mc_conf_min (saldırı güven eşiği)",
            min_value=0.2,
            max_value=0.9,
            value=0.6,
            step=0.1,
        )

        submitted = st.form_submit_button("Değerlendir")

    st.caption("Not: çoklu sınıf güvenliği sadece ikili_pred==1 (saldırı) için anlamlıdır.")

    if submitted:
        X_input = demo_cases[case_name].copy()

        scored = ids.score(
            X_input,
            mode=mode,
            multiclass_conf_min=float(conf_min),
            topk=3,
        ).reset_index(drop=True)

        st.session_state.inspector_result = scored
        st.session_state.inspector_input_df = X_input.reset_index(drop=True)
        st.session_state.inspector_meta = {
            "case_name": case_name,
            "mode_label": mode_label,
            "conf_min": float(conf_min),
        }

with right:
    st.subheader("Sonuç")

    scored = st.session_state.inspector_result

    if scored is None or scored.empty:
        st.info("Soldan bir örnek ve mod seçip 'Değerlendir' butonuna bas.")
    else:
        row = scored.iloc[0]

        final_decision = str(row["final_decision"])
        binary_pred = int(row["binary_pred"])
        unknown_pred = int(row["unknown_pred"])

        if final_decision == "unknown":
            result_class = "result-unknown"
            final_label = "BİLİNMEYEN"
            final_sub = "Kayıt sistem tarafından bilinmeyen / açık-set aday olarak işaretlendi."
        elif binary_pred == 0:
            result_class = "result-benign"
            final_label = "İYİ HUYLU"
            final_sub = "Kayıt iyi huylu trafik olarak değerlendirildi."
        else:
            mc_pred = str(row["mc_pred"]) if pd.notna(row["mc_pred"]) else ""
            result_class = "result-attack"
            final_label = f"SALDIRI ({mc_pred})" if mc_pred else "SALDIRI"
            final_sub = "Kayıt saldırı olarak değerlendirildi ve saldırı ailesi tahmin edildi."

        st.markdown(
            f"""
            <div class="result-box {result_class}">
                <h4>Nihai karar</h4>
                <div class="decision">{html.escape(final_label)}</div>
                <div class="sub">{html.escape(final_sub)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("İkili", f"{float(row['binary_prob']):.6f}")
        c2.metric("İkili tahmin", "saldırı" if binary_pred == 1 else "iyi huylu")

        c3, c4 = st.columns(2)
        c3.metric("Anormallik skoru", f"{float(row['anomaly_score']):.6f}")
        c4.metric("Bilinmeyen tahmini", str(unknown_pred))

        mc_pred = str(row["mc_pred"]) if pd.notna(row["mc_pred"]) else ""
        mc_max_prob = row["mc_max_prob"]

        c5, c6 = st.columns(2)
        c5.metric("Çoklu sınıf", mc_pred if mc_pred else "-")
        c6.metric(
            "Çoklu sınıf maksimum",
            f"{float(mc_max_prob):.6f}" if pd.notna(mc_max_prob) else "-",
        )

        topk_cols = [
            ("mc_top1_class", "mc_top1_prob"),
            ("mc_top2_class", "mc_top2_prob"),
            ("mc_top3_class", "mc_top3_prob"),
        ]

        topk_rows = []
        for cls_col, prob_col in topk_cols:
            if cls_col in scored.columns and prob_col in scored.columns:
                cls_val = row[cls_col]
                prob_val = row[prob_col]
                if pd.notna(prob_val) and str(cls_val) != "":
                    topk_rows.append(
                        {
                            "Sıra": cls_col.replace("_class", "").replace("mc_", "").upper(),
                            "Sınıf": str(cls_val),
                            "Olasılık": float(prob_val),
                        }
                    )

        st.markdown("### Top-k saldırı adayları")
        if topk_rows:
            st.dataframe(pd.DataFrame(topk_rows), use_container_width=True, hide_index=True)
        else:
            st.write("-")

        if st.session_state.inspector_meta is not None:
            st.markdown("### Kullanılan ayarlar")
            st.write(
                {
                    "demo": st.session_state.inspector_meta["case_name"],
                    "mod": st.session_state.inspector_meta["mode_label"],
                    "mc_conf_min": st.session_state.inspector_meta["conf_min"],
                }
            )

        st.markdown("### Ham karar detayları")
        raw_details = {
            "final_decision": final_decision,
            "unknown_pred": int(row["unknown_pred"]),
            "binary_pred": int(row["binary_pred"]),
            "mc_pred": mc_pred if mc_pred else None,
            "mc_max_prob": float(mc_max_prob) if pd.notna(mc_max_prob) else None,
            "mode": st.session_state.inspector_meta["mode_label"] if st.session_state.inspector_meta else None,
            "mc_conf_min": st.session_state.inspector_meta["conf_min"] if st.session_state.inspector_meta else None,
        }
        st.json(raw_details)

st.markdown("---")
st.markdown("**Seçili giriş kaydı**")

if st.session_state.inspector_input_df is not None:
    st.dataframe(st.session_state.inspector_input_df, use_container_width=True, hide_index=True)
else:
    st.info("Henüz değerlendirilmiş bir kayıt yok.")