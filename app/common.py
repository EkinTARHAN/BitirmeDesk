from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Proje root'unu sys.path'e ekle
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ids.score import IDSSystem  # noqa: E402


@st.cache_resource
def load_ids_system() -> IDSSystem:
    return IDSSystem.from_yaml(ROOT / "configs/ids_system.yaml")


@st.cache_data
def load_test_columns() -> list[str]:
    df = pd.read_csv(ROOT / "data/processed/splits/binary_test.csv", nrows=1)
    return df.columns.tolist()


def get_feature_cols() -> list[str]:
    cols = load_test_columns()
    drop = {"label", "attack_cat", "is_unknown_target"}
    return [c for c in cols if c not in drop]


@st.cache_data
def load_open_set_table() -> pd.DataFrame:
    p = ROOT / "reports/tables/ids_system_open_set_eval.csv"
    return pd.read_csv(p)


@st.cache_data
def load_system_summary() -> dict:
    import json
    p = ROOT / "reports/tables/ids_system_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_demo_cases() -> pd.DataFrame:
    p = ROOT / "reports/demo/demo_cases.csv"
    return pd.read_csv(p)


@st.cache_data
def load_demo_batch() -> pd.DataFrame:
    p = ROOT / "reports/demo/demo_batch.csv"
    return pd.read_csv(p)


@st.cache_data
def load_cached_scored(mode: str) -> pd.DataFrame:
    if mode == "anomaly_only":
        p = ROOT / "reports/cache/test_scored_anomaly_only.csv.gz"
    elif mode == "union_or_conf0p9":
        p = ROOT / "reports/cache/test_scored_union_or_conf0p9.csv.gz"
    else:
        raise ValueError("Unknown cache mode")
    return pd.read_csv(p)


@st.cache_resource
def load_binary_pipeline():
    """
    Binary pipeline'i doğrudan diskten yükleriz.
    (IDSSystem içinden de alınabilir ama SHAP için net olsun.)
    """
    import joblib
    p = ROOT / "models/binary/lgbm/binary_lgbm_pipeline.joblib"
    return joblib.load(p)


@st.cache_resource
def load_shap_binary_assets():
    """
    SHAP explainer + feature names.
    """
    try:
        from src.explain.shap_binary import build_shap_assets_from_pipeline  # noqa: E402
    except Exception as e:
        # SHAP kurulu değilse veya import sorunu varsa
        return None, str(e)

    pipe = load_binary_pipeline()
    assets = build_shap_assets_from_pipeline(pipe)
    return assets, None