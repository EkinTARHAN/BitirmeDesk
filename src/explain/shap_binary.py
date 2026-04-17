from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# shap importunu burada yapıyoruz (streamlit tarafında try/except ile yöneteceğiz)
import shap


@dataclass
class ShapBinaryAssets:
    explainer: object
    feature_names: list[str]
    model_output_scale: str  # "log_odds" or "unknown"


def _get_feature_names(preprocess) -> list[str]:
    """
    ColumnTransformer + OneHotEncoder içeren pipeline'dan feature isimlerini çıkarır.
    """
    try:
        # sklearn >= 1.0 için genellikle çalışır
        return preprocess.get_feature_names_out().tolist()
    except Exception:
        # fallback: el ile toparlama (nadir)
        names = []
        if hasattr(preprocess, "transformers_"):
            for name, trans, cols in preprocess.transformers_:
                if name == "remainder" or trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    try:
                        names.extend(trans.get_feature_names_out(cols).tolist())
                    except Exception:
                        # trans isim veremezse ham kolon adları
                        names.extend([str(c) for c in cols])
                else:
                    names.extend([str(c) for c in cols])
        return names


def build_shap_assets_from_pipeline(binary_pipeline) -> ShapBinaryAssets:
    """
    binary_pipeline: sklearn Pipeline([('preprocess', ...), ('model', LGBMClassifier)])
    """
    preprocess = binary_pipeline.named_steps["preprocess"]
    model = binary_pipeline.named_steps["model"]

    feature_names = _get_feature_names(preprocess)

    # TreeExplainer genelde LightGBM ile iyi çalışır.
    # SHAP değerleri çoğu durumda log-odds uzayındadır (bunu Inspector'da belirteceğiz).
    explainer = shap.TreeExplainer(model)

    return ShapBinaryAssets(
        explainer=explainer,
        feature_names=feature_names,
        model_output_scale="log_odds",
    )


def explain_one(
    binary_pipeline,
    assets: ShapBinaryAssets,
    X_df: pd.DataFrame,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Tek örnek için SHAP tablo üretir.
    Returns:
      - shap_df: feature, value, shap_value, abs_shap
      - base_value: expected_value (varsa)
    """
    preprocess = binary_pipeline.named_steps["preprocess"]

    Xt = preprocess.transform(X_df)
    # 1 satır bekliyoruz
    if hasattr(Xt, "toarray"):
        x_row = Xt.toarray()
    else:
        x_row = np.asarray(Xt)

    # SHAP hesapla
    # shap_values bazen list döndürür (binaryde [class0, class1]) bazen direkt array döndürür
    shap_vals = assets.explainer.shap_values(x_row)

    if isinstance(shap_vals, list):
        # class1 (attack) için al
        if len(shap_vals) >= 2:
            sv = shap_vals[1]
        else:
            sv = shap_vals[0]
    else:
        sv = shap_vals

    sv = np.asarray(sv).reshape(-1)  # (n_features,)

    # base value
    base = getattr(assets.explainer, "expected_value", None)
    if isinstance(base, (list, np.ndarray)):
        # binaryde bazen [base0, base1]
        try:
            base_value = float(base[1])
        except Exception:
            base_value = float(base[0])
    elif base is None:
        base_value = None
    else:
        base_value = float(base)

    # feature values
    fv = x_row.reshape(-1)
    fn = assets.feature_names
    if len(fn) != len(fv):
        # olası edge: isim sayısı uyuşmazsa indeks isimleri
        fn = [f"f_{i}" for i in range(len(fv))]

    df = pd.DataFrame(
        {
            "feature": fn,
            "value": fv,
            "shap_value": sv,
            "abs_shap": np.abs(sv),
        }
    ).sort_values("abs_shap", ascending=False)

    df_top = df.head(int(top_k)).reset_index(drop=True)
    return df_top, base_value