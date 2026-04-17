from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


# Bu uyarı sadece log kirliliği; modeli bozmaz.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _get_thr_from_summary(summary: dict, fpr_target: float) -> float:
    m = summary.get("thresholds_by_fpr_target", {})
    key = str(float(fpr_target))
    if key in m:
        return float(m[key])

    # fallback: en yakın anahtarı seç
    keys = [(abs(float(k) - float(fpr_target)), float(v)) for k, v in m.items()]
    if not keys:
        raise ValueError("thresholds_by_fpr_target not found in summary.")
    keys.sort(key=lambda x: x[0])
    return float(keys[0][1])


@dataclass
class IDSArtifacts:
    binary_pipe: object                 # sklearn Pipeline (preprocess + lgbm)
    multiclass_pipe: object             # sklearn Pipeline (preprocess + lgbm)
    multiclass_label_encoder: object    # LabelEncoder
    if_preprocess: object               # ColumnTransformer
    if_model: object                    # IsolationForest


@dataclass
class IDSConfig:
    cat_cols: list[str]
    label_col: str
    attack_cat_col: str
    unknown_flag_col: str
    drop_feature_cols: list[str]

    binary_thr: float
    anomaly_thr: float

    report_dir: Path


class IDSSystem:
    def __init__(self, cfg: IDSConfig, art: IDSArtifacts):
        self.cfg = cfg
        self.art = art

    @staticmethod
    def from_yaml(path: str | Path) -> "IDSSystem":
        cfg_raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        # threshold seçimi summary dosyalarından geliyor
        bin_sum = _read_json(Path(cfg_raw["paths"]["binary_summary"]))
        an_sum = _read_json(Path(cfg_raw["paths"]["iforest_summary"]))

        bin_fpr_t = float(cfg_raw["thresholds"]["binary_fpr_target"])
        an_fpr_t = float(cfg_raw["thresholds"]["anomaly_fpr_target"])

        binary_thr = _get_thr_from_summary(bin_sum, bin_fpr_t)
        anomaly_thr = _get_thr_from_summary(an_sum, an_fpr_t)

        data = cfg_raw["data"]
        out = cfg_raw["outputs"]

        cfg = IDSConfig(
            cat_cols=list(data["cat_cols"]),
            label_col=str(data["label_col"]),
            attack_cat_col=str(data["attack_cat_col"]),
            unknown_flag_col=str(data["unknown_flag_col"]),
            drop_feature_cols=list(data["drop_feature_cols"]),
            binary_thr=binary_thr,
            anomaly_thr=anomaly_thr,
            report_dir=Path(out["report_dir"]),
        )
        cfg.report_dir.mkdir(parents=True, exist_ok=True)

        # artifacts yükle
        binary_pipe = joblib.load(Path(cfg_raw["paths"]["binary_pipeline"]))

        mc_bundle = joblib.load(Path(cfg_raw["paths"]["multiclass_bundle"]))
        multiclass_pipe = mc_bundle["pipeline"]
        multiclass_label_encoder = mc_bundle["label_encoder"]

        if_bundle = joblib.load(Path(cfg_raw["paths"]["iforest_bundle"]))
        if_preprocess = if_bundle["preprocess"]
        if_model = if_bundle["model"]

        art = IDSArtifacts(
            binary_pipe=binary_pipe,
            multiclass_pipe=multiclass_pipe,
            multiclass_label_encoder=multiclass_label_encoder,
            if_preprocess=if_preprocess,
            if_model=if_model,
        )
        return IDSSystem(cfg, art)

    def _features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        return _safe_drop(df.copy(), self.cfg.drop_feature_cols)

    def predict_binary_prob(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features_only(df)
        p = self.art.binary_pipe.predict_proba(X)[:, 1]
        return p.astype(float)

    def predict_multiclass(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred_str: (n,) predicted class labels (string)
          max_prob: (n,) max class probability
          proba: (n, C) probability matrix
        """
        X = self._features_only(df)
        proba = self.art.multiclass_pipe.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        pred_str = self.art.multiclass_label_encoder.inverse_transform(pred_idx)
        max_prob = np.max(proba, axis=1)
        return pred_str.astype(str), max_prob.astype(float), proba.astype(float)

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        IsolationForest.decision_function: higher => more normal.
        We convert to anomaly_score: higher => more anomalous.
        """
        X = self._features_only(df)
        Xt = self.art.if_preprocess.transform(X)
        dec = self.art.if_model.decision_function(Xt)
        score = (-dec).astype(float)
        return score

    def score(
        self,
        df: pd.DataFrame,
        mode: str = "anomaly_only",
        multiclass_conf_min: float = 0.6,
        topk: int = 3,
    ) -> pd.DataFrame:
        """
        mode seçenekleri (sunum için çok değerli):
          - "anomaly_only": unknown = anomaly_score > anomaly_thr (hem benign hem attack için)
          - "mc_conf_only": benign unknown = anomaly, attack unknown = (mc_max_prob < conf_min)
          - "union_or": unknown = anomaly OR (attack & low_conf)
          - "hybrid_and": benign unknown = anomaly, attack unknown = (anomaly & low_conf)

        Not: multiclass sadece binary_pred==1 (attack) için çalıştırılır.
        """
        out = pd.DataFrame(index=df.index)

        # Binary
        bin_prob = self.predict_binary_prob(df)
        bin_pred = (bin_prob >= self.cfg.binary_thr).astype(int)

        # Anomaly
        an_score = self.anomaly_score(df)
        unknown_by_anom = (an_score > self.cfg.anomaly_thr).astype(int)

        out["binary_prob"] = bin_prob
        out["binary_pred"] = bin_pred
        out["anomaly_score"] = an_score
        out["unknown_by_anom"] = unknown_by_anom

        # Multi-class sadece attack tahmin edilenlerde
        mc_pred = np.array([""] * len(df), dtype=object)
        mc_maxp = np.full(len(df), np.nan, dtype=float)

        attack_idx = np.where(bin_pred == 1)[0]
        proba = None
        if len(attack_idx) > 0:
            df_attack = df.iloc[attack_idx]
            pred_str, max_prob, proba = self.predict_multiclass(df_attack)
            mc_pred[attack_idx] = pred_str
            mc_maxp[attack_idx] = max_prob

        out["mc_pred"] = mc_pred
        out["mc_max_prob"] = mc_maxp

        # top-k (sunumda iyi durur)
        if proba is not None:
            k = min(int(topk), proba.shape[1])
            topk_idx = np.argsort(-proba, axis=1)[:, :k]
            classes = self.art.multiclass_label_encoder.classes_.astype(str)

            for j in range(k):
                cls_j = classes[topk_idx[:, j]]
                prob_j = np.take_along_axis(proba, topk_idx[:, j:j+1], axis=1).ravel()
                tmp_cls = np.array([""] * len(df), dtype=object)
                tmp_prb = np.full(len(df), np.nan, dtype=float)
                tmp_cls[attack_idx] = cls_j
                tmp_prb[attack_idx] = prob_j
                out[f"mc_top{j+1}_class"] = tmp_cls
                out[f"mc_top{j+1}_prob"] = tmp_prb

        # Low confidence sadece attacklarda anlamlı
        low_conf_attack = ((bin_pred == 1) & (mc_maxp < float(multiclass_conf_min))).astype(int)

        # Unknown kararları
        if mode == "anomaly_only":
            unknown = unknown_by_anom

        elif mode == "mc_conf_only":
            # benign: anomaly ile uyar, attack: sadece low_conf ile uyar
            unknown = np.where(bin_pred == 0, unknown_by_anom, low_conf_attack).astype(int)

        elif mode == "union_or":
            # benign: anomaly, attack: anomaly OR low_conf
            unknown = (unknown_by_anom | low_conf_attack).astype(int)

        elif mode == "hybrid_and":
            # benign: anomaly, attack: anomaly AND low_conf
            unknown = np.where(bin_pred == 0, unknown_by_anom, (unknown_by_anom & low_conf_attack)).astype(int)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        out["unknown_pred"] = unknown
        out["mode"] = mode
        out["mc_conf_min"] = float(multiclass_conf_min)

        # Final decision (sunum için anlaşılır)
        final = []
        for i in range(len(df)):
            if unknown[i] == 1:
                final.append("unknown")
            else:
                if bin_pred[i] == 0:
                    final.append("benign")
                else:
                    final.append(f"attack:{mc_pred[i] if mc_pred[i] else 'unknowncat'}")
        out["final_decision"] = final

        return out