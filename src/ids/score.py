from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _get_nested(d: dict, path: list[str]) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_relative_path(path_str: str, summary_path: Path) -> Path | None:
    p = Path(path_str)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((summary_path.parent / p).resolve())
        candidates.append(p.resolve())

    for cand in candidates:
        if cand.exists():
            return cand

    return None


def _extract_threshold_from_artifact_csv(
    summary: dict,
    fpr_target: float,
    summary_path: Path,
) -> float | None:
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None

    candidate_keys = [
        "best_operating_points_path",
        "operating_points_path",
        "best_threshold_sweep_path",
        "threshold_sweep_path",
    ]

    for key in candidate_keys:
        rel_path = artifacts.get(key)
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue

        csv_path = _resolve_relative_path(rel_path, summary_path)
        if csv_path is None or not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        threshold_col = None
        for col in ["threshold", "selected_threshold", "best_threshold", "best_thr"]:
            if col in df.columns:
                threshold_col = col
                break

        if threshold_col is None:
            continue

        work = df.copy()

        if "selection_method" in work.columns:
            target_rows = work[
                work["selection_method"].astype(str).str.contains("target", case=False, na=False)
            ]
            if not target_rows.empty:
                work = target_rows.copy()

        target_col = None
        for col in ["target_fpr", "fpr_target", "requested_fpr", "target_fpr_requested"]:
            if col in work.columns:
                target_col = col
                break

        work[threshold_col] = pd.to_numeric(work[threshold_col], errors="coerce")

        if target_col is not None:
            work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
            work = work.dropna(subset=[target_col, threshold_col])

            if not work.empty:
                work["_dist"] = (work[target_col] - float(fpr_target)).abs()
                row = work.sort_values("_dist", ascending=True).iloc[0]
                return float(row[threshold_col])

        work = work.dropna(subset=[threshold_col])
        if not work.empty:
            return float(work.iloc[0][threshold_col])

    return None


def _get_thr_from_summary(summary: dict, fpr_target: float, summary_path: Path) -> float:
    """
    Farklı summary şemalarından threshold çekmeye çalışır.

    Öncelik sırası:
    1) thresholds_by_fpr_target[<target>]
    2) selected_operating_point.threshold / best_operating_point.threshold
    3) selected_threshold / best_threshold / best_thr / threshold
    4) artifacts içindeki operating-points CSV dosyasından threshold
    """
    m = summary.get("thresholds_by_fpr_target", {})
    if isinstance(m, dict) and len(m) > 0:
        target_keys = [
            str(fpr_target),
            str(float(fpr_target)),
            f"{float(fpr_target):.3f}",
            f"{float(fpr_target):.4f}",
            f"{float(fpr_target):.6f}",
        ]

        for k in target_keys:
            if k in m:
                return float(m[k])

        keys = []
        for k, v in m.items():
            try:
                keys.append((abs(float(k) - float(fpr_target)), float(v)))
            except Exception:
                pass

        if keys:
            keys.sort(key=lambda x: x[0])
            return float(keys[0][1])

    candidate_paths = [
        ["selected_operating_point", "threshold"],
        ["best_operating_point", "threshold"],
        ["official_selected_operating_point", "threshold"],
        ["val_selected_operating_point", "threshold"],
        ["summary", "selected_operating_point", "threshold"],
        ["summary", "best_operating_point", "threshold"],
        ["summary", "official_selected_operating_point", "threshold"],
        ["selected_threshold"],
        ["best_threshold"],
        ["best_thr"],
        ["threshold"],
        ["official_threshold"],
        ["summary", "selected_threshold"],
        ["summary", "best_threshold"],
        ["summary", "best_thr"],
        ["summary", "threshold"],
        ["summary", "official_threshold"],
    ]

    for path in candidate_paths:
        value = _get_nested(summary, path)
        if value is not None:
            return float(value)

    csv_thr = _extract_threshold_from_artifact_csv(
        summary=summary,
        fpr_target=fpr_target,
        summary_path=summary_path,
    )
    if csv_thr is not None:
        return float(csv_thr)

    raise ValueError(
        "Binary/anomaly threshold bulunamadı. "
        "Beklenen alanlar: thresholds_by_fpr_target veya "
        "selected_threshold / selected_operating_point.threshold "
        "veya artifacts içindeki operating-points CSV."
    )


class _ExplicitLabelMapper:
    """
    label_encoder olmayan bundle'lar için küçük adapter.
    label_encoder gibi inverse_transform ve classes_ sağlar.
    """

    def __init__(self, label_to_int: dict[str, int], int_to_label: dict[int, str]):
        self.label_to_int = {str(k): int(v) for k, v in label_to_int.items()}
        self.int_to_label = {int(k): str(v) for k, v in int_to_label.items()}

        ordered_ids = sorted(self.int_to_label.keys())
        self.classes_ = np.asarray(
            [self.int_to_label[i] for i in ordered_ids],
            dtype=str,
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values).astype(int).ravel()
        decoded = [self.int_to_label[int(v)] for v in arr]
        return np.asarray(decoded, dtype=str)


def _load_multiclass_artifacts(mc_bundle: dict) -> tuple[object, object]:
    if "pipeline" not in mc_bundle:
        raise ValueError("Multiclass bundle içinde 'pipeline' anahtarı bulunamadı.")

    multiclass_pipe = mc_bundle["pipeline"]

    if "label_encoder" in mc_bundle and hasattr(mc_bundle["label_encoder"], "inverse_transform"):
        return multiclass_pipe, mc_bundle["label_encoder"]

    if "label_to_int" in mc_bundle and "int_to_label" in mc_bundle:
        mapper = _ExplicitLabelMapper(
            label_to_int=mc_bundle["label_to_int"],
            int_to_label=mc_bundle["int_to_label"],
        )
        return multiclass_pipe, mapper

    raise ValueError(
        "Multiclass bundle içinde ne 'label_encoder' ne de "
        "'label_to_int'/'int_to_label' mapping'i bulundu."
    )


@dataclass
class IDSArtifacts:
    binary_pipe: object
    multiclass_pipe: object
    multiclass_label_encoder: object
    if_preprocess: object
    if_model: object


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

        binary_summary_path = Path(cfg_raw["paths"]["binary_summary"])
        iforest_summary_path = Path(cfg_raw["paths"]["iforest_summary"])

        bin_sum = _read_json(binary_summary_path)
        an_sum = _read_json(iforest_summary_path)

        bin_fpr_t = float(cfg_raw["thresholds"]["binary_fpr_target"])
        an_fpr_t = float(cfg_raw["thresholds"]["anomaly_fpr_target"])

        bin_thr_override = cfg_raw["thresholds"].get("binary_threshold_override")
        an_thr_override = cfg_raw["thresholds"].get("anomaly_threshold_override")

        binary_thr = (
            float(bin_thr_override)
            if bin_thr_override is not None
            else _get_thr_from_summary(
                summary=bin_sum,
                fpr_target=bin_fpr_t,
                summary_path=binary_summary_path,
            )
        )

        anomaly_thr = (
            float(an_thr_override)
            if an_thr_override is not None
            else _get_thr_from_summary(
                summary=an_sum,
                fpr_target=an_fpr_t,
                summary_path=iforest_summary_path,
            )
        )

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

        binary_pipe = joblib.load(Path(cfg_raw["paths"]["binary_pipeline"]))

        mc_bundle = joblib.load(Path(cfg_raw["paths"]["multiclass_bundle"]))
        multiclass_pipe, multiclass_label_encoder = _load_multiclass_artifacts(mc_bundle)

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
        X = self._features_only(df)
        proba = self.art.multiclass_pipe.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        pred_str = self.art.multiclass_label_encoder.inverse_transform(pred_idx)
        max_prob = np.max(proba, axis=1)
        return pred_str.astype(str), max_prob.astype(float), proba.astype(float)

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
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
        out = pd.DataFrame(index=df.index)

        bin_prob = self.predict_binary_prob(df)
        bin_pred = (bin_prob >= self.cfg.binary_thr).astype(int)

        an_score = self.anomaly_score(df)
        unknown_by_anom = (an_score > self.cfg.anomaly_thr).astype(int)

        out["binary_prob"] = bin_prob
        out["binary_pred"] = bin_pred
        out["anomaly_score"] = an_score
        out["unknown_by_anom"] = unknown_by_anom

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

        if proba is not None:
            k = min(int(topk), proba.shape[1])
            topk_idx = np.argsort(-proba, axis=1)[:, :k]
            classes = self.art.multiclass_label_encoder.classes_.astype(str)

            for j in range(k):
                cls_j = classes[topk_idx[:, j]]
                prob_j = np.take_along_axis(proba, topk_idx[:, j:j + 1], axis=1).ravel()

                tmp_cls = np.array([""] * len(df), dtype=object)
                tmp_prb = np.full(len(df), np.nan, dtype=float)

                tmp_cls[attack_idx] = cls_j
                tmp_prb[attack_idx] = prob_j

                out[f"mc_top{j+1}_class"] = tmp_cls
                out[f"mc_top{j+1}_prob"] = tmp_prb

        low_conf_attack = ((bin_pred == 1) & (mc_maxp < float(multiclass_conf_min))).astype(int)

        if mode == "anomaly_only":
            unknown = unknown_by_anom
        elif mode == "mc_conf_only":
            unknown = np.where(bin_pred == 0, unknown_by_anom, low_conf_attack).astype(int)
        elif mode == "union_or":
            unknown = (unknown_by_anom | low_conf_attack).astype(int)
        elif mode == "hybrid_and":
            unknown = np.where(
                bin_pred == 0,
                unknown_by_anom,
                (unknown_by_anom & low_conf_attack),
            ).astype(int)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out["unknown_pred"] = unknown
        out["mode"] = mode
        out["mc_conf_min"] = float(multiclass_conf_min)

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