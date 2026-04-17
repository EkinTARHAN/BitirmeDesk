from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def _clean_attack_cat(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _stratified_split(df: pd.DataFrame, y: pd.Series, val_size: float, seed: int):
    """Stratify mümkün değilse fallback random split."""
    # sınıf sayıları çok azsa stratify patlayabilir
    vc = y.value_counts()
    min_count = int(vc.min()) if len(vc) > 0 else 0
    stratify = y if min_count >= 2 else None

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df, val_df


def main():
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))

    seed = int(cfg["seed"])
    val_size = float(cfg["val_size"])
    holdout = str(cfg["holdout_attack_cat"]).strip()

    label_col = cfg["target_label"]
    attack_col = cfg["target_attack_cat"]
    cat_cols = list(cfg["categorical_cols"])
    drop_cols = list(cfg.get("drop_cols", []))

    train_path = Path(cfg["paths"]["train_csv"])
    test_path = Path(cfg["paths"]["test_csv"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Temizlik
    train[attack_col] = _clean_attack_cat(train[attack_col])
    test[attack_col] = _clean_attack_cat(test[attack_col])

    # id vb drop (varsa)
    train = _safe_drop(train, drop_cols)
    test = _safe_drop(test, drop_cols)

    # Hold-out (Worms) çıkar
    train_known = train[train[attack_col].str.lower() != holdout.lower()].copy()
    train_holdout = train[train[attack_col].str.lower() == holdout.lower()].copy()

    # Testte unknown flag
    test = test.copy()
    test["is_unknown_target"] = (test[attack_col].str.lower() == holdout.lower()).astype(int)

    # ========== BINARY SPLITS ==========
    y_bin = train_known[label_col]
    bin_train, bin_val = _stratified_split(train_known, y_bin, val_size, seed)

    bin_test = test.copy()

    # ========== MULTICLASS SPLITS (only attacks) ==========
    train_known_attacks = train_known[train_known[label_col] == 1].copy()
    y_mc = train_known_attacks[attack_col]
    mc_train, mc_val = _stratified_split(train_known_attacks, y_mc, val_size, seed)

    mc_test = test[test[label_col] == 1].copy()  # sadece attack satırları
    # mc_test içinde Worms zaten is_unknown_target=1 olacak

    # Kaydet
    bin_train.to_csv(out_dir / "binary_train.csv", index=False)
    bin_val.to_csv(out_dir / "binary_val.csv", index=False)
    bin_test.to_csv(out_dir / "binary_test.csv", index=False)

    mc_train.to_csv(out_dir / "multiclass_train.csv", index=False)
    mc_val.to_csv(out_dir / "multiclass_val.csv", index=False)
    mc_test.to_csv(out_dir / "multiclass_test.csv", index=False)

    summary = {
        "holdout_attack_cat": holdout,
        "raw_train_shape": list(train.shape),
        "raw_test_shape": list(test.shape),
        "train_known_shape": list(train_known.shape),
        "train_holdout_shape": list(train_holdout.shape),
        "binary_train_shape": list(bin_train.shape),
        "binary_val_shape": list(bin_val.shape),
        "binary_test_shape": list(bin_test.shape),
        "multiclass_train_shape": list(mc_train.shape),
        "multiclass_val_shape": list(mc_val.shape),
        "multiclass_test_shape": list(mc_test.shape),
        "test_unknown_count": int(test["is_unknown_target"].sum()),
        "categorical_cols": cat_cols,
        "drop_cols_applied": [c for c in drop_cols if c in train.columns]  # not perfect but ok
    }

    (out_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Konsola kısa özet
    print("\n=== SPLIT SUMMARY ===")
    for k, v in summary.items():
        if "shape" in k or "count" in k or k == "holdout_attack_cat":
            print(f"{k}: {v}")

    # kontrol: Worms train'e girmedi mi?
    if attack_col in bin_train.columns:
        worms_in_train = (bin_train[attack_col].str.lower() == holdout.lower()).sum()
        print("worms_in_binary_train:", int(worms_in_train))


if __name__ == "__main__":
    main()