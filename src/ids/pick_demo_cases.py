from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def pick_one(df: pd.DataFrame, mask: np.ndarray, sort_by: str | None = None, ascending: bool = True) -> pd.Series:
    sub = df[mask].copy()
    if len(sub) == 0:
        raise RuntimeError("No rows matched the mask.")
    if sort_by is not None and sort_by in sub.columns:
        sub = sub.sort_values(by=sort_by, ascending=ascending)
    return sub.iloc[0]


def main():
    cache_dir = Path("reports/cache")
    demo_dir = Path("reports/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Stabil mod cache
    path_anom = cache_dir / "test_scored_anomaly_only.csv.gz"
    if not path_anom.exists():
        raise FileNotFoundError(f"Cache not found: {path_anom}. Run build_demo_cache first.")

    df = pd.read_csv(path_anom)

    # Kolonlar (veride var)
    label_col = "label"
    unknown_target_col = "is_unknown_target"     # Worms = 1
    unknown_pred_col = "unknown_pred"
    bin_pred_col = "binary_pred"
    an_score_col = "anomaly_score"

    # 1) Benign örnek (unknown değil, düşük anomaly)
    benign_mask = (
        (df[label_col] == 0) &
        (df[unknown_pred_col] == 0) &
        (df[bin_pred_col] == 0)
    ).to_numpy()

    benign_case = pick_one(df, benign_mask, sort_by=an_score_col, ascending=True)

    # 2) Known attack örnek (unknown değil, binary attack, multiclass var)
    known_attack_mask = (
        (df[label_col] == 1) &
        (df[unknown_target_col] == 0) &
        (df[unknown_pred_col] == 0) &
        (df[bin_pred_col] == 1)
    ).to_numpy()

    # burada “güvenli” bir case: mc_max_prob yüksek olanı seçelim
    if "mc_max_prob" in df.columns:
        ka = df[known_attack_mask].copy()
        if len(ka) == 0:
            raise RuntimeError("No known-attack rows matched for demo.")
        ka = ka.sort_values(by="mc_max_prob", ascending=False)
        known_attack_case = ka.iloc[0]
    else:
        known_attack_case = pick_one(df, known_attack_mask, sort_by=an_score_col, ascending=True)

    # 3) Worms (unknown_target=1) ve sistem unknown_pred=1 yakalamış olsun
    worms_unknown_mask = (
        (df[unknown_target_col] == 1) &
        (df[unknown_pred_col] == 1)
    ).to_numpy()

    if worms_unknown_mask.sum() > 0:
        worms_case = pick_one(df, worms_unknown_mask, sort_by=an_score_col, ascending=False)
        worms_note = "worms_found_as_unknown"
    else:
        # Eğer anomaly_only modunda hiç yakalanmadıysa: en anomali Worms'u seçip 'miss' diye işaretle
        worms_mask = (df[unknown_target_col] == 1).to_numpy()
        worms_case = pick_one(df, worms_mask, sort_by=an_score_col, ascending=False)
        worms_note = "worms_not_caught_in_this_mode_pick_top_anomaly"

    demo_cases = pd.DataFrame([
        benign_case,
        known_attack_case,
        worms_case
    ])

    demo_cases.insert(0, "demo_case", ["benign", "known_attack", worms_note])

    out_path = demo_dir / "demo_cases.csv"
    demo_cases.to_csv(out_path, index=False)

    # Batch demo için küçük bir CSV üretelim (100 satır):
    # 40 benign + 40 known attack + tüm worms (44)
    benign_rows = df[df[label_col] == 0].sample(n=40, random_state=42)
    known_attack_rows = df[(df[label_col] == 1) & (df[unknown_target_col] == 0)].sample(n=40, random_state=42)
    worms_rows = df[df[unknown_target_col] == 1]

    demo_batch = pd.concat([benign_rows, known_attack_rows, worms_rows], axis=0).reset_index(drop=True)
    batch_path = demo_dir / "demo_batch.csv"
    demo_batch.to_csv(batch_path, index=False)

    print("\nSaved demo selections:")
    print(" -", out_path)
    print(" -", batch_path)
    print("\nDemo cases labels:")
    print(demo_cases[["demo_case", "label", "attack_cat", "is_unknown_target", "binary_pred", "unknown_pred", "anomaly_score"]])


if __name__ == "__main__":
    main()