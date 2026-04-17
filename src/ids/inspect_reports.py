from __future__ import annotations

from pathlib import Path
import pandas as pd


def _print_head(path: Path, n: int = 10):
    print(f"\n--- {path.as_posix()} (head {n}) ---")
    if not path.exists():
        print("NOT FOUND")
        return
    df = pd.read_csv(path)
    print(df.head(n).to_string(index=False))


def _print_binary_threshold_table(path: Path):
    print(f"\n=== BINARY THRESHOLD TABLE: {path.as_posix()} ===")
    if not path.exists():
        print("NOT FOUND")
        return
    df = pd.read_csv(path)

    # daha okunur kolon seçimi
    cols = [c for c in [
        "split", "fpr_target", "threshold", "fpr", "fnr", "precision", "recall", "f1", "fp", "tn", "tp", "fn"
    ] if c in df.columns]
    if cols:
        df2 = df[cols].copy()
    else:
        df2 = df

    # val/test ayrı görünsün
    df2 = df2.sort_values(by=["fpr_target", "split"])
    print(df2.to_string(index=False))


def _print_iforest_table(path: Path):
    print(f"\n=== IFORREST OPEN-SET TABLE: {path.as_posix()} ===")
    if not path.exists():
        print("NOT FOUND")
        return
    df = pd.read_csv(path)

    cols = [c for c in [
        "fpr_target", "threshold",
        "benign_fpr_test",
        "worms_detection_rate_test",
        "known_attack_unknown_rate_test",
        "test_unknown_count",
        "test_benign_count",
        "test_known_attack_count",
    ] if c in df.columns]
    if cols:
        df2 = df[cols].copy()
    else:
        df2 = df
    df2 = df2.sort_values(by=["fpr_target"])
    print(df2.to_string(index=False))


def _print_ids_system_table(path: Path):
    print(f"\n=== IDS SYSTEM OPEN-SET TABLE: {path.as_posix()} ===")
    if not path.exists():
        print("NOT FOUND")
        return
    df = pd.read_csv(path)

    cols = [c for c in [
        "mode", "mc_conf_min",
        "benign_unknown_rate",
        "worms_detection_rate",
        "known_attack_unknown_rate",
        "sys_known_attack_coverage",
        "sys_mc_macro_f1",
        "sys_mc_weighted_f1",
        "binary_thr_used",
        "anomaly_thr_used",
    ] if c in df.columns]
    if cols:
        df2 = df[cols].copy()
    else:
        df2 = df

    print(df2.to_string(index=False))


def main():
    base = Path("reports/tables")

    binary_thr = base / "binary_lgbm_threshold_eval.csv"
    if_eval = base / "anomaly_iforest_open_set_eval.csv"
    ids_eval = base / "ids_system_open_set_eval.csv"

    _print_binary_threshold_table(binary_thr)
    _print_iforest_table(if_eval)
    _print_ids_system_table(ids_eval)

    # ayrıca dosyaların ilk satırlarını da görmek istersen:
    # _print_head(binary_thr, 6)
    # _print_head(if_eval, 6)
    # _print_head(ids_eval, 12)


if __name__ == "__main__":
    main()