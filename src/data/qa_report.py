import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/unsw_nb15")

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH  = RAW_DIR / "UNSW_NB15_testing-set.csv"

def basic_report(df: pd.DataFrame, name: str):
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)

    # hedef kolonlar
    for col in ["label", "attack_cat"]:
        if col in df.columns:
            print(f"{col} value counts (top 15):")
            print(df[col].value_counts(dropna=False).head(15))
        else:
            print(f"WARNING: {col} not found!")

    # NaN / Inf
    nan_total = df.isna().sum().sum()
    print("Total NaNs:", int(nan_total))

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 0:
        inf_total = np.isinf(numeric.to_numpy()).sum()
        print("Total Inf in numeric:", int(inf_total))
        # basit özet
        print("Numeric columns:", numeric.shape[1])
    else:
        print("No numeric columns detected.")

    # muhtemel kategorikler
    cat_candidates = [c for c in ["proto", "service", "state"] if c in df.columns]
    if cat_candidates:
        for c in cat_candidates:
            print(f"{c} unique:", df[c].nunique(dropna=False))
    else:
        print("proto/service/state not found (ok, will verify later).")

def worms_check(df: pd.DataFrame, name: str):
    if "attack_cat" not in df.columns:
        return
    worms = df["attack_cat"].astype(str).str.strip().str.lower().eq("worms")
    print(f"[{name}] Worms count:", int(worms.sum()))

def main():
    # düşük RAM için read_csv optimize: ilk etapta tam oku, gerekirse chunk'a geçeriz
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    basic_report(train, "TRAIN")
    worms_check(train, "TRAIN")

    basic_report(test, "TEST")
    worms_check(test, "TEST")

    # benign/attack hızlı kontrol
    if "label" in train.columns and "label" in test.columns:
        print("\nLabel distribution (%):")
        print("TRAIN:", (train["label"].value_counts(normalize=True)*100).round(2).to_dict())
        print("TEST :", (test["label"].value_counts(normalize=True)*100).round(2).to_dict())

if __name__ == "__main__":
    main()