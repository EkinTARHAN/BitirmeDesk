from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.ids.score import IDSSystem


def main():
    cfg = yaml.safe_load(Path("configs/ids_system.yaml").read_text(encoding="utf-8"))
    ids = IDSSystem.from_yaml("configs/ids_system.yaml")

    test_path = Path(cfg["paths"]["test_csv"])
    df = pd.read_csv(test_path)

    cache_dir = Path("reports/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sunum için stabil mod: anomaly_only
    scored_anom = ids.score(df, mode="anomaly_only", multiclass_conf_min=0.6, topk=3)
    out_anom = pd.concat([df.reset_index(drop=True), scored_anom.reset_index(drop=True)], axis=1)
    out_anom.to_csv(cache_dir / "test_scored_anomaly_only.csv.gz", index=False, compression="gzip")

    # 2) Trade-off göstermek için agresif mod örneği: union_or + conf=0.9
    scored_union = ids.score(df, mode="union_or", multiclass_conf_min=0.9, topk=3)
    out_union = pd.concat([df.reset_index(drop=True), scored_union.reset_index(drop=True)], axis=1)
    out_union.to_csv(cache_dir / "test_scored_union_or_conf0p9.csv.gz", index=False, compression="gzip")

    print("\nSaved demo caches:")
    print(" -", cache_dir / "test_scored_anomaly_only.csv.gz")
    print(" -", cache_dir / "test_scored_union_or_conf0p9.csv.gz")


if __name__ == "__main__":
    main()