# Final Model Recommendation

## Binary Final Decision
- Conservative target FPR choice: **0.005**
- Main binary model: **XGBoost Tuned v2** (`xgboost_tuned_v2`)
  - threshold: `0.940000`
  - test_f1: `0.932448`, test_fpr: `0.017946`, test_pr_auc: `0.989072`
- Backup binary model: **LightGBM Main** (`lgbm_main`)
  - threshold: `0.950258`
  - test_f1: `0.932044`, test_fpr: `0.029486`, test_pr_auc: `0.988060`

## Multiclass Final Decision
- Main multiclass model: **Multiclass Random Forest** (`multiclass_rf`)
  - test_macro_f1_known_only: `0.539542`, test_weighted_f1_known_only: `0.790918`
- Backup multiclass model: **Multiclass LightGBM** (`multiclass_lgbm`)
  - test_macro_f1_known_only: `0.531506`, test_weighted_f1_known_only: `0.786532`

## Notes
- Binary final choice is based on the conservative operating point table.
- Multiclass evaluation remains known-only to preserve the Worms hold-out open-set design.
- Other models stay in the project as comparison / baseline / literature reference models.