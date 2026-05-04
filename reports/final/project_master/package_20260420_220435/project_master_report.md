# Project Master Results Package

## 1. Project structure

The project consists of an explainable hybrid IDS built primarily on UNSW-NB15, supported by two additional independent benchmarks: CIC-IDS2017 and CIC-ToN-IoT.

## 2. Official UNSW decisions

- Official binary main model: **XGBoost Tuned v2** (test PR-AUC = **0.989072**, test ROC-AUC = **0.985146**, test F1 = **0.932448**, test FPR = **0.017946**).
- Official binary backup model: **LightGBM Main**.
- Official multiclass main model: **Random Forest**.
- Official multiclass backup model: **LightGBM**.

## 3. CIC-IDS2017 methodological takeaway

CIC-IDS2017 showed that row-wise results can be overly optimistic. The file-based benchmark was much harder (test PR-AUC = **0.468943**, test ROC-AUC = **0.834459**, test F1 = **0.129533**), supporting the thesis that split strategy strongly affects IDS conclusions.

## 4. CIC-ToN-IoT benchmark takeaway

- Temporal benchmark test PR-AUC: **0.984050**
- Temporal benchmark test ROC-AUC: **0.993641**
- Temporal benchmark test F1: **0.950091**
- Temporal benchmark test FPR: **0.006375**
- Seen attack detection rate: **0.999984**
- Unseen attack detection rate: **0.646661**

This benchmark supports the same high-level story observed on CIC-IDS2017: ranking/AUC can remain strong while operational behavior becomes harder under temporal drift and unseen attacks.

## 5. CIC-ToN-IoT unseen attack-family notes

- **ddos**: detection rate = **0.980198** (n = 202)
- **dos**: detection rate = **0.979310** (n = 145)
- **mitm**: detection rate = **0.808511** (n = 517)
- **ransomware**: detection rate = **0.677717** (n = 5,098)
- **backdoor**: detection rate = **0.633487** (n = 27,145)

Unseen-family generalization is not uniform: some families such as ddos/dos remain highly detectable, whereas backdoor and ransomware are harder.

## 6. Final thesis positioning

The final thesis can therefore position the work as a hybrid IDS project that combines model selection, explainability, operating-point analysis, and benchmark methodology. UNSW-NB15 remains the main dataset, while CIC-IDS2017 and CIC-ToN-IoT serve as complementary independent benchmarks that validate the methodological conclusions.