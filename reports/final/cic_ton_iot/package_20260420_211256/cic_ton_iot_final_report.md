# CIC-ToN-IoT Final Results Package

## 1. Dataset snapshot

- Row count: **5,351,760**
- Column count: **85**
- Feature count after prep: **79**
- Total inf cleaned during prep: **1,177**
- Binary label column: **Label** was used together with derived binary target.

## 2. Binary benchmark summary

- Stratified baseline test PR-AUC: **0.999881**, test ROC-AUC: **0.999866**, test F1: **0.994643**, test FPR: **0.009147**.
- Temporal benchmark test PR-AUC: **0.984050**, test ROC-AUC: **0.993641**, test F1: **0.950091**, test FPR: **0.006375**.

## 3. Main methodological interpretation

The stratified split produced near-perfect results and therefore should be interpreted as an optimistic baseline. The temporal split is more realistic because the data are ordered chronologically and the test period includes attack families that are not present in the training set.

In the temporal benchmark, the model still preserved strong ranking performance (test PR-AUC = **0.984050**, test ROC-AUC = **0.993641**), but the performance was lower than the stratified baseline, indicating temporal/domain shift.

## 4. Seen vs unseen attack behavior

- Seen attack detection rate (best-val-F1 threshold): **0.999984**
- Unseen attack detection rate (best-val-F1 threshold): **0.646661**
- Benign false-positive rate (best-val-F1 threshold): **0.006375**

This shows that the temporal test set behaves not only as a chronological split, but also as a partial unseen-attack stress scenario. The model generalizes almost perfectly to seen attack families, while performance drops on unseen families.

## 5. Attack-family observations at best-val-F1 threshold

### Seen attacks

- **injection**: detection rate = **1.000000** (n = 11,014)
- **password**: detection rate = **1.000000** (n = 15,364)
- **xss**: detection rate = **0.999990** (n = 102,431)
- **scanning**: detection rate = **0.965517** (n = 29)

### Unseen attacks

- **ddos**: detection rate = **0.980198** (n = 202)
- **dos**: detection rate = **0.979310** (n = 145)
- **mitm**: detection rate = **0.808511** (n = 517)
- **ransomware**: detection rate = **0.677717** (n = 5,098)
- **backdoor**: detection rate = **0.633487** (n = 27,145)

Among unseen families, **ddos** and **dos** remained highly detectable, while **backdoor** and **ransomware** were harder. This indicates that unseen-attack generalization is family-dependent rather than uniformly strong or weak.

## 6. Final positioning in the thesis

CIC-ToN-IoT should be positioned as a third independent benchmark. It supports the same main conclusion observed on CIC-IDS2017: strong AUC/ranking performance does not guarantee stable operational behavior under more realistic split conditions, especially when temporal shift and unseen attack families are present.