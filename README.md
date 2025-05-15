# Classifier-Chains-Experiment-IVT
# 🔬 Experiment- Classifier Chains for Surgical Verb Classification (CholecT45)

This repository explores the **Classifier Chains** method for mitigating **multi-label class imbalance** in surgical verb classification using the CholecT45 dataset. Each instrument (e.g., grasper, bipolar) is used to filter the dataset and train a dedicated ResNet-based model.

---

## 📌 Motivation

The CholecT45 dataset suffers from **severe verb imbalance**—verbs like `grasp`, `dissect`, and `retract` dominate, while others like `aspirate`, `clip`, and `pack` are extremely rare.

Our idea:
> Split the dataset based on **active instruments** and train a **separate classifier for each**, only on the verbs that instrument can perform.

---

## 🧪 Methodology

### ➕ Instrument-Specific Verb Mapping

| **Instrument** | **Verbs** |
| --- | --- |
| **grasper** | dissect, grasp, pack, retract, null_verb |
| **bipolar** | coagulate, dissect, grasp, retract, null_verb |
| **hook** | coagulate, cut, dissect, retract, null_verb |
| **scissors** | coagulate, cut, dissect, null_verb |
| **clipper** | clip, null_verb |
| **irrigator** | aspirate, dissect, irrigate, retract, null_verb |

Each model is trained with:
- ✅ `ResNet18` backbone
- 🎯 `BCEWithLogitsLoss`
- ⚙️ Optimizer: `Adam(lr=0.001)`
- 🔄 Trained for `50 epochs`

---

## 📉 Results (Per Instrument Classifier)

| **Model** | **Accuracy** | Grasp | Retract | Dissect | Coagulate | Clip | Cut | Aspirate | Irrigate | Pack | Null_verb |
|----------|--------------|-------|---------|---------|-----------|------|-----|----------|----------|------|------------|
| grasper | 50.58 | 59.51 | 74.00 | 79.2 | 75.7 | 77.21 | 42.18 | 58.0 | 3.08 | 14.62 | 22.28 |
| bipolar | 58.94 | 39.46 | 40.83 | 83.55 | 90.13 | - | - | - | - | - | 40.7 |
| hook | 49.33 | 60.16 | 80.36 | 84.66 | 3.1 | - | - | - | - | - | 18.37 |
| scissors | 56.97 | 85.93 | 67.93 | 2.88 | - | - | 94.01 | - | - | - | - |
| clipper | 55.45 | 83.94 | 46.10 | - | - | 85.25 | - | - | - | 6.5 | 6.5 |
| irrigator | 37.14 | - | 52.84 | 47.07 | - | - | - | - | 52.81 | 19.09 | - |

---

## 🧠 Observations

1. ✅ **Faster inference** due to fewer output classes per model.
2. ❌ **No significant improvement in accuracy or confidence** over the baseline ResNet model trained on all verbs.
3. 🔁 **Tail classes like `aspirate`, `pack`, and `null_verb`** remained poorly predicted across all models.
4. ⚡ Some models performed **extremely well on dominant verbs** (e.g., *clip* with Clipper, *cut* with Scissors).
5. 📉 **Class imbalance remains unsolved** — classifier chaining alone is insufficient.

---

```markdown
![image](https://github.com/user-attachments/assets/8b037b8f-27aa-4f15-9b2f-12d1f961ffe8)
![image.png](attachment:0cf5f13a-a837-4b46-a165-f18a41c84671:image.png)
![image.png](attachment:01eb6cb0-e251-4677-9526-dd278cd5cb52:image.png)
![image.png](attachment:b4605916-60e3-43de-9722-beb25ad1d6de:image.png)
![image.png](attachment:ee15aa6c-743a-4d35-bba6-281a48b8a0d2:image.png)
![image.png](attachment:a0f09c4a-cbba-4468-a670-ec7d30472d64:image.png)
![image.png](attachment:abc8a90b-79de-4295-accb-c14c8f783e46:image.png)
