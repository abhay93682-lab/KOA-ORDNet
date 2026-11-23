# Results

All numbers come from `evaluate.py` on checkpoints trained with `notebooks/koa_ordnet_colab.ipynb`
(Google Colab, NVIDIA T4, PyTorch). Raw output: [`results.json`](results.json).

- **In-distribution test set:** Chen (2018) Knee Osteoarthritis Severity Grading dataset, `test/` split, n = 1656.
- **External test set (RG3):** Digital Knee X-ray Images (Gornale & Patravali, 2020, doi:10.17632/t9ndx37v5h.1),
  labels from Medical Expert I, n = 1650. Never used for training, tuning or model selection.
- **Model selection:** each model's checkpoint is the epoch with the best validation macro-F1. For KOA-ORDNet,
  three focal-γ settings were trained (γ = 2, 1, 0) and the one with the best validation macro-F1 was kept
  (γ = 2: 0.6606, γ = 0: 0.6484, γ = 1: 0.6447). The test sets played no part in any choice.

## Table II — In-distribution test (n = 1656)

| Model | Acc. | Macro-F1 | QWK | MAE | Grade 3–4 Recall | Grad-CAM IoU |
|---|---|---|---|---|---|---|
| ResNet-50 + Cross-Entropy (baseline) | 64.01% | 0.6439 | 0.8056 | 0.4209 | 79.56% | 0.133 |
| ResNet-50 + Ordinal Loss (Chen-style, reproduced) | **66.12%** | **0.6646** | **0.8286** | **0.3859** | 77.37% | 0.127 |
| KOA-ORDNet (proposed, γ = 2) | 60.81% | 0.6308 | 0.8090 | 0.4336 | **79.56%** | 0.118 |

Per-grade recall (KL 0 / 1 / 2 / 3 / 4):

| Model | KL 0 | KL 1 | KL 2 | KL 3 | KL 4 |
|---|---|---|---|---|---|
| CE baseline | 79.2% | 31.1% | 58.8% | 72.6% | 72.5% |
| Ordinal baseline | 86.4% | 31.4% | 57.0% | 69.5% | 78.4% |
| KOA-ORDNet | 73.6% | **47.0%** | 45.6% | 69.5% | 76.5% |

## Table III — Cross-dataset generalization (external, n = 1650)

| Model | In-dist. Acc. | In-dist. QWK | External Acc. | External QWK | External Grade 3–4 Recall |
|---|---|---|---|---|---|
| CE baseline | 64.01% | 0.8056 | 29.82% | 0.4919 | 87.59% |
| Ordinal baseline | 66.12% | 0.8286 | 33.82% | 0.5766 | 68.62% |
| KOA-ORDNet | 60.81% | 0.8090 | 28.24% | 0.5052 | 98.36% |

## What the results show

- **KOA-ORDNet does not beat the baselines** on accuracy, macro-F1, QWK or MAE in this run. The plain ordinal
  loss (Chen-style) is the strongest model on these metrics.
- The class-weighted focal-ordinal loss does what it is designed to do for minority grades in part:
  KL 1 recall rises from ~31% to 47%, and Grade 3–4 recall matches the CE baseline (79.6%). This comes at the cost
  of KL 0 and KL 2 recall.
- **All models drop sharply on the external dataset** (QWK 0.81–0.83 → 0.49–0.58), with predictions shifted towards
  higher grades. The external images are whole-knee radiographs from a different scanner and population, while the
  training images are joint-centred crops, so this is the generalization gap RG3 warns about.
- **Grad-CAM IoU with the fixed joint region is low (0.12–0.13)** for all models. Either the networks attend to
  regions beyond the joint band, or the fixed box is a poor proxy for the joint margin; per-image joint annotations
  would be needed to tell these apart.
