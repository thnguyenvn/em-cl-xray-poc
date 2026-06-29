# EM-CL-Xray PoC

**Expectation-Maximization Guided Continual Learning for Chest X-ray Classification**

This repository contains a Proof-of-Concept (PoC) for studying how **Expectation-Maximization (EM)** can be integrated into **Continual Learning (CL)** for chest X-ray image classification using **CheXpert-v1.0-small** and **NIH Chest X-rays / ChestX-ray14**.

## 1. Research Motivation

Deep learning models for medical X-ray analysis are usually trained on static datasets. In real clinical settings, however, data distributions may change over time due to hospital-specific acquisition protocols, imaging devices, preprocessing pipelines, patient populations, and annotation uncertainty.

This PoC investigates whether EM can help continual learning models:

- reduce catastrophic forgetting,
- handle uncertain or weak labels,
- construct distribution-aware replay memory,
- improve adaptation under cross-dataset/domain shift.

## 2. Datasets

This PoC uses two chest X-ray datasets:

| Dataset | Role |
|---|---|
| CheXpert-v1.0-small | Main dataset for uncertainty-aware learning and task-incremental CL |
| NIH Chest X-rays / ChestX-ray14 | Cross-dataset/domain-shift evaluation |

Dataset files are **not stored in this repository**. They should be attached to Kaggle notebooks as Kaggle datasets.

Example Kaggle paths:

```python
CHEXPERT_ROOT = "/kaggle/input/chexpert-v10-small"
NIH_ROOT = "/kaggle/input/nih-chest-xrays"
OUTPUT_DIR = "/kaggle/working/outputs"
```

## 3. PoC Research Questions

1. How severe is catastrophic forgetting when training sequentially on chest X-ray tasks?
2. Does replay reduce forgetting compared with naive fine-tuning?
3. Can EM improve replay memory by selecting representative samples based on feature distributions?
4. Can EM help use uncertain labels in CheXpert through pseudo-labeling?
5. Does EM-guided memory improve transfer from CheXpert to NIH Chest X-rays?

## 4. Methods

The PoC compares the following methods:

| Method | Description |
|---|---|
| Fine-tuning | Sequential training without forgetting mitigation |
| Random Replay | Stores random samples from previous tasks |
| EWC / LwF | Regularization-based continual learning baselines |
| EM Pseudo-labeling | Uses EM to estimate soft labels for uncertain samples |
| EM-guided Replay | Uses EM/GMM over feature vectors to select replay samples |
| Prototype EM Replay | Maintains class/task prototypes updated by EM-style soft assignment |

## 5. Repository Structure

```text
em-cl-xray-poc/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── chexpert.yaml
│   ├── nih.yaml
│   └── poc_em_cl.yaml
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── continual/
│   ├── em/
│   ├── metrics/
│   └── utils/
│
├── notebooks/
│   ├── 01_prepare_data.ipynb
│   ├── 02_finetune_baseline.ipynb
│   ├── 03_replay_baseline.ipynb
│   ├── 04_em_pseudolabel.ipynb
│   ├── 05_em_guided_replay.ipynb
│   ├── 06_chexpert_to_nih_shift.ipynb
│   └── 07_report_figures.ipynb
│
├── scripts/
│   ├── run_finetune.py
│   ├── run_replay.py
│   ├── run_em_pseudolabel.py
│   ├── run_em_memory.py
│   └── run_all_poc.sh
│
├── outputs/
│   ├── metrics/
│   ├── figures/
│   └── checkpoints/
│
└── docs/
    ├── experiment_protocol.md
    ├── kaggle_setup.md
    └── results_template.md
```

## 6. Kaggle Setup

Clone the repository in a Kaggle notebook:

```bash
git clone https://github.com/<your-username>/em-cl-xray-poc.git
cd em-cl-xray-poc
pip install -r requirements.txt
```

Check GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 7. Example Commands

Fine-tuning baseline:

```bash
python scripts/run_finetune.py \
  --config configs/poc_em_cl.yaml \
  --dataset chexpert
```

Random replay baseline:

```bash
python scripts/run_replay.py \
  --config configs/poc_em_cl.yaml \
  --memory_size 1000
```

EM-guided replay:

```bash
python scripts/run_em_memory.py \
  --config configs/poc_em_cl.yaml \
  --memory_size 1000 \
  --em_components 5
```

## 8. Evaluation Metrics

Classification metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1-score

Continual learning metrics:

- Final Average Accuracy
- Average Forgetting
- Backward Transfer
- Accuracy Matrix

EM-related metrics:

- Log-likelihood
- BIC
- Cluster purity
- Prototype drift

## 9. Expected Outputs

The experiments should generate:

```text
outputs/
├── metrics/
│   ├── accuracy_matrix.csv
│   ├── continual_metrics.json
│   └── classification_report.csv
│
├── figures/
│   ├── accuracy_heatmap.png
│   ├── forgetting_curve.png
│   └── auc_comparison.png
│
└── checkpoints/
    └── model_task_*.pt
```

## 10. Research Contribution Direction

The target contribution of this PoC is not only to compare existing CL methods, but to validate a concrete methodological idea:

> EM can act as an adaptive distribution-estimation mechanism for continual learning memory in chest X-ray classification.

Possible paper title:

**Distribution-Aware EM Replay for Continual Chest X-ray Classification under Cross-Dataset Domain Shift**

## 11. License

For academic research use only. Dataset licenses and access terms must be respected separately.
