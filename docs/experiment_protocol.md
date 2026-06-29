# Experiment Protocol

## Goal

Evaluate whether EM can improve continual learning for chest X-ray classification.

## Experimental Scenarios

1. CheXpert task-incremental learning.
2. CheXpert uncertainty-aware pseudo-labeling.
3. CheXpert to NIH cross-dataset domain shift.
4. Replay memory comparison: random replay vs EM-guided replay.

## Baselines

- Fine-tuning
- Random Replay
- EWC or LwF

## Proposed Methods

- EM pseudo-labeling
- EM-guided replay memory
- Prototype-based EM replay

## Main Metrics

- AUC
- Accuracy
- F1-score
- Final Average Accuracy
- Average Forgetting
- Backward Transfer
- BIC / log-likelihood for EM
