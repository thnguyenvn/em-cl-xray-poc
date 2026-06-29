#!/usr/bin/env bash
set -e

python scripts/run_finetune.py --config configs/poc_em_cl.yaml --dataset chexpert
python scripts/run_replay.py --config configs/poc_em_cl.yaml --memory_size 1000
python scripts/run_em_pseudolabel.py --config configs/poc_em_cl.yaml --threshold 0.7
python scripts/run_em_memory.py --config configs/poc_em_cl.yaml --memory_size 1000 --em_components 5
