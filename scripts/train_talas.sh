#!/bin/bash
# Bash script for TALAS training on Linux/Mac

python ../main.py \
  --method talas \
  --train_data_path "data/AllNLI.csv" \
  --student_model_name "bert-base-uncased" \
  --teacher_model_name "Qwen/Qwen3-Embedding-0.6B" \
  --save_dir "checkpoints/talas" \
  --batch_size 128 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --max_length 256 \
  --temperature 0.05
