#!/bin/bash

# Training script for Contextual Dynamic Mapping (CDM)
# This script trains a student model using CDM distillation method

echo "======================================"
echo "Training with CDM method"
echo "======================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs if available
export TOKENIZERS_PARALLELISM=false

# Training parameters
METHOD="cdm"
TRAIN_DATA="..\data\merged_3_data_5k_each.csv"
STUDENT_MODEL="..\model_hub\MiniLMv2-L6-H384-distilled-from-BERT-Base\MiniLM-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/cdm"

# Run training
python3 ../main.py \
    --method $METHOD \
    --train_data $TRAIN_DATA \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length $MAX_LENGTH \
    --save_dir $SAVE_DIR \
    --w_task 0.5 \
    --alpha_dtw 0.5 \
    --num_workers 2 \
    --debug

echo "======================================"
echo "Training completed!"
echo "======================================"
