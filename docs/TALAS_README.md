# TALAS (Teacher Anchor Learning with Cached Embeddings)

## Overview

TALAS is an efficient knowledge distillation method that uses **cached teacher embeddings** to reduce GPU memory usage and accelerate training. Instead of running the teacher model during training, TALAS:

1. **Pre-computes** teacher embeddings once and saves them to disk
2. **Trains** the student model using the cached embeddings (no teacher model in GPU memory)

## Benefits

- **Memory efficient**: Teacher model is freed after caching, enabling single-GPU training
- **Faster training**: No teacher inference during training
- **Auto-detection**: Automatically detects if cache exists and loads/creates as needed

## Quick Start

### Windows (PowerShell)
```powershell
.\scripts\train_talas.ps1
```

### Linux/Mac (Bash)
```bash
chmod +x scripts/train_talas.sh
./scripts/train_talas.sh
```

### Python Command
```bash
python main.py \
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
```

## Configuration

Key parameters in `config/talas_config.py`:

```python
cache_teacher = True                  # Enable teacher caching
cache_path = "cache/teacher_train.pt" # Where to save/load cache
pooling_method = "last_token"         # Pooling method for teacher
normalize_cache = False               # Whether to normalize cached embeddings
cache_dtype = "float32"               # Data type for cache

w_task = 0.5                          # Task loss weight
w_kd = 0.5                            # Knowledge distillation loss weight
```

## Workflow

### First Run (Cache Creation)
```
1. Load teacher model
2. Pre-compute teacher embeddings for entire dataset
3. Save cache to disk (cache/teacher_train.pt)
4. Free teacher model from GPU memory
5. Train student using cached embeddings
```

### Subsequent Runs (Cache Loading)
```
1. Detect existing cache
2. Load cached embeddings from disk
3. Train student using cached embeddings (no teacher model loaded)
```

## Architecture

- **Student Model**: bert-base-uncased (768-dim)
- **Teacher Model**: Qwen/Qwen3-Embedding-0.6B (896-dim, bfloat16)
- **Projection**: Optional linear layer (768 → 896) if dimensions differ
- **Pooling**: Last token pooling (matching teacher)
- **Loss**: MSE between normalized student and teacher embeddings + InfoNCE task loss

## Files Created

- `config/talas_config.py` - TALAS configuration
- `scripts/train_talas.ps1` - Windows training script
- `scripts/train_talas.sh` - Linux/Mac training script
- `cache/teacher_train.pt` - Cached teacher embeddings (auto-created)

## Technical Details

### Cache Structure
The cache file contains:
```python
{
    'teacher_cls': [tensor1, tensor2, ...],  # List of teacher embeddings
    'metadata': {
        'teacher_model': 'Qwen/Qwen3-Embedding-0.6B',
        'pooling_method': 'last_token',
        'normalized': False,
        'dtype': 'float32',
        'num_samples': N
    }
}
```

### Training Step
1. Student forward pass (2 sequences for contrastive learning)
2. Compute task loss (InfoNCE)
3. Extract cached teacher embedding from batch
4. Pool student embedding (last token)
5. Project student if needed
6. Normalize both embeddings
7. Compute MSE loss
8. Combined loss: `w_task * loss_task + w_kd * loss_kd`

## Comparison with Other Methods

| Method | Teacher Inference | GPU Memory | Training Speed |
|--------|------------------|------------|----------------|
| CDM    | Every step       | High       | Slow           |
| DSKD   | Every step       | High       | Slow           |
| EMO    | Every step       | High       | Slow           |
| Stella | Every step       | High       | Slow           |
| **TALAS** | **None (cached)** | **Low** | **Fast** |

## Tips

1. **Cache Location**: Ensure `cache/` directory has enough disk space
2. **Batch Size**: Can use larger batch size since teacher model is freed
3. **Cache Reuse**: Cache can be reused across multiple training runs
4. **Cache Invalidation**: Delete cache if changing teacher model or pooling method
5. **Single GPU**: TALAS is ideal for single-GPU setups due to low memory usage

## Example Output

```
Loading training data from: data/AllNLI.csv
Cache not found. Pre-computing teacher embeddings...
Cached 10000 teacher embeddings to cache/teacher_train.pt
Teacher model freed from GPU memory
Training samples: 10000
Training batches: 79

Epoch 1/5: 100%|█████| 79/79 [00:15<00:00, 5.2it/s, avg_loss=0.3421]
Epoch 2/5: 100%|█████| 79/79 [00:14<00:00, 5.5it/s, avg_loss=0.2156]
...
```

## Integration with Main System

TALAS is fully integrated into the unified training framework:
- Shares codebase with CDM, DSKD, EMO, Stella
- Uses same `main.py` entry point
- Supports all standard features (checkpointing, evaluation, etc.)
