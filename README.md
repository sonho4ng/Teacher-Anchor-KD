# Teacher Anchor KD - Knowledge Distillation Framework

A comprehensive knowledge distillation framework for NLP models, featuring multiple state-of-the-art distillation methods including our novel Teacher Anchor KD approach.

## Tổng quan

Project này implement 5 phương pháp Knowledge Distillation hiện đại:

### 1. **Teacher Anchor KD (TALAS)** ⭐
Our novel approach combining:
- **Teacher-anchored distillation**: Align student layers with cached teacher embeddings
- **Structural loss**: Preserve layer-wise representation consistency
- **SAM optimizer**: Sharpness-Aware Minimization for better generalization
- **Efficient caching**: Pre-compute teacher embeddings to reduce memory usage

### 2. **Dual Space Knowledge Distillation (DSKD)**
- Dual-space alignment (sequence-level + CLS-level)
- Learnable projection heads
- DTW-based sequence alignment

### 3. **Contextual Dynamic Mapping (CDM)**
- Token-level alignment with DTW
- Context-aware mapping between teacher and student
- Special token handling for different tokenizers

### 4. **EMO Embedding Distillation**
- CKA-based attention alignment
- Sinkhorn Optimal Transport loss
- Token importance projection
- Per-batch processing for accurate alignment

### 5. **Stella Distillation**
- Two-stage training (fc1 → full model)
- Matryoshka representation learning
- Multi-scale similarity preservation

## Cấu trúc Project

```
Teacher-Anchor-KD/
├── src/
│   ├── data_utils/              # Dataset & Data loaders
│   │   ├── dataset.py           # TextPairRaw, DualTokenizerCollate
│   │   └── dataset_cache.py     # TextPairWithTeacher (for TALAS)
│   ├── criterions/              # Knowledge Distillation methods
│   │   ├── teacher_anchor_kd.py      # TALAS implementation
│   │   ├── dual_space_kd.py          # DSKD implementation
│   │   ├── contextual_dynamic_mapping.py  # CDM implementation
│   │   ├── emo_embedding_distillation.py  # EMO implementation
│   │   └── stella_distillation.py    # Stella implementation
│   ├── evaluation/              # Evaluation metrics & tasks
│   │   ├── evaluation_automodel.py
│   │   └── evaluation_model_define.py
│   ├── loss.py                  # Loss functions
│   ├── pooling.py               # Pooling utilities
│   └── cache_teacher.py         # Teacher embedding caching
├── config/                      # Configuration files
│   ├── talas_config.py          # TALAS configuration
│   ├── dskd_config.py           # DSKD configuration
│   ├── cdm_config.py            # CDM configuration
│   ├── emo_config.py            # EMO configuration
│   └── stella_config.py         # Stella configuration
├── scripts/                     # Training scripts (.ps1 & .sh)
├── distiller.py                 # Unified training engine
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── README.md
```

## Cài đặt

### 1. Tạo môi trường ảo

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

- **Deep Learning**: PyTorch >= 2.0.0
- **Transformers**: transformers, huggingface_hub, tokenizers, PEFT
- **Data Processing**: pandas, numpy
- **ML/Metrics**: scikit-learn, scipy
- **String Matching**: Levenshtein, editdistance, fastdtw
- **Utils**: tqdm, kagglehub, datasets

## Tasks được hỗ trợ

### 1. **Semantic Textual Similarity (STS)**
- SICK, STS12, STSb datasets
- Metric: Spearman correlation

### 2. **Text Classification**
- Banking77, Emotion, Tweet datasets
- Metric: Accuracy, F1-score (macro)

### 3. **Pair Classification**
- MRPC, SciTail, WiC datasets
- Metric: Accuracy, F1, Precision, Recall, Average Precision

## Sử dụng

### Quick Start

#### 1. Chuẩn bị dữ liệu

Đặt file CSV training vào thư mục `data/`:
```csv
# Format: CSV với cột 'text' hoặc 'premise', 'hypothesis'
# Example: data/AllNLI.csv, data/merged_3_data_5k_each.csv
```

#### 2. Chạy training

**Windows PowerShell:**
```powershell
# TALAS method (our method)
cd scripts
.\train_talas.ps1

# Other methods
.\train_dskd.ps1
.\train_cdm.ps1
.\train_emo.ps1
.\train_stella.ps1
```

**Linux/Mac:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# TALAS method
cd scripts
./train_talas.sh

# Other methods
./train_dskd.sh
./train_cdm.sh
./train_emo.sh
./train_stella.sh
```

#### 3. Hoặc dùng Python trực tiếp

```bash
python main.py \
    --method talas \
    --train_data data/merged_3_data_5k_each.csv \
    --student_model model_hub/MiniLM-L6-H384-distilled-from-BERT-Base \
    --teacher_model Qwen/Qwen3-Embedding-0.6B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --save_dir checkpoints/talas
```

### Import modules cho custom training

```python
from config import TALASConfig, DSKDConfig, CDMConfig, EMOConfig
from distiller import KnowledgeDistiller

# TALAS training
config = TALASConfig(
    train_data_path="data/merged_3_data_5k_each.csv",
    student_model_name="model_hub/MiniLM-L6-H384-distilled-from-BERT-Base",
    teacher_model_name="Qwen/Qwen3-Embedding-0.6B",
    batch_size=32,
    epochs=5,
    learning_rate=2e-5,
    last_layer_idx=2,      # Use last 2 layers for KD
    start_rkd=0,           # Start structural loss from layer 0
    w_task=0.1,
    w_kd=0.75,
    w_struct=10.0
)

distiller = KnowledgeDistiller(config)
distiller.train()
```


## Loss Functions

1. **info_nce**: InfoNCE loss cho contrastive learning
2. **cosine_embedding_loss**: Cosine similarity loss
3. **pair_inbatch_similarity_loss**: In-batch similarity loss
4. **pair_inbatch_triplet_loss**: Triplet loss với margin

## Evaluation

```python
# Evaluate STS tasks
eval_sts_task(model, test_sts_tasks)

# Evaluate classification tasks
eval_classification_task(model, test_cls_tasks)

# Evaluate pair classification tasks
eval_pair_task(model, test_pair_tasks)
```

## Files Structure

```
Teacher-Anchor-KD/
├── main.py                      # Main entry point
├── distiller.py                 # Unified training engine
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── base_config.py          # Base configuration
│   ├── talas_config.py         # TALAS method config
│   ├── dskd_config.py          # DSKD method config
│   ├── cdm_config.py           # CDM method config
│   ├── emo_config.py           # EMO method config
│   └── stella_config.py        # Stella method config
│
├── src/                         # Core modules
│   ├── data_utils/              # Dataset & loaders
│   │   ├── dataset.py           # TextPairRaw
│   │   └── dataset_cache.py     # TextPairWithTeacher
│   ├── criterions/              # KD methods
│   │   ├── teacher_anchor_kd.py      # TALAS
│   │   ├── dual_space_kd.py          # DSKD
│   │   ├── contextual_dynamic_mapping.py  # CDM
│   │   ├── emo_embedding_distillation.py  # EMO
│   │   └── stella_distillation.py    # Stella
│   ├── evaluation/              # Evaluation tools
│   │   ├── evaluation_automodel.py
│   │   └── evaluation_model_define.py
│   ├── loss.py                  # Loss functions
│   ├── pooling.py               # Pooling utilities
│   └── cache_teacher.py         # Teacher caching
│
├── scripts/                     # Training scripts
│   ├── train_talas.ps1/.sh     # TALAS training
│   ├── train_dskd.ps1/.sh      # DSKD training
│   ├── train_cdm.ps1/.sh       # CDM training
│   ├── train_emo.ps1/.sh       # EMO training
│   └── train_stella.ps1/.sh    # Stella training
│
├── data/                        # Training data (gitignored)
├── model_hub/                   # Local models (gitignored)
└── checkpoints/                 # Model checkpoints (gitignored)
```

## Command Line Arguments

```bash
python main.py --help

Arguments:
  --method              Distillation method (talas, dskd, cdm, emo, stella)
  --train_data         Path to training CSV
  --student_model      Student model name/path
  --teacher_model      Teacher model name/path
  --batch_size         Batch size (default: 32)
  --epochs             Number of epochs (default: 5)
  --lr                 Learning rate (default: 2e-5)
  --max_length         Max sequence length (default: 256)
  --w_task             Task loss weight
  --alpha_dtw          DTW alignment weight (for CDM/DSKD)
  --save_dir           Checkpoint directory
  --seed               Random seed (default: 42)
  --num_workers        Dataloader workers (default: 0)
```

## Key Features

### 🚀 Teacher Anchor KD (TALAS)
- **Efficient**: Cache teacher embeddings, reduce GPU memory by ~50%
- **Flexible**: Use ALL model layers automatically (no need to specify layer indices)
- **Robust**: SAM optimizer for better generalization
- **Scalable**: Supports large teacher models (Qwen, GTE, etc.)

### 🔧 Implementation Highlights
- **Unified training engine**: Single `distiller.py` handles all methods
- **Automatic layer detection**: No manual layer selection needed
- **Multi-GPU support**: Teacher and student on separate GPUs
- **Mixed precision**: AMP for faster training
- **Automatic evaluation**: STS, classification, and pair tasks after each epoch

### 📊 Evaluation Metrics
- **STS tasks**: SICK, STS12-16, STSb (Spearman correlation)
- **Classification**: Banking77, Emotion, Tweet (Accuracy, F1)
- **Pair tasks**: MRPC, SciTail, WiC (Accuracy, F1, AP)

## Architecture Details

### TALAS Loss Function
```
Total Loss = w_task × L_task + w_kd × L_kd + w_struct × L_struct

where:
- L_task: InfoNCE contrastive loss
- L_kd: 1 - cosine_similarity(student_proj, teacher) for last N layers
- L_struct: Pair-wise similarity loss between consecutive layers
```

### Key Improvements
1. **Removed base_layers parameter**: Now uses ALL layers automatically
2. **Simplified last_layer_idx**: Changed from list of indices to a single number
   - `last_layer_idx=2` → use last 2 layers for KD
3. **Added start_rkd**: Control where structural loss computation starts
   - `start_rkd=0` → compute from layer 0 to n-1
4. **Dynamic projection heads**: Initialized on first forward pass
5. **Teacher model cleanup**: Automatically freed after caching

## License

Educational and Research Purpose

## Citation

```bibtex
@misc{teacher-anchor-kd,
  title={Teacher Anchor Knowledge Distillation},
  author={TALAS Research Team},
  year={2026}
}
```

## Contributors

TALAS Research Team
