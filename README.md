# TALAS - Knowledge Distillation for NLP Models

Project nghiên cứu về Knowledge Distillation cho các mô hình NLP, tập trung vào transfer learning từ Teacher models sang Student models.

## Tổng quan

Project này implement nhiều phương pháp Knowledge Distillation:
- **Dual Space Knowledge Distillation (DSKD)**
- **Contextual Dynamic Mapping**
- **Teacher Anchor KD**
- **Stella & Jasper methods**
- **EMO Embedding Model Distillation**

## Cấu trúc Project

```
TALAS/
├── src/
│   ├── data_utils/          # Dataset & Data loaders
│   │   ├── dataset.py       # TextPairRaw, DualTokenizerCollate
│   │   └── dataset_cache.py # DualTokenizerCollateWithTeacher
│   ├── criterions/          # Knowledge Distillation methods
│   ├── evaluation/          # Evaluation metrics & tasks
│   │   ├── evaluation_automodel.py
│   │   └── evaluation_model_define.py
│   ├── loss.py              # Loss functions (info_nce, cosine, similarity, triplet)
│   ├── pooling.py           # Pooling utilities (last_token, mean_pooling)
│   └── cache_teacher.py     # Teacher model caching
├── config/                  # Configuration files
├── data/                    # Data directory
├── scripts/                 # Training scripts
├── requirements.txt         # Dependencies
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
```bash
# Data format: CSV với các cột premise, hypothesis (hoặc text)
# Example: data/train.csv
```

#### 2. Chạy training

**Windows PowerShell:**
```powershell
# CDM method
.\scripts\train_cdm.ps1

# DSKD method
.\scripts\train_dskd.ps1

# MINED method
.\scripts\train_mined.ps1
```

**Linux/Mac:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# CDM method
./scripts/train_cdm.sh

# DSKD method
./scripts/train_dskd.sh

# MINED method
./scripts/train_mined.sh
```

#### 3. Hoặc dùng Python trực tiếp

```bash
python main.py \
    --method cdm \
    --train_data data/train.csv \
    --student_model bert-base-uncased \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --save_dir checkpoints/cdm
```

### Import modules cho custom training

```python
from config import CDMConfig, DSKDConfig
from distiller import KnowledgeDistiller
from src.data_utils import TextPairRaw, DualTokenizerCollate
from src.loss import info_nce, cosine_embedding_loss
from src.pooling import last_token_pool, mean_pooling
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping

# Create config
config = CDMConfig(
    train_data_path="data/train.csv",
    batch_size=32,
    epochs=5
)

# Create distiller and train
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
TALAS/
├── main.py                  # Main entry point
├── distiller.py             # Training engine
├── requirements.txt         # Dependencies
├── README.md               # Documentation
│
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── base_config.py      # Base configuration
│   ├── cdm_config.py       # CDM method config
│   ├── dskd_config.py      # DSKD method config
│   └── mined_config.py     # MINED method config
│
├── src/                    # Core modules
│   ├── data_utils/         # Dataset & loaders
│   ├── criterions/         # KD methods
│   │   └── contextual_dynamic_mapping.py  # CDM implementation
│   ├── evaluation/         # Evaluation tools
│   ├── loss.py            # Loss functions
│   └── pooling.py         # Pooling utilities
│
├── scripts/                 # Training scripts
│   ├── train_cdm.sh        # Linux/Mac
│   ├── train_dskd.sh
│   ├── train_mined.sh
│   ├── train_cdm.ps1       # Windows PowerShell
│   ├── train_dskd.ps1
│   └── train_mined.ps1
│
├── data/                    # Training data
└── checkpoints/            # Model checkpoints
```

## Command Line Arguments

```bash
python main.py --help

Arguments:
  --method              Distillation method (cdm, dskd, mined)
  --train_data         Path to training CSV
  --student_model      Student model name/path
  --teacher_model      Teacher model name/path
  --batch_size         Batch size
  --epochs             Number of epochs
  --lr                 Learning rate
  --max_length         Max sequence length
  --w_task             Task loss weight
  --alpha_dtw          DTW KD loss weight
  --save_dir           Checkpoint directory
  --seed               Random seed
  --debug              Enable debug mode
  --num_workers        Dataloader workers
```

## TODO

- [ ] Implement DSKD criterion method
- [ ] Implement MINED criterion method  
- [ ] Implement cache_teacher.py for faster training
- [ ] Add evaluation during training
- [ ] Add tensorboard logging
- [ ] Add model export utilities
- [ ] Add inference scripts

## License

Research project - Educational purpose

## Contributors

TALAS Research Team
