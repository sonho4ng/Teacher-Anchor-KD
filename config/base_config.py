"""
Base Configuration for Knowledge Distillation
"""

class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Task settings
    task_type = "pair_cls"  # "single_cls", "pair_cls", "pair_reg"
    max_length = 256
    
    # Training hyperparameters
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    warmup_ratio = 0.06
    
    # Loss weights
    w_task = 0.5      # Task loss weight
    alpha_dtw = 0.5   # DTW KD loss weight
    w_cls = 1.0       # CLS-level KD weight
    temperature = 0.07  # Temperature for contrastive loss
    
    # Model settings
    student_model_name = "bert-base-uncased"
    teacher_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dtype = "float32"  # "float32", "float16", "bfloat16"
    
    # Special tokens
    student_special_token = "##"  # BERT-style
    teacher_special_token = "_"   # Default
    
    # Data settings
    train_data_path = "data/merged_3_data_5k_each.csv"
    eval_data_path = None
    num_workers = 2
    
    # Distillation method
    distill_method = "cdm"  # "cdm", "dskd", "standard"
    
    # Checkpointing
    save_dir = "checkpoints"
    save_every = 1
    save_best = True
    
    # Debugging
    debug_align = False
    
    # Random seed
    seed = 42
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() 
                if not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
