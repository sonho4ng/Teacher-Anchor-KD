from .base_config import BaseConfig


class CDMConfig(BaseConfig):
    """Configuration for CDM distillation method."""
    
    # Method
    distill_method = "cdm"
    
    # Model settings (CDM typically uses larger teacher)
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    # Special tokens (important for CDM alignment)
    student_special_token = "##"   # BERT uses ##
    teacher_special_token = "G"    # Qwen uses G (customize based on tokenizer)
    
    # Loss weights (CDM emphasizes DTW alignment)
    w_task = 0.5
    alpha_dtw = 0.5    # DTW alignment weight
    w_cls = 1.0        # CLS distillation weight
    
    # Training settings
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    
    # CDM-specific
    debug_align = True  # Enable alignment debugging for first batch
    
    # Paths
    save_dir = "checkpoints/cdm"
    
    def __init__(self, **kwargs):
        """Allow overriding config values."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
