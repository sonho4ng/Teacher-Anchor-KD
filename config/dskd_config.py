from .base_config import BaseConfig


class DSKDConfig(BaseConfig):
    distill_method = "dskd"
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    # Special tokens
    student_special_token = "##"
    teacher_special_token = "_"
    
    # Loss weights (DSKD uses dual alignment)
    w_task = 1.0
    alpha_dtw = 1.0
    w_cls = 1.0
    
    # DSKD-specific settings
    use_cross_attention = True
    bidirectional_align = True
    
    # Training settings
    batch_size = 32
    epochs = 10
    learning_rate = 2e-5
    
    # Paths
    save_dir = "checkpoints/dskd"
    
    def __init__(self, **kwargs):
        """Allow overriding config values."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
