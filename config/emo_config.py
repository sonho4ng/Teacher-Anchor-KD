"""
Configuration for EMO (Embedding Model Distillation)
"""

from .base_config import BaseConfig


class EMOConfig(BaseConfig):
    """Configuration for EMO distillation method."""
    
    # Method
    distill_method = "emo"
    
    # Model settings
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    # Special tokens for EMO
    student_special_token = "[CLS]"
    teacher_special_token = "<s>"
    
    # Loss weights
    w_task = 0.5          # Bi-encoder task loss weight
    alpha_kd = 0.5        # KD loss weight
    att_loss_weight = 0.1 # Attention alignment loss weight
    ot_loss_weight = 1.0  # Optimal transport loss weight
    
    # EMO-specific settings
    k_layers = 1          # Number of last layers for attention alignment
    alpha_ot = 0.1        # Sinkhorn regularization parameter
    max_iter_ot = 100     # Maximum Sinkhorn iterations
    
    # Training settings
    batch_size = 32
    epochs = 5
    learning_rate = 5e-5
    min_lr = 2e-6
    warmup_ratio = 0.1
    temperature = 0.05
    
    # Paths
    save_dir = "checkpoints/emo"
    
    def __init__(self, **kwargs):
        """Allow overriding config values."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
