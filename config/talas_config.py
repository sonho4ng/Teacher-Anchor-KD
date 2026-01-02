from .base_config import BaseConfig


class TALASConfig(BaseConfig):
    
    distill_method = "talas"
    
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    student_special_token = "##"
    teacher_special_token = "G"
    
    # TALAS parameters
    last_layer_idx = 2  # Number of last layers to use for final KD (from bottom up)
    start_rkd = 0  # Start index for structural loss computation
    w_task = 0.1
    w_kd = 0.75
    w_struct = 10.0
    eps_norm = 1e-8
    temperature = 0.07
    rho = 0.05  # SAM optimizer rho parameter
    
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    
    cache_teacher = True
    cache_path = "cache/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = False
    cache_dtype = "float32"
    
    save_dir = "checkpoints/talas"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
