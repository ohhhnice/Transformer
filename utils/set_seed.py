import torch
import random
import numpy as np
import os
def set_all_seeds(seed=42):
    """
    设置Python、NumPy、PyTorch和CUDA的随机数种子，以确保实验可重复
    
    参数:
    seed (int): 随机数种子，默认为42
    """
    # 设置Python随机数种子
    random.seed(seed)
    
    # 设置NumPy随机数种子
    np.random.seed(seed)
    
    # 设置PyTorch随机数种子
    torch.manual_seed(seed)
    
    # 如果使用CUDA（GPU），设置CUDA随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量（用于Python哈希等）
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"所有随机数种子已设置为: {seed}")