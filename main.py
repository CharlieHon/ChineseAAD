import config as cfg

import numpy as np
import torch

# 设置随机种子，保证结果的可重复性
rand_seed = 2024
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(rand_seed)
