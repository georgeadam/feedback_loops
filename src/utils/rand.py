import numpy as np
import random
import torch


def set_seed(seed: int=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)