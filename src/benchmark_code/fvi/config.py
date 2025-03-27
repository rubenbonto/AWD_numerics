import torch
import numpy as np
import random

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)
# check gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
MEM_SIZE = 3000
BATCH_SIZE = 80
DISCOUNT = 1.0
N_INSTANCE = 10
