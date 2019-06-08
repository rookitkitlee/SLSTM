import torch
from model import SLSTMCell

torch.manual_seed(1)

base_size = 64
target_size = 10

cell = SLSTMCell(base_size, target_size)


