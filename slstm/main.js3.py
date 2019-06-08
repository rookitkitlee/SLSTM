import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


list = []          ## 空列表
list.append('Google')   ## 使用 append() 添加元素
list.append('Runoob')
for item in list:
    print(item)