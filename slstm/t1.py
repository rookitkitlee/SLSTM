import torch
import torch.nn as nn
from torch.autograd import Variable

x = torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
z = x*x
J = torch.mean(z)

m = x*x
n = torch.mean(m)

o = J+n

# J.backward(retain_graph=True)
# print(x.grad)
# n.backward(retain_graph=True)
# print(x.grad)

o.backward()
print(x.grad)
