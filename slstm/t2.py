import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.Tensor([[1,1],[2,2],[3,3]]),requires_grad=True)
y = x
z = x*x
J = z.mean()

J.backward(retain_graph=True)
print(x.grad)

print(y.grad)
