import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


x = nn.Parameter(torch.randn(2))
y = Variable(torch.Tensor([[1,1],[2,2],[3,3]]))
z = x*y
w = y*x
print(z)
print(w)


#
# print(x)
# print(x.shape)
#
# xx = nn.Parameter(torch.randn(2))
# print(xx)
# print(xx.shape)
#
# print(xx * x)
#
# print("----------------------------------------------------")
#
# net_out = Variable(torch.Tensor([[1],[2],[3]]))
# target = Variable( torch.LongTensor([[0],[0],[1]]))
#
# softmax = nn.Softmax()
# print(softmax(net_out,dim=1))
#
# criterion = nn.CrossEntropyLoss()
# print(criterion(softmax(net_out), target))
#

# y = xx * xx
# out = y.mean()
# out.backward()
#
# print(xx)
# print(xx.grad)
#
# xx.grad.zero_()
#
# z = xx*2
# out = z.mean()
# out.backward()
# print(xx.grad)


# x = x+2
#
# out = x.mean()
# out.backward()
# print(x.grad)


# print(x.grad_fn)
# print(y.grad_fn)

# z = y * y * 3
# out = z.mean()
# print(z)
# print(out)
#
# out.backward()
# print(x.grad)