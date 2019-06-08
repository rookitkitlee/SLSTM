import torch
from torch.autograd import Variable

def create_cuda_variable(size):
    return Variable(torch.randn(size), requires_grad=True)

interval0 = create_cuda_variable(2)



y = interval0*interval0
z = y.mean()

z.backward()
print(interval0.grad)
print(interval0.grad)
print(interval0.device)

print(interval0.device == 'cpu')