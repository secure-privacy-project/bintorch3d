import bintorch
from bintorch.autograd import Variable
import bintorch.nn.functional as F
import autograd.numpy as np

target = Variable(np.array((1, 3, 4, 3, 3)), requires_grad=False)
y = Variable(np.zeros((5, 5)), requires_grad=False)

l = Variable(np.ones((5, 5)), requires_grad=True)
m = Variable(np.eye(5), requires_grad=True)

x = l + m + y

x = F.cross_entropy(x, target)

x.backward()

print(x.data)

print(np.array(l.grad.data))



import torch

target = torch.tensor(np.array((1, 3, 4, 3, 3), dtype=np.int64), requires_grad=False)

y = torch.tensor(np.zeros((5, 5)), requires_grad=False)

l = torch.tensor(np.ones((5, 5)), requires_grad=True)
m = torch.tensor(np.eye(5), requires_grad=True)

x = l + m + y

x = torch.nn.functional.cross_entropy(x, target)

x.backward()

print(x.data)

print(np.array(l.grad.data))
