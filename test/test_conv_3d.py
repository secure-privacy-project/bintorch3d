import bintorch
from bintorch.autograd import Variable
import bintorch.nn.functional as F
import autograd.numpy as np

from bintorch.nn._functions.img2col_3d import get_im2col_indices
import bintorch.nn as nn

import torch



print(f"{get_im2col_indices(np.array((1,1,3,3,3)), 3, 3, 3, padding=0, stride=1)=}")

print(f"{get_im2col_indices(np.array((1,1,1,1,1)), 3, 3, 3, padding=1, stride=1)=}")

print(f"{get_im2col_indices(np.array((1,1,2,2,2)), 2, 2, 2, padding=0, stride=2)=}")

print(f"{get_im2col_indices(np.array((1,1,4,4,4)), 2, 2, 2, padding=0, stride=2)=}")


conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)
conv_torch = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1)

conv_torch.weight.data = torch.tensor(conv.weight.data).to(torch.float32)
conv_torch.bias.data = torch.tensor(conv.bias.data.ravel()).to(torch.float32)

x = np.arange(2*3*3*3).astype('float32').reshape(1,2,3,3,3).astype('float32')

x_bintorch = Variable(x)

print(f"{conv(x_bintorch).data=}")

x_torch = torch.tensor(x)

print(f"{conv_torch(x_torch).data=}")



batchsize = 2

x = np.arange(batchsize*2*4*4*4).astype('float32').reshape(batchsize,2,4,4,4).astype('float32')

x += np.random.normal(size=x.shape)

x_bintorch = Variable(x, requires_grad=True)

print(f"{F.max_pool3d(x_bintorch, 2).data=}")

x_torch = torch.tensor(x, requires_grad=True)

print(f"{torch.max_pool3d(x_torch, 2).data=}")



xx_bintorch = F.max_pool3d(F.max_pool3d(x_bintorch, 2), 2) # shape = (1,2,1,1,1)
xx_torch = torch.max_pool3d(torch.max_pool3d(x_torch, 2), 2) # shape = (1,2,1,1,1)

print(f"{xx_bintorch.data=}")
print(f"{xx_torch.data=}")



fc = nn.Linear(2,2)
fc_torch = torch.nn.Linear(2,2)

fc_torch.weight.data = torch.tensor(fc.weight.data).to(torch.float32)
fc_torch.bias.data = torch.tensor(fc.bias.data.ravel()).to(torch.float32)

xxx_bintorch = fc(xx_bintorch.view(batchsize,2))
xxx_torch = fc_torch(xx_torch.view(batchsize,2))

print(f"{xxx_bintorch.data=}")
print(f"{xxx_torch.data=}")



target = np.ones(batchsize).astype(np.int64)
target_bintorch = Variable(target, requires_grad=False)
target_torch = torch.tensor(target, requires_grad=False)



loss_bintorch = F.cross_entropy(xxx_bintorch, target_bintorch)
loss_torch = torch.nn.functional.cross_entropy(xxx_torch, target_torch)



print(f"{loss_bintorch.data=}")
print(f"{loss_torch=}")



loss_bintorch.backward()
loss_torch.backward()



print(f"{np.array(x_bintorch.grad.data)=}")
print(f"{x_torch.grad=}")
