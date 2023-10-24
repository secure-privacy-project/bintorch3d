
import sys
sys.path.append('../')

import bintorch.nn as nn
import bintorch.nn.functional as F
from bintorch.autograd import Variable
from examples.data_mnist_3d import MnistDataset_3d, collate_fn
from bintorch.utils.data import DataLoader
import bintorch
import autograd.numpy as np

import DP_util
import opacus


num_epochs = 300
batch_size_test = 64
learning_rate = 0.01
max_grad_norm = 1
noise_multiplier = 0.1
stochastic_batch_size = 64

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*((((28-4)//2-4)//2)**3), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2(x), 2))
        x = x.view(-1, 20*((((28-4)//2-4)//2)**3))
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = ConvNet()

train_loader = DataLoader(dataset=MnistDataset_3d(training=True, flatten=False),
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  batch_size=1)

test_loader = DataLoader(dataset=MnistDataset_3d(training=False, flatten=False),
                                  collate_fn=collate_fn,
                                  shuffle=False,
                                  batch_size=batch_size_test)



def eval_model(epoch):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for images, labels in test_loader:
        images = Variable(images)
        labels_var = Variable(labels)
        outputs = model(images)
        predicted = np.argmax(outputs.data, 1)
        loss += F.cross_entropy(outputs, labels_var, size_average=False).data
        total += labels.shape[0]
        correct += (predicted == labels).sum()

    print('\rEpoch [{}/{}], Test Accuracy: {}%  Loss: {:.4f}'.format(epoch + 1, num_epochs, 100 * correct / total, loss/total))

index_in_stochastic_batch_size = 0

for param in model.parameters():
    param.summed_grad = None

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        # Forward pass
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        
        # Backward and optimize
        # zero_grad (for each sample)
        for param in model.parameters():
            if param.grad is not None: 
                param.grad.fill(0)

        loss.backward()

        # hand-made optimization
        # gradient clipping
        L2norm = 0
        for param in model.parameters():
            # calculate L2 norm
            L2norm += np.sum(np.ravel(param.grad)**2)
        L2norm = L2norm ** .5
        clip_factor = (max_grad_norm / L2norm + 1e-6)
        clip_factor = np.minimum(clip_factor, 1.0)
        for param in model.parameters():
            # clip by L2 norm
            clipped_grad = param.grad * clip_factor
            # noising
            noised_grad = clipped_grad + np.random.normal(
                size=param.grad.shape, 
                scale=noise_multiplier * max_grad_norm
            )
            # add
            if param.summed_grad is None:
                param.summed_grad = noised_grad
            else:
                param.summed_grad += noised_grad
        
        index_in_stochastic_batch_size += 1
        if index_in_stochastic_batch_size == stochastic_batch_size:
            # optimize
            print("*")
            for param in model.parameters():
                param.grad = param.summed_grad
                param.data += -learning_rate * param.grad
                param.summed_grad = None
            index_in_stochastic_batch_size = 0

        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.data), end=' ')
    eval_model(epoch)
