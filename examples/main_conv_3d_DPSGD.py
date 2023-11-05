
import sys
sys.path.append('../')

import bintorch.nn as nn
import bintorch.nn.functional as F
from bintorch.autograd import Variable
from examples.data_mnist_3d import MnistDataset_3d, collate_fn
from bintorch.utils.data import DataLoader
import bintorch
import autograd.numpy as np
import dp_numpy as dnp

import DP_util
# import opacus


num_epochs = 300
batch_size_test = 64
learning_rate = 0.01
max_grad_norm = 1
noise_multiplier = 0.1
batch_size_stochastic_train = 64
eps_per_minibatch = 10000.0 * batch_size_stochastic_train

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

index_in_stochastic_batch = 0
DP_util.zero_grad(model.parameters())

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        # Forward pass
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        # loss = DP_util.cross_entropy_loss_without_reduction(outputs, labels)
        
        # Backward and optimize
        grad_reducer = None

        # zero_grad (for each sample)
        DP_util.zero_grad(model.parameters())

        # calculate the gradient vector of each case in batch
        loss.backward()

        # serialize the grad
        grad = DP_util.load_and_serialize_grads(model.parameters())

        # reducer
        if grad_reducer is None:
            grad_reducer = dnp.reducer.mean(
                np.ones_like(grad) * -max_grad_norm, # minimum
                np.ones_like(grad) * +max_grad_norm # maximum
            )
        grad_reducer.add(grad)

        index_in_stochastic_batch += 1

        if index_in_stochastic_batch == batch_size_stochastic_train:
            # get accumulated grad with Laplace mechanism
            accumlated_grad = grad_reducer.laplace(eps_per_minibatch)

            # write back to the model's grad
            DP_util.deserialize_and_save(accumlated_grad, model.parameters())

            # add to the weight
            for param in model.parameters():
                if param.grad is not None:
                    param.data += -learning_rate * param.grad

            # clear accumulator and count
            grad_reducer = None
            index_in_stochastic_batch = 0

            DP_util.zero_grad(model.parameters())
            print('+')


        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.data), end=' ')
    eval_model(epoch)
