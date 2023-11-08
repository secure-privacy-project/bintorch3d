
import sys
sys.path.append('../')

import bintorch.nn as nn
import bintorch.nn.functional as F
from bintorch.autograd import Variable
from data_mnist_3d import MnistDataset_3d, collate_fn
from bintorch.utils.data import DataLoader
import bintorch
import autograd.numpy as np
import dp_numpy as dnp

import DP_util

# from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement

from opacus.accountants.rdp import RDPAccountant



kernel_size = 5
padding = 0

num_epochs = 300
batch_size_test = 64
learning_rate = 0.1
max_grad_norm = 1.0
batch_size_stochastic_train = 64
batch_size_max_train = 256
noise_multiplier = 0.5
delta = 1.0e-8

# inspired by https://programming-dp.com/ch8.html#renyi-differential-privacy

print(f"noise_multiplier={noise_multiplier}")
print(f"max_grad_norm={max_grad_norm}")
print(f"batch_size_stochastic_train={batch_size_stochastic_train}")



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(20*((((28-(kernel_size-padding*2-1))//2-(kernel_size-padding*2-1))//2)**3), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2(x), 2))
        x = x.view(-1, 20*((((28-(kernel_size-padding*2-1))//2-(kernel_size-padding*2-1))//2)**3))
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

    print('\rEpoch [{}/{}], Test Accuracy: {}%  Loss: {:.4f}****'.format(epoch + 1, num_epochs, 100 * correct / total, loss/total))

training_sample_size = len(train_loader)
print(f"training_sample_size={training_sample_size}")

index_in_stochastic_batch = 0
DP_util.zero_grad(model.parameters())
grad_reducer = None

def get_minibatch_length():
    while(1):
        l = np.random.poisson(batch_size_stochastic_train)
        if 0<l:
            break
    return l

minibatch_length = get_minibatch_length()

step_count = 0
epsilon = 0

accountant = RDPAccountant()



# main loop

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
        # zero_grad (for each sample)
        DP_util.zero_grad(model.parameters())

        # calculate the gradient vector of each case in batch
        loss.backward()

        # serialize the grad
        grad = DP_util.load_and_serialize_grads(model.parameters())

        # reducer
        if grad_reducer is None:
            grad_reducer = dnp.reducer_rdp.sum(max_grad_norm)
        grad_reducer.add(grad)

        index_in_stochastic_batch += 1

        if index_in_stochastic_batch == minibatch_length:
            # get accumulated grad with Gaussian mechanism
            accumlated_grad = grad_reducer.gaussian(noise_multiplier)

            # write back to the model's grad
            DP_util.deserialize_and_save(accumlated_grad, model.parameters())

            # add to the weight
            for param in model.parameters():
                if param.grad is not None:
                    param.data += -learning_rate * param.grad / batch_size_stochastic_train

            # clear accumulator and count
            grad_reducer = None
            index_in_stochastic_batch = 0
            minibatch_length = get_minibatch_length()

            # step privacy accountant
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=batch_size_stochastic_train/training_sample_size)
            epsilon = accountant.get_epsilon(delta=delta)
            step_count += 1

            # clear 
            DP_util.zero_grad(model.parameters())

        print('\rEpoch [{}/{}], Step [{}/{}] ({} times accumulated), Loss: {:.4f} epsilon: {:.4f} delta: {}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), step_count, loss.data, epsilon, delta), end=' ')
    eval_model(epoch)
