from . import _functions

def mse_loss(input, target, size_average=True):

    return _functions.MSELoss.apply(input, target, size_average)

def cross_entropy(input, target, size_average=True, weight=None, reduction='mean'):

    return _functions.CrossEntropy.apply(input, target, size_average, weight, reduction)

def linear(input, weight, bias=None):

    return _functions.Linear.apply(input, weight, bias)

def relu(input):

    return _functions.ReLU.apply(input)

def conv2d(input, weight, bias=None, stride=1, padding=0):

    return _functions.Conv2d.apply(input, weight, bias, stride, padding)

def conv3d(input, weight, bias=None, stride=1, padding=0):

    return _functions.Conv3d.apply(input, weight, bias, stride, padding)

def max_pool2d(input, kernel_size):

    return _functions.Max_pool2d.apply(input, kernel_size)

def max_pool3d(input, kernel_size):

    return _functions.Max_pool3d.apply(input, kernel_size)

def dropout(input, p=0.5, training=False):

    return _functions.Dropout.apply(input, p, training)