

# copyright Shouhei HANAOKA, ICAL, Dept. of Radiology, the Univ. of Tokyo Hospital



# Cross entropy without reduction

from bintorch.autograd import Function
from bintorch.autograd import Variable
import bintorch
import autograd.numpy as np

from typing import Callable, List, Optional, Union



class CrossEntropyLossForOneInBatch(Function):
    """
        calculate cross entropy, but no minibatch-size-wise reduction is done
    """
    @staticmethod
    def forward(ctx, input, target, index_in_batch):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input, targets, index_in_batch):
            probs = np.exp(input - np.max(input, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            N = input.shape[0]

            ll = np.log(np.array(probs[np.arange(N), targets]))

            return -ll[index_in_batch] # only one loss in the minibatch

        np_args = (input.data, target.data, index_in_batch)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(CrossEntropyLossForOneInBatch, CrossEntropyLossForOneInBatch).backward(ctx, grad_output)



def cross_entropy_loss_for_one_in_batch(input, target, index_in_batch):

    return CrossEntropyLossForOneInBatch.apply(input, target, index_in_batch)

