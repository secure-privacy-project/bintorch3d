

# copyright Shouhei HANAOKA, ICAL, Dept. of Radiology, the Univ. of Tokyo Hospital



# Cross entropy without reduction

from bintorch.autograd import Function
from bintorch.autograd import Variable
import bintorch
import autograd.numpy as np

from typing import Callable, List, Optional, Union



class CrossEntropyLossWithoutReduction(Function):
    """
        calculate cross entropy, but no minibatch-size-wise reduction is done
    """
    @staticmethod
    def forward(ctx, input, target):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input, targets):
            probs = np.exp(input - np.max(input, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            N = input.shape[0]

            ll = np.log(np.array(probs[np.arange(N), targets]))

            return -ll # no reduction along minibatch dimension

        np_args = (input.data, target.data)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(CrossEntropyLossWithoutReduction, CrossEntropyLossWithoutReduction).backward(ctx, grad_output)



def cross_entropy_loss_without_reduction(input, target):

    return CrossEntropyLossWithoutReduction.apply(input, target)



def zero_grad(parameters):
    """
        clear the grad
    """
    for param in parameters:
        if param.grad is not None: 
            param.grad.fill(0)



def load_and_serialize_grads(parameters):
    """
        serialize parameters to a 1-d np.ndarray

        parameters : list
            parameters of a case

        return value: np.ndarray((num_param))
            serialized parameter gradient vector
    """
    params = list(parameters)

    # analyze
    num_params = 0
    for param in params:
        if param.grad is not None:
            num_params += np.prod(param.grad.shape)

    # alloc
    res = np.ndarray((num_params), dtype=np.float32)

    
    # load
    index_params = 0
    for param in params:
        if param.grad is not None:
            size_param = np.prod(param.grad.shape)
            res[index_params:index_params+size_param] = np.copy(param.grad.flatten())
            index_params += size_param

    assert index_params == num_params
    return res



def deserialize_and_save(serialized, parameters):
    """
        deserialize "serialized" and save to "parameters"'s grad
    """
    # save
    index_params = 0
    for param in parameters:
        if param.grad is not None:
            size_param = np.prod(param.grad.shape)
            param.grad = np.reshape(
                serialized[index_params:index_params+size_param], 
                param.grad.shape
            )
            index_params += size_param
    assert index_params == len(serialized)

