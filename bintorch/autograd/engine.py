from .variable import Variable
from .function import AccumulateGrad
import os
if os.environ.get("DP_NUMPY", "1") == "0":
    import numpy as np
else:
    import dp_numpy as np

def excute(fn, grad_in=None):
    if fn is not None:
        if isinstance(fn, AccumulateGrad):
            if fn.variable.requires_grad and grad_in is not None:
                if fn.variable.grad is None:
                    fn.variable.grad = np.zeros(fn.variable.data.shape)

                fn.variable.grad += grad_in
            return

        grad_outs = fn.apply(grad_in)
        if type(grad_outs) is not tuple:
            grad_outs = (grad_outs, )

        for i, next_func in enumerate(fn.next_functions):
                excute(next_func, grad_outs[i])


def backward(variables):

    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    for variable in variables:

        if variable.grad_fn is not None:
            excute(variable.grad_fn)
