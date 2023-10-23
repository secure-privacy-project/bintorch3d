import math
from .optimizer import Optimizer
import autograd.numpy as np

class SGD(Optimizer):
    """Implements stochastic gradient descent algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum (default: 0)

    """

    def __init__(self, params, lr, weight_decay=0, momentum=0, dampening=0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        .. note:: (copied from github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py)
            The implementation of SGD with Momentum/Nesterov subtly differs from
            Sutskever et. al. and implementations in some other frameworks.

            Considering the specific case of Momentum, the update can be written as

            .. math::
                \begin{aligned}
                    v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                    p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
                \end{aligned}

            where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
            parameters, gradient, velocity, and momentum respectively.

            This is in contrast to Sutskever et. al. and
            other frameworks which employ an update of the form

            .. math::
                \begin{aligned}
                    v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                    p_{t+1} & = p_{t} - v_{t+1}.
                \end{aligned}

            The Nesterov version is analogously modified.

            Moreover, the initial value of the momentum buffer is set to the
            gradient value at the first step. This is in contrast to some other
            frameworks that initialize it to all zeros.
    
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # momentum_buffer
                    if group['momentum'] != 0:
                        state['momentum_buffer'] = np.copy(grad)

                state['step'] += 1

                # weight decay
                if group['weight_decay'] != 0:
                    grad += group['weight_decay'] * p.data

                # momentum 
                if group['momentum'] != 0:
                    state['momentum_buffer'] *= group['momentum'] 
                    state['momentum_buffer'] += grad * (1 - group['dampening'])
                    grad = state['momentum_buffer']

                step_size = group['lr']

                p.data += -step_size * grad

        return loss
