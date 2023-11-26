from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.numpy as np

class CrossEntropy(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True, weight=None, reduction='mean'):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(inputs, targets, size_average=True, weight=None, reduction='mean'):

            if not size_average:
                assert reduction=='mean'
                # deprecated option
                reduction = 'sum'

            N = inputs.shape[0]
#            probs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#            probs /= np.sum(probs, axis=1, keepdims=True)
#
#            ll = np.log(np.array(probs[np.arange(N), targets])+ 1.0e-10)

            # https://jaykmody.com/blog/stable-softmax/
            def log_softmax(x):
                x_max = np.max(x, axis=1, keepdims=True)
                return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=1, keepdims=True))            
            ll = log_softmax(inputs)[np.arange(N), targets]

            if weight is not None:
                if reduction=='mean':
                    # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    raise NotImplementedError()
                ll = ll * weight[targets]

            if reduction=='mean':
                return -np.sum(ll / N)
            elif reduction=='sum':
                return -np.sum(ll)
            else:
                raise NotImplementedError()

        np_args = (input.data, target.data, size_average, weight, reduction)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(CrossEntropy, CrossEntropy).backward(ctx, grad_output)



class MSELoss(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input_np, target_np, size_average=True):
            if size_average:
                return np.mean((input_np - target_np) ** 2)
            else:
                return np.sum((input_np - target_np) ** 2)

        np_args = (input.data, target.data, size_average)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(MSELoss, MSELoss).backward(ctx, grad_output)

