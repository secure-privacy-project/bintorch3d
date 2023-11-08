from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.numpy as np
from .img2col_3d import *

def _pool_forward_3d(X, size=2, stride=2):
    n, c, d, h, w = X.shape
    d_out = (d - size) / stride + 1
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not d_out.is_integer() or not w_out.is_integer() or not h_out.is_integer():
        print(f"X.shape={X.shape}")
        raise Exception('Invalid output dimension!')

    d_out, h_out, w_out = int(d_out), int(h_out), int(w_out)

    X_reshaped = X.reshape(n * c, 1, d, h, w)
    X_col = im2col_indices(X_reshaped, size, size, size, padding=0, stride=stride)

    max_idx = np.argmax(X_col, axis=0)
    out = np.array(X_col[max_idx, range(max_idx.size)])

    out = out.reshape(d_out, h_out, w_out, n, c)
    out = np.transpose(out, (3, 4, 0, 1, 2))

    return out

class Max_pool3d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size):
        assert isinstance(input, Variable)

        def np_fn(input_np, kernel_size):

            return _pool_forward_3d(input_np, kernel_size)

        np_args = (input.data, kernel_size)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Max_pool3d, Max_pool3d).backward(ctx, grad_output)

