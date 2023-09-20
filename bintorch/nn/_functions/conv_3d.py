from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.scipy.signal
import autograd.numpy as np
from .img2col_3d import *

# conv = autograd.scipy.signal.convolve
class Conv3d(Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weights_np, bias=None, stride=1, padding=0):
            out = conv_forward_3d(input_np, weights_np, bias, stride, padding)

            if bias is None:
                return out
            else:
                return out

        np_args = (input.data, weights.data, None if bias is None else bias.data)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Conv3d, Conv3d).backward(ctx, grad_output)


def conv_forward_3d(X, W, b, stride=1, padding=0):
    # cache = W, b, stride, padding
    n_filters, c_filter, d_filter, h_filter, w_filter = W.shape
    n_x, c_x, d_x, h_x, w_x = X.shape
    d_out = (d_x - d_filter + 2 * padding) / stride + 1
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not d_out.is_integer() or not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    d_out, h_out, w_out = int(d_out), int(h_out), int(w_out)

    X_col = im2col_indices(X, d_filter, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = np.matmul(W_col, X_col)
    if b is not None:
        out += b
    out = out.reshape(n_filters, d_out, h_out, w_out, n_x)
    out = np.transpose(out, (4, 0, 1, 2, 3))

    return out


