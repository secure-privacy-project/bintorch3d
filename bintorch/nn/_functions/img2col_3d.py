import autograd.numpy as np

def get_im2col_indices(x_shape, field_depth, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, D, H, W = x_shape
    assert (D + 2 * padding - field_depth) % stride == 0
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_depth = int((D + 2 * padding - field_depth) / stride + 1)
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_depth), field_height*field_width)
    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(out_depth), out_height*out_width)

    j0 = np.repeat(np.arange(field_height), field_width)
    j0 = np.tile(j0, C*field_depth)

    j1 = stride * np.tile(np.repeat(np.arange(out_height), out_width), out_depth)

    k0 = np.arange(field_width)
    k0 = np.tile(k0, C*field_depth*field_height)

    k1 = stride * np.tile(np.arange(out_width), out_depth*out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = k0.reshape(-1, 1) + k1.reshape(1, -1)

    l = np.repeat(np.arange(C), field_depth * field_height * field_width).reshape(-1, 1)

    return (l.astype(int), i.astype(int), j.astype(int), k.astype(int))


def im2col_indices(x, field_depth, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p), (p, p)), mode='constant')

    l, i, j, k = get_im2col_indices(x.shape, field_depth, field_height, field_width, padding, stride)

    cols = np.array(x_padded[:, l, i, j, k])
    C = x.shape[1]
    cols = np.transpose(cols, (1, 2, 0))
    cols = cols.reshape(field_depth * field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_depth=3, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, D, H, W = x_shape
    D_padded, H_padded, W_padded = D + 2 * padding, H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, D_padded, H_padded, W_padded), dtype=cols.dtype)
    l, i, j, k = get_im2col_indices(x_shape, field_depth, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_depth * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), l, i, j, k), cols_reshaped)
    if padding == 0:
        return x_padded
    return np.array(x_padded[:, :, padding:-padding, padding:-padding, padding:-padding])
