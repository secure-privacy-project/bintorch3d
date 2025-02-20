from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
from torch.utils.data import DataLoader, Dataset

install_aliases()

import os
import gzip
import struct
import array
import autograd.numpy as np
from urllib.request import urlretrieve

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        print('Downloading ' + filename)
        urlretrieve(url, out_file)

def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels

def load_mnist(flatten=True):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

    train_images, train_labels, test_images, test_labels = mnist()

    # make it 3d
    train_images = np.repeat(train_images[0:10000, np.newaxis, :, :], 28, axis=1)
    test_images = np.repeat(test_images[0:1000, np.newaxis, :, :], 28, axis=1)

    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0

    return train_images if flatten else train_images.reshape((train_images.shape[0], 1, 28, 28, 28)), \
           train_labels, \
           test_images if flatten else test_images.reshape((test_images.shape[0], 1, 28, 28, 28)), \
           test_labels

def collate_fn(batch):
    images = np.asarray([b[0] for b in batch])
    labels = np.asarray([b[1] for b in batch])
    return images, labels

class MnistDataset_3d(Dataset):

    def __init__(self, training=True, flatten=True):
        self.training = training
        self.train_images, self.train_labels, self.test_images, self.test_labels = load_mnist(flatten)

    def __len__(self):
        if self.training:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):

        if self.training:
            return self.train_images[idx], self.train_labels[idx]
        else:
            return self.test_images[idx], self.test_labels[idx]
