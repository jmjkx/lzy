from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import sklearn
from sklearn.utils.random import sample_without_replacement
import numpy
import gzip


# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


if __name__ == '__main__':
    x_train = extract_data(r'/home/liyuan/Programming/python/lzy/graph/data/train-images-idx3-ubyte.gz', 60000)
    y_train = extract_labels(r'/home/liyuan/Programming/python/lzy/graph/data/train-labels-idx1-ubyte.gz', 60000)
    x_test = extract_data(r'/home/liyuan/Programming/python/lzy/graph/data/t10k-images-idx3-ubyte.gz', 10000)
    y_test = extract_labels(r'/home/liyuan/Programming/python/lzy/graph/data/t10k-labels-idx1-ubyte.gz', 10000)

# 定义超参
    EPOCH = 1000
    BATCH_SIZE = 1000
    classes_name = [str(c) for c in range(10)]  # 分类地物数量

    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28, 28)
    #x_train = x_train[0:3000]



