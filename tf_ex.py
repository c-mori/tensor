import tensorflow as tf
import numpy as np


def rmse(predictions, actual):
    sse = np.sum(np.square(predictions - actual))
    return np.sqrt(sse / float(actual.shape[0]))


def linear_data(examples=100000, inputs=5, outputs=1, weights=None, biases=None, noise=0.2, classify=False):
    if weights:
        inputs = weights.shape[0]

    x = np.random.uniform(-1., 1., size=(examples, inputs))

    if biases:
        outputs = biases.shape[1]

    if not weights:
        weights = np.random.uniform(-1.0, size=(inputs, outputs))
    if not biases:
        biases = np.random.uniform(-1.0, size=(1, outputs))

    y = np.dot(x, weights) + biases

    if classify:
        y = y > y.mean()

    noise = np.maximum(np.minimum(noise, .99), 0)
    noise = 1 / (1 - noise) - 1
    x += noise * np.random.normal(0, x.std(), size=x.shape)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    return x, y


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.  Adapted (with minor changes) from:
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    data = np.array(list(zip(x, y)))
    data_size = len(data)
    num_batches_per_epoch = int(np.ceil(float(len(data)) / batch_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train():
    pass
