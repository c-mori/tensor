import numpy as np
import os
import time
import datetime
import tensorflow as tf
from fcrvnet import FCRVNet


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


def train(train_x, train_y, val_x, val_y, layer_sizes, activation=None, learn_rate=1e-3,
          dropouts=0, l2_lambda=0.0, epochs=100, batch_size=96, shuffle=True, name=None):

    if not isinstance(layer_sizes, list):
        layer_sizes = [layer_sizes]
    num_layers = len(layer_sizes)
    n_in = train_x.shape[1]
    n_out = train_y.shape[1]

    # set up keep probability vectors for training and evaluation
    if not isinstance(dropouts, list):
        dropouts = [dropouts]
    if len(dropouts) < num_layers:
        dropouts += [0.0] * (num_layers - 1)
    keep_probs = [1.0 - d for d in dropouts]
    keep_all = [1.0] * len(keep_probs)

    # ensure output paths exist
    if not name:
        name = 'default'
    home_path = os.path.join(os.path.expanduser('~'), 'tensorflow_output')
    if not os.path.exists(home_path):
        os.mkdir(home_path)
    home_path = os.path.join(home_path, name)
    if not os.path.exists(home_path):
        os.mkdir(home_path)
    check_path = os.path.join(home_path, 'checkpoints')
    if not os.path.exists(check_path):
        os.mkdir(check_path)
    summary_path = os.path.join(home_path, 'summaries')
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # build graph
            nn = FCRVNet(input_factors=n_in,
                         output_factors=n_out,
                         layer_sizes=layer_sizes,
                         activation=activation,
                         l2_lambda=l2_lambda)
            # set parameters
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(nn.loss)
            batches = batch_iter(train_x, train_y, batch_size, epochs, shuffle=shuffle)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())