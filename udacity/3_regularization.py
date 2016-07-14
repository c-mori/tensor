
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 3
# ------------
# 
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.
# 
# The goal of this assignment is to explore regularization techniques.

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import time


# First reload the data we generated in _notmist.ipynb_.

# In[3]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[4]:

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[5]:

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

batch_size = 64
h0 = 256
h1 = 256
h2 = 256
lambda_ = 5e-5
lr = .0005
keep_prob = .675

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    tf_tr_dataset = tf.constant(train_dataset)

    # Variables.
    w0 = tf.Variable(tf.truncated_normal([image_size * image_size, h0]))
    b0 = tf.Variable(tf.zeros([h0]))

    w1 = tf.Variable(tf.truncated_normal([h0, h1]))
    b1 = tf.Variable(tf.zeros([h1]))

    w2 = tf.Variable(tf.truncated_normal([h1, h2]))
    b2 = tf.Variable(tf.zeros([h2]))

    w3 = tf.Variable(tf.truncated_normal([h2, num_labels]))
    b3 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    s0 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, w0) + b0), keep_prob)
    s1 = tf.nn.dropout(tf.nn.relu(tf.matmul(s0, w1) + b1), keep_prob)
    s2 = tf.nn.relu(tf.matmul(s1, w2) + b2)
    logits = tf.matmul(s2, w3) + b3

    reg = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
    # reg = tf.reduce_sum(tf.square(w0)) + tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w1))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + (lambda_ * reg)

    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    v0 = tf.nn.relu(tf.matmul(tf_valid_dataset, w0) + b0)
    v1 = tf.nn.relu(tf.matmul(v0, w1) + b1)
    v2 = tf.nn.relu(tf.matmul(v1, w2) + b2)
    valid_prediction = tf.nn.softmax(tf.matmul(v2, w3) + b3)

    t0 = tf.nn.relu(tf.matmul(tf_test_dataset, w0) + b0)
    t1 = tf.nn.relu(tf.matmul(t0, w1) + b1)
    t2 = tf.nn.relu(tf.matmul(t1, w2) + b2)
    test_prediction = tf.nn.softmax(tf.matmul(t2, w3) + b3)

    r0 = tf.nn.relu(tf.matmul(tf_tr_dataset, w0) + b0)
    r1 = tf.nn.relu(tf.matmul(r0, w1) + b1)
    r2 = tf.nn.relu(tf.matmul(r1, w2) + b2)
    tr_prediction = tf.nn.softmax(tf.matmul(r2, w3) + b3)

batches = np.ceil(float(train_labels.shape[0]) / batch_size)
num_epochs = 175
num_steps = int(np.ceil(float(train_labels.shape[0]) / batch_size))

start = time.time()

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for epoch in range(num_epochs):
        l_mean = 0
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            # offset = (step * batch_size) % (int(batch_size * 2.5) - batch_size)
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            l_mean += l

        print("Avg Minibatch loss for epoch %d: %f" % (epoch, l_mean / num_steps))
        print(" Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        if epoch % 5 == 0:
            print("  ***   Training set accuracy: %.2f%% ***" % accuracy(tr_prediction.eval(), train_labels))
            print("  *** Validation set accuracy: %.2f%% ***" % accuracy(valid_prediction.eval(), valid_labels))
    print("\nFinal Training set accuracy: %.2f%%" % accuracy(tr_prediction.eval(), train_labels))
    print("Final Validation set accuracy: %.2f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test set accuracy: %.2f%%" % accuracy(test_prediction.eval(), test_labels))
print("Total time: %.3f minutes" % ((time.time() - start) / 60.))

# In[161]:

best_w0 = w0
best_w1 = w1
best_w2 = w2
best_w3 = w3

best_b0 = b0
best_b1 = b1
best_b2 = b2
best_b3 = b3

best_lr = lr
best_h0 = h0
best_h1 = h1
best_h2 = h2
best_lambda = lambda_
best_keep = keep_prob

best_epochs = 175
