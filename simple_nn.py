import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt


def rmse(predictions, actual):
    sse = np.sum(np.square(predictions - actual))
    return np.sqrt(sse / float(actual.shape[0]))

n = 3
hidden = n * 5

# Generate training data and outputs
train_x = np.random.normal(size=(20000, n))

mults = np.arange(n) + 1
bias = -3

train_y = np.dot(train_x, mults) + bias
train_y = np.expand_dims(train_y.astype(np.float32), 1)

# add some noise to X
# train_x = (9 * train_x + np.random.normal(size=train_x.shape)) / 10
train_x = train_x.astype(np.float32)

batch_size = 100
keep_percentage = .5
learn_rate = 5e-4
lambda_ = 0

num_labels = train_y.shape[1]

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, n))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_tr_dataset = tf.constant(train_x)

    # Variables.
    w0 = tf.Variable(tf.truncated_normal([n, hidden]))
    b0 = tf.Variable(tf.zeros([hidden]))
    w1 = tf.Variable(tf.truncated_normal([hidden, num_labels]))
    b1 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.nn.relu(tf.matmul(tf_train_dataset, w0) + b0)
    logits = tf.nn.dropout(logits, keep_percentage)
    logits = tf.matmul(logits, w1) + b1
    loss = tf.reduce_sum(tf.square(
        logits - tf_train_labels) + lambda_ * (tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1)))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # Predictions
    train_prediction = logits
    tr_preds = tf.nn.relu(tf.matmul(tf_tr_dataset, w0) + b0)
    tr_preds = tf.nn.dropout(tr_preds, 1.0)
    tr_preds = tf.matmul(tr_preds, w1) + b1

num_steps = 50001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_y.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_x[offset:(offset + batch_size), :]
        batch_labels = train_y[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {
            tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch RMSE: %.5f" % rmse(predictions, batch_labels))
            print("Train set RMSE: %.5f" % rmse(tr_preds.eval(), train_y))

    print("")
    preds = tr_preds.eval()

print("actual y values:")
print(train_y)
print("\n predictions:")
print(preds)

plt.scatter(train_y.reshape([-1, 1]), preds.reshape([-1, 1]))
plt.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], 'k-', lw=2, color='r')
plt.show()
