import numpy as np
import os
import datetime
import tensorflow as tf
from fcnet import FCRVNet


def rmse(predictions, actual):
    sse = np.sum(np.square(predictions - actual))
    return np.sqrt(sse / float(actual.shape[0]))


def linear_data(examples=100000, inputs=5, outputs=1, val=0., weights=None, biases=None, noise=0.2, classify=False):
    if weights:
        inputs = weights.shape[0]

    examples = int(np.round(examples * (1.0 + val),0))

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

    if val == 0.0:
        return x, y

    else:
        tr = int(examples / (1.0 + val))
        return x[:tr], y[:tr], x[tr:], y[tr:]


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
          dropouts=0, l2_lambda=0.0, epochs=100, batch_size=96, shuffle=True, name=None,
          eval_every=1, chkpt_every=1):

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
    home_path = os.path.join(os.path.expanduser('~'), 'tensorflow_output', name)
    if not os.path.exists(home_path):
        os.makedirs(home_path)
    check_path = os.path.join(home_path, 'checkpoints')
    if not os.path.exists(check_path):
        os.mkdir(check_path)
    check_prefix = os.path.join(check_path, "model")
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
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(nn.loss)
            grads_vars = optimizer.compute_gradients(nn.loss)
            train_op = optimizer.apply_gradients(grads_vars, global_step=global_step)

            # summaries:
            grad_summary = []
            for g, v in grads_vars:
                if g is not None:
                    grad_hist_summ = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summ = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summary.append(grad_hist_summ)
                    grad_summary.append(sparsity_summ)
            grad_summaries_merged = tf.merge_summary(grad_summary)

            loss_summary = tf.scalar_summary("loss", nn.loss)
            rmse_summary = tf.scalar_summary("rmse", nn.rmse)

            train_summary_op = tf.merge_summary([loss_summary, rmse_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(summary_path, "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            val_summary_op = tf.merge_summary([loss_summary, rmse_summary])
            val_summary_dir = os.path.join(summary_path, "val")
            val_summary_writer = tf.train.SummaryWriter(val_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            # define training and validation steps
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    nn.input_x: x_batch,
                    nn.input_y: y_batch,
                    nn.keep_prob: keep_probs
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, nn.loss, nn.rmse],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    nn.input_x: x_batch,
                    nn.input_y: y_batch,
                    nn.keep_prob: keep_all
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, val_summary_op, nn.loss, nn.rmse],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # generate batches and loop through training
            batches = batch_iter(train_x, train_y, batch_size, epochs, shuffle=shuffle)
            current_epoch = 0
            for batch in batches:
                current_epoch += 1
                batch_x, batch_y = (np.array(i) for i in zip(*batch))
                # run train operation
                train_step(batch_x, batch_y)
                current_step = tf.train.global_step(sess, global_step)
                if current_epoch % eval_every == 0:
                    print("\nEvaluation:")
                    val_step(val_x, val_y, writer=val_summary_writer)
                    print("")
                if current_epoch % chkpt_every == 0:
                    path = saver.save(sess, check_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
