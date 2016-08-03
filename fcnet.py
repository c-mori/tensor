import numpy as np
import os
import datetime
import tensorflow as tf


class FCNet(object):
    """
    A fully connected neural network
    """

    def __init__(self, input_factors, output_factors, layer_sizes, activation=None, l2_lambda=0.0, classify=False):
        self.input_x = tf.placeholder(tf.float32, [None, input_factors], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, output_factors], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, [len(layer_sizes)], name="dropout_keep_prob")

        if not activation:
            activation = 'relu'
        self.activation = activation
        self.weights = {}
        self.biases = {}
        self.rmse = tf.constant(0, 'float', name="rmse")
        self.accuracy = tf.constant(0, 'float', name='accuracy')
        self.logits = None

        l2_loss = tf.constant(0.0)

        layer_sizes = [input_factors] + list(layer_sizes)
        with tf.name_scope("inputs"):
            s = self.input_x
        for i, layer_size in enumerate(layer_sizes[1:]):
            with tf.name_scope("hidden_%s-%s" % (i, layer_size)):
                # w = tf.Variable(tf.truncated_normal([layer_sizes[i], layer_size]), name='w%s' % i)
                w = tf.get_variable(name='w%s' % i, shape=[layer_sizes[i], layer_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([layer_size]), name='b%s' % i)
                self.weights['w_h%s' % i] = w
                self.biases['b_h%s' % i] = b
                l2_loss += tf.nn.l2_loss(w)

                s = tf.nn.bias_add(tf.matmul(s, w), b)
                if activation == 'relu':
                    s = tf.nn.relu(s, name='s%s' % i)
                elif activation == 'sigmoid':
                    s = tf.nn.sigmoid(s, name='s%s' % i)
                elif activation == 'tanh':
                    s = tf.nn.tanh(s, name='s%s' % i)
                else:
                    print "Unknown activation function: %s.  Valid: ['relu', 'sigmoid', 'tanh']" % activation
                    return
                s = tf.nn.dropout(s, self.keep_prob[i], name='dropout%s' % i)
                self.signal = s
        with tf.name_scope('output'):
            # w = tf.Variable(tf.truncated_normal([layer_sizes[-1], output_factors]), name='w_out')
            w = tf.get_variable(name='w_out', shape=[layer_sizes[-1], output_factors],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([output_factors]), name='b_out')
            self.weights['w_out'] = w
            self.biases['b_out'] = b
            l2_loss += tf.nn.l2_loss(w)

            self.predictions = tf.nn.bias_add(tf.matmul(self.signal, w), b)

        with tf.name_scope('loss'):
            if classify:
                loss = tf.nn.softmax_cross_entropy_with_logits(self.predictions, self.input_y)
                with tf.name_scope('accuracy'):
                    self.logits = self.predictions
                    self.predictions = tf.nn.softmax(self.predictions, name='softmax%d' %i)
                    correct = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
            else:
                loss = tf.square(self.predictions - self.input_y)
                with tf.name_scope('rmse'):
                    self.rmse = tf.sqrt(tf.reduce_mean(loss), name='rmse')

            self.loss = tf.reduce_mean(loss) + l2_lambda * l2_loss


def rmse(predictions, actual):
    sse = np.sum(np.square(predictions - actual))
    return np.sqrt(sse / float(actual.shape[0]))


def linear_data(examples=100000, inputs=5, outputs=1, val=0., weights=None, biases=None, noise=0.2, classify=False):
    if weights:
        inputs = weights.shape[0]

    examples = int(np.round(examples * (1.0 + val), 0))

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
        y = (np.arange(len(np.unique(y))) == y[:, None]).astype(np.float32)[:, 0]

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


def train(train_x, train_y, val_x, val_y, layer_sizes, activation=None, learn_rate=1e-3, classify=False,
          dropouts=0, l2_lambda=0.0, epochs=10, batch_size=96, shuffle=True, name=None,
          eval_every=100, chkpt_every=200):

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

    ts = "_" + datetime.datetime.now().isoformat()

    tf.reset_default_graph()
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # build graph
            nn = FCNet(input_factors=n_in,
                       output_factors=n_out,
                       layer_sizes=layer_sizes,
                       activation=activation,
                       l2_lambda=l2_lambda,
                       classify=classify)
            # set parameters
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learn_rate)
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
            acc_summary = tf.scalar_summary("accuracy", nn.accuracy)

            train_summary_op = tf.merge_summary([loss_summary, rmse_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(summary_path, "train" + ts)
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            val_summary_op = tf.merge_summary([loss_summary, rmse_summary, acc_summary])
            val_summary_dir = os.path.join(summary_path, "val" + ts)
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
                _, step, summaries, loss, err, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, nn.loss, nn.rmse, nn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if classify:
                    print "{}: step {}, train loss {:g}, train accuracy {:g}".format(time_str, step, loss, accuracy)
                else:
                    print "{}: step {}, train loss {:g}, train rmse {:g}".format(time_str, step, loss, err)
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
                step, summaries, loss, err, accuracy = sess.run(
                    [global_step, val_summary_op, nn.loss, nn.rmse, nn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if classify:
                    print "{}: step {}, val loss {:g}, val accuracy {:g}".format(time_str, step, loss, accuracy)
                else:
                    print "{}: step {}, val loss {:g}, val rmse {:g}".format(time_str, step, loss, err)
                if writer:
                    writer.add_summary(summaries, step)

            # generate batches and loop through training
            batches = batch_iter(train_x, train_y, batch_size, epochs, shuffle=shuffle)
            for batch in batches:
                batch_x, batch_y = (np.array(i) for i in zip(*batch))
                # run train operation
                train_step(batch_x, batch_y)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % eval_every == 0:
                    print "\nEvaluation:"
                    val_step(val_x, val_y, writer=val_summary_writer)
                    print("")
                if current_step % chkpt_every == 0:
                    path = saver.save(sess, check_prefix, global_step=current_step)
                    print "Saved model checkpoint to {}\n".format(path)


            print "\nFinal Evaluation:"
            val_step(val_x, val_y, writer=val_summary_writer)
            path = saver.save(sess, check_prefix, global_step=current_step)
            print "Saved final model checkpoint to {}\n".format(path)

            print "Done."
            return nn
