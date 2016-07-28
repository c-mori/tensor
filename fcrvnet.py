import tensorflow as tf


class FCRVNet(object):
    """
    A fully connected neural network
    """

    def __init__(self, input_factors, output_factors, layer_sizes, activation=None, l2_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32, [None, input_factors], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, output_factors], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, [len(layer_sizes)], name="dropout_keep_prob")

        if not activation:
            activation = 'relu'
        self.activation = activation
        self.weights = {}
        self.biases = {}

        l2_loss = tf.constant(0.0)

        layer_sizes = [input_factors] + list(layer_sizes)
        with tf.name_scope("inputs"):
            s = tf.Variable(self.input_x, name="s0")
        for i, layer_size in xrange(len(layer_sizes[1:])):
            with tf.name_scope("hidden_%s-%s" % (i, layer_size)):
                w = tf.Variable(tf.truncated_normal([layer_sizes[i - 1], layer_size]), name='w%s' % i)
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
                s = tf.nn.dropout(s, self.keep_prob[i], name='dropout%s')
                self.signal = s
        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([layer_sizes[-1], output_factors]), name='w_out')
            b = tf.Variable(tf.zeros([output_factors]), name='b_out')
            self.weights['w_out'] = w
            self.biases['b_out'] = b
            l2_loss += tf.nn.l2_loss(w)

            self.predictions = tf.nn.bias_add(tf.matmul(self.signal, w), b)

        with tf.name_scope('loss'):
            loss = tf.square(self.predictions - self.input_y)
            self.loss = tf.reduce_sum(loss) + l2_lambda * l2_loss

        with tf.name_scope('rmse'):
            self.rmse = tf.sqrt(tf.reduce_mean(loss), name='rmse')
