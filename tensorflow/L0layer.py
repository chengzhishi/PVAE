import tensorflow as tf
import math
from tensorflow.keras.layers import Layer
import numpy as np

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class L0Pair(Layer):
    """Implementation of L0 regularization for the input units of a fully connected layer"""

    def __init__(self, in_features, out_features, bias=True, weight_decay=0.01, droprate_init=0.2, temperature=1. / 20.,
                 lamba=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Pair, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        self.reg_record = tf.zeros([1])
        self.mask_record = tf.zeros([1])
        self.global_step = 0
        if bias:
            self.use_bias = True
        # self.global_step = tf.Variable(0, trainable=False)
        #     def build(self):
        self.mask = tf.stop_gradient(tf.zeros(out_features))
        self.mean_weights = tf.get_variable("mean_weights", shape=[self.in_features, self.out_features],
                                            initializer=tf.initializers.he_normal())
        self.var_weights = tf.get_variable("var_weights", shape=[self.in_features, self.out_features],
                                           initializer=tf.initializers.he_normal())
        # self.step_count = tf.zeros([1])
        weight_initer = tf.truncated_normal_initializer(
            mean=math.log(1 - self.droprate_init) - math.log(self.droprate_init), stddev=0.01)
        self.qz_loga = tf.get_variable("qz_loga", dtype=tf.float32, shape=[self.out_features],
                                       initializer=weight_initer)

        if self.use_bias:
            self.mean_bias = tf.get_variable(name="mean_bias", shape=[self.out_features],
                                             initializer=tf.initializers.zeros())
            self.var_bias = tf.get_variable(name="var_bias", shape=[self.out_features],
                                            initializer=tf.initializers.zeros())

    def constrain_parameters(self, **kwargs):
        clipped_value = tf.clip_by_value(self.qz_loga, math.log(1e-2), math.log(1e2))
        tf.assign(qz_loga, clipped_value)

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = tf.math.log(xn) - tf.math.log(1 - xn)
        return tf.clip_by_value(tf.math.sigmoid(logits * self.temperature - self.qz_loga), epsilon, 1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = tf.math.sigmoid((tf.math.log(x) - tf.math.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""

        logpw_col = tf.math.reduce_sum(
            (.5 * self.prior_prec * (tf.pow(self.mean_weights, 2) + tf.pow(self.var_weights, 2))) + self.lamba, 0)
        logpw = tf.math.reduce_sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - tf.math.reduce_sum(
            .5 * self.prior_prec * (tf.pow(self.mean_bias, 2) + tf.pow(self.var_bias, 2)))
        return logpw + logpb

    def regularization(self):
        reg = self._reg_w()/50000
        # tf.summary.scalar('L0 Regularization', reg)
        return reg

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = tf.random_uniform(size, epsilon, 1 - epsilon)
        return eps

    def tanh_hard(self, x):
        """Hard tanh."""
        return tf.minimum(1.0, tf.maximum(0.0, x))

    def sample_mask(self):
        z = self.quantile_concrete(self.tanh_hard(0.8 * self.get_eps([self.out_features]) + 0.1))
        mask = self.tanh_hard(z)
        # tf.summary.scalar('mask count', tf.reduce_sum(mask))
        return tf.reshape(mask, [1, self.out_features])

    def __call__(self, input):
        # self.step_count = self.step_count + tf.ones([1])
        # sess = tf.get_default_session()
        # train_summary = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('./log/', sess.graph)
        # # summary_writer = tf.train.SummaryWriter(, sess.graph)
        # train_writer.add_summary(train_summary, self.step_count)
        mask = self.sample_mask()
        self.mask = mask
        mean = tf.matmul(input, self.mean_weights) * mask
        var = tf.matmul(input, self.var_weights) * mask
        if self.use_bias:
            mean = tf.add(mean, self.mean_bias)
            var = tf.add(var, self.var_bias)
        var = tf.clip_by_value(var, 1e-6, tf.float32.max)
        # if self.global_step//1000 == 0:
        #     tf.summary.scalar("Regularization"+str(self.global_step//1000), self.regularization())
        #     tf.summary.scalar("Mask Sum"+str(self.global_step//1000), self.sample_mask())
        # self.global_step += 1
        return mean, var, self.regularization(), self.sample_mask()

    def __repr__(self):
        s = ('{name}({in_features} -> 2*{out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
