from abstract_network import *
import numpy as np
import tensorflow as tf

def inference0(input_x, cs, is_training=False):
    with tf.variable_scope("inference0"):
        conv1 = conv2d_bn_lrelu(input_x, cs[1], [4, 4], 2, is_training)
        conv2 = conv2d_bn_lrelu(conv1, cs[2], [4, 4], 2, is_training)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = tf.contrib.layers.fully_connected(conv2, cs[3], activation_fn=tf.identity)
        return fc1


def ladder0(input_x, cs, ladder0_dim, is_training=False):
    conv1 = conv2d_bn_lrelu(input_x, cs[1], [4, 4], 2, is_training)
    conv2 = conv2d_bn_lrelu(conv1, cs[2], [4, 4], 2, is_training)
    conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
    fc1_mean = tf.contrib.layers.fully_connected(conv2, ladder0_dim, activation_fn=tf.identity)
    fc1_stddev = tf.contrib.layers.fully_connected(conv2, ladder0_dim, activation_fn=tf.sigmoid)
    return fc1_mean, fc1_stddev


def inference1(latent1, cs, is_training=False):
    fc1 = fc_bn_lrelu(latent1, cs[3], is_training)
    fc2 = fc_bn_lrelu(fc1, cs[3], is_training)
    fc3 = tf.contrib.layers.fully_connected(fc2, cs[3], activation_fn=tf.identity)
    return fc3

def ladder1(latent1, cs, ladder1_dim, is_training=False):
    fc1 = fc_bn_lrelu(latent1, cs[3], is_training)
    fc2 = fc_bn_lrelu(fc1, cs[3], is_training)
    fc3_mean = tf.contrib.layers.fully_connected(fc2, ladder1_dim, activation_fn=tf.identity)
    fc3_stddev = tf.contrib.layers.fully_connected(fc2, ladder1_dim, activation_fn=tf.sigmoid)
    return fc3_mean, fc3_stddev


def ladder2(latent1, cs, ladder2_dim, is_training=False):
    fc1 = fc_bn_lrelu(latent1, cs[3], is_training)
    fc2 = fc_bn_lrelu(fc1, cs[3], is_training)
    fc3_mean = tf.contrib.layers.fully_connected(fc2, ladder2_dim, activation_fn=tf.identity)
    fc3_stddev = tf.contrib.layers.fully_connected(fc2, ladder2_dim, activation_fn=tf.sigmoid)
    return fc3_mean, fc3_stddev

def combine_noise(latent, ladder, name="default"):
    gate = tf.get_variable("gate", shape=latent.get_shape()[1:], initializer=tf.constant_initializer(0.1))
    return latent + tf.multiply(gate, ladder)

def generative0(latent1, cs, fs, ladder0=None, reuse=False, is_training=False):
    data_range = [0, 255]
    if ladder0 is not None:
        ladder0 = fc_bn_lrelu(ladder0, cs[3])
        if latent1 is not None:
            latent1 = combine_noise(latent1, ladder0, name="generative0")
        else:
            latent1 = ladder0
    elif latent1 is None:
        print("Generative layer must have input")
        exit(0)
    fc1 = fc_bn_relu(latent1, int(fs[2] * fs[2] * cs[2]), is_training)
    fc1 = tf.reshape(fc1,
                     tf.stack([tf.shape(fc1)[0], fs[2], fs[2], cs[2]]))
    conv1 = conv2d_t_bn_relu(fc1, cs[1], [4, 4], 2, is_training)
    output = tf.contrib.layers.convolution2d_transpose(conv1, 3, [4, 4], 2,
                                                       activation_fn=tf.sigmoid)
    output = (data_range[1] - data_range[0]) * output + \
             data_range[0]
    return output

def generative1(latent2, cs, ladder1=None, reuse=False, is_training=False):
    if ladder1 is not None:
        ladder1 = fc_bn_relu(ladder1, cs[3], is_training)
        if latent2 is not None:
            latent2 = combine_noise(latent2, ladder1, name="generative1")
        else:
            latent2 = ladder1
    elif latent2 is None:
        print("Generative layer must have input")
        exit(0)
    fc1 = fc_bn_relu(latent2, cs[3], is_training)
    fc2 = fc_bn_relu(fc1, cs[3], is_training)
    fc3 = tf.contrib.layers.fully_connected(fc2, cs[3], activation_fn=tf.identity)
    return fc3

def generative2(ladder2, cs, reuse=False, is_training=False):
    fc1 = fc_bn_relu(ladder2, cs[3], is_training)
    fc2 = fc_bn_relu(fc1, cs[3], is_training)
    fc3 = tf.contrib.layers.fully_connected(fc2, cs[3], activation_fn=tf.identity)
    return fc3
