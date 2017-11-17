#!/usr/bin/python
"""
author : jie.zhou@sjtu.edu.cn
functions for building the captcha model
"""


import tensorflow as tf
import numpy as np
from captcha_config import config

# BATCH_SIZE = config['batch_size']
NUM_EXAMPLES_PER_EPOCH = 50000
VALIDATION_SIZE = 10000
WIDTH = config['image_width']
HEIGHT = config['image_height']
CHANNELS = config['image_channels']
CLASSES = config['num_classes']
NUMBERS = config['num_digits']


def inference(inputs):
    inputs = tf.cast(inputs, tf.float32)
    with tf.variable_scope("conv_pool_1"):
        kernel = tf.get_variable(name="kernel",
                                 shape=[5, 5, 3, 48],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        biases = tf.get_variable(name="biases",
                                 shape=[48],
                                 initializer=tf.constant_initializer(value=0.),
                                 dtype=tf.float32)
        conv = tf.nn.conv2d(input=inputs,
                            filter=kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        relu = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pooling")

    with tf.variable_scope("conv_pool_2"):
        kernel = tf.get_variable(name="kernel",
                                 shape=[5, 5, 48, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        biases = tf.get_variable(name="biases",
                                 shape=[64],
                                 initializer=tf.constant_initializer(value=0.),
                                 dtype=tf.float32)
        conv = tf.nn.conv2d(input=pool,
                            filter=kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        relu = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=relu,
                              ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1],
                              padding="SAME",
                              name="pooling")
    with tf.variable_scope("conv_pool_3"):
        kernel = tf.get_variable(name="kernel",
                                 shape=[5, 5, 64, 128],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        biases = tf.get_variable(name="biases",
                                 shape=[128],
                                 initializer=tf.constant_initializer(value=0.),
                                 dtype=tf.float32)
        conv = tf.nn.conv2d(input=pool,
                            filter=kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        relu = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pooling")
        dims = np.prod(pool.get_shape().as_list()[1:])
    reshape = tf.reshape(pool,
                         shape=[-1, dims],
                         name="reshape")
    with tf.variable_scope("fully_conn"):
        weights = tf.get_variable(name="weights",
                                  shape=[dims, 2048],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05),
                                  dtype=tf.float32)
        biases = tf.get_variable(name="biases",
                                 shape=[2048],
                                 initializer=tf.constant_initializer(value=0.),
                                 dtype=tf.float32)
        output = tf.nn.xw_plus_b(x=reshape,
                                 weights=weights,
                                 biases=biases)
        conn = tf.nn.relu(output)
    with tf.variable_scope("output"):
        weights = tf.get_variable(name="weights",
                                  shape=[2048, NUMBERS * CLASSES],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05),
                                  dtype=tf.float32)
        biases = tf.get_variable(name="biases",
                                 shape=[NUMBERS * CLASSES],
                                 initializer=tf.constant_initializer(value=0.),
                                 dtype=tf.float32)
        logits = tf.nn.xw_plus_b(x=conn,
                                 weights=weights,
                                 biases=biases)
        reshape = tf.reshape(logits, shape=[-1, NUMBERS, CLASSES])
    return reshape


def loss(logits, labels):
    cross_entropy_per_number = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=labels)
    cross_entropy = tf.reduce_mean(cross_entropy_per_number)
    return cross_entropy


def evaluation(logits, labels):
    prediction = tf.argmax(logits, 2)
    prediction = tf.cast(prediction, tf.int32)
    equal = tf.equal(prediction, labels)
    equal_all = tf.reduce_all(equal, axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_all, tf.float32), name="accuracy")
    return accuracy


def train(loss, learning_rate=0.00001):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
