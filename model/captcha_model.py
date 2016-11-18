#!/usr/bin/python
"""
author : jie.zhou@sjtu.edu.cn
functions for building the captcha model
"""


import tensorflow as tf

BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH = 50000
WIDTH = 128
HEIGHT = 64
CHANNELS = 3
CLASSES = 10
NUMBERS = 4


def inference(inputs):

    with tf.variable_scope("conv_pool_1"):
        print("conv 1: ")
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
        print(conv.get_shape())
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        active = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=active,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pooling")
        print(pool.get_shape())

    with tf.variable_scope("conv_pool_2"):
        print("conv 2: ")
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
        print(conv.get_shape())
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        active = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=active,
                              ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1],
                              padding="SAME",
                              name="pooling")
        print(pool.get_shape())
    with tf.variable_scope("conv_pool_3"):
        print("conv 3: ")
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
        print(conv.get_shape())
        conv_bias = tf.nn.bias_add(value=conv,
                                   bias=biases,
                                   name="add_biases")
        active = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(value=active,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pooling")
        print(pool.get_shape())
    reshape = tf.reshape(pool,
                         shape=[BATCH_SIZE, -1],
                         name="reshape")
    print(reshape.get_shape())
    dims = reshape.get_shape().as_list()[-1]
    with tf.variable_scope("fully_conn"):
        print("fully_conn: ")
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
        print(conn.get_shape())
    with tf.variable_scope("output"):
        print("output: ")
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
        print(logits.get_shape())
        reshape = tf.reshape(logits, shape=[BATCH_SIZE, NUMBERS, CLASSES])
    return reshape


def loss(logits, labels):
    cross_entropy_per_number = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy = tf.reduce_mean(cross_entropy_per_number)
    tf.add_to_collection("loss", cross_entropy)
    return cross_entropy


def accuracy(logits, labels):
    reshaped_logits = tf.reshape(logits, [-1, NUMBERS, CLASSES])
    inference = tf.argmax(reshaped_logits, 2)
    true_labels = tf.argmax(labels, 2)
    equal = tf.reduce_all(tf.equal(inference, true_labels), 1)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32), name="accuracy")
    return accuracy


def train(loss, learning_rate=0.0005):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
