#!/usr/bin/python
"""
author : windows98@ruc.edu.cn
"""
import tensorflow as tf
import os
import numpy as np
import re


def read_and_decode(path):
    filenames = map(lambda filename: os.path.join(path, filename), os.listdir(path))
    labels = np.array(map(lambda x: decode_label(re.findall("_(.*?)\.", x)[0]), filenames))
    filename_tensor = tf.convert_to_tensor(filenames)
    label_tensor = tf.convert_to_tensor(labels)
    input_queue = tf.train.slice_input_producer([filename_tensor,label_tensor])
    image_name, label = input_queue
    image_file = tf.read_file(image_name)
    image = tf.image.decode_png(image_file)
    image.set_shape([60, 160, 3])
    return image, label


def decode_label(label):
    one_hot_label = np.zeros([4, 10])
    index = [[0, 1, 2, 3], map(int, list(label))]
    one_hot_label[index] = 1.0
    return one_hot_label


def inputs(path, batch_size, min_after_dequeue, capacity):
    image, label = read_and_decode(path)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      min_after_dequeue=min_after_dequeue,
                                                      capacity=capacity)
    return image_batch, label_batch
