#!/usr/bin/python

import tensorflow as tf
import sys

def read_and_decode(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return image


def inputs(filename,batch_size):
    filename_queue = tf.train.string_input_producer([filename],num_epoches=None)

    image = read_and_decode([filename])

    batch_images = tf.train.shuffle_batch([image],batch_size=batch_size)

    return batch_images


