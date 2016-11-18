#!/usr/bin/python
"""
author : windows98@ruc.edu.cn
"""
import tensorflow as tf
import os
import numpy as np
import re
import captcha_model
from scipy.ndimage import imread


def read_and_decode(path):
    filenames = map(lambda filename: os.path.join(path, filename), os.listdir(path))
    labels = np.array(map(lambda x: decode_label(re.findall("_(.*?)\.", x)[0]), filenames))
    filename_tensor = tf.convert_to_tensor(filenames)
    label_tensor = tf.convert_to_tensor(labels)
    input_queue = tf.train.slice_input_producer([filename_tensor, label_tensor])
    image_name, label = input_queue
    image_file = tf.read_file(image_name)
    image = tf.image.decode_png(image_file)
    image.set_shape([64, 128, 3])
    return tf.cast(image, tf.float32), tf.cast(label, tf.float32)


def decode_label(label):
    one_hot_label = np.zeros([4, 10])
    index = [[0, 1, 2, 3], map(int, list(label))]
    one_hot_label[index] = 1.0
    return one_hot_label.astype(np.uint8)


def inputs(path, min_after_dequeue):
    image, label = read_and_decode(path)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=captcha_model.BATCH_SIZE,
                                                      min_after_dequeue=min_after_dequeue,
                                                      capacity=min_after_dequeue + 3 * captcha_model.BATCH_SIZE)
    return image_batch, label_batch


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dir, name, path):
    filenames = map(lambda filename: os.path.join(dir, filename), os.listdir(dir))
    tfrecords_name = os.path.join(path, name+".tfrecords")
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    for filename in filenames:
        image = imread(filename)
        label = decode_label(re.findall("_(.*?)\.", filename)[0])
        example = tf.train.Example(features=tf.train.Features(feature={'label': bytes_feature(label.tostring()),
                                                                       'image': bytes_feature(image.tostring())}))
        writer.write(example.SerializeToString())
    writer.close()
    print("successfully convert data to tfrecords!")


def read_records(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([captcha_model.HEIGHT * captcha_model.WIDTH * captcha_model.CHANNELS])
    label = tf.decode_raw(features['label'], tf.uint8)
    label.set_shape([captcha_model.NUMBERS * captcha_model.CLASSES])
    reshape_image = tf.reshape(image, shape=[captcha_model.WIDTH, captcha_model.HEIGHT, captcha_model.CHANNELS])
    reshape_label = tf.reshape(label, shape=[captcha_model.NUMBERS, captcha_model.CLASSES])
    return tf.cast(reshape_image, tf.float32), tf.cast(reshape_label, tf.float32)


def records_inputs(image, label, min_after_dequeue):
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=captcha_model.BATCH_SIZE,
                                            capacity=min_after_dequeue + 3 * captcha_model.BATCH_SIZE,
                                            min_after_dequeue=min_after_dequeue)
    return images, labels