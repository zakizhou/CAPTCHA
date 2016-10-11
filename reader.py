#!/usr/bin/python

import tensorflow as tf
import sys

filename = sys.argv[1]
filename_queue = tf.train.string_input_producer([filename],num_epoches=None)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,features={"image":tf.FixedLenFeature([],tf.int64),
                                                                "label":tf.FixedLenFeature([784,],tf.int64)})

image = features["image"]
label = features["label"]


