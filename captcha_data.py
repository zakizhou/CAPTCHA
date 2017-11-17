import tensorflow as tf
# import os

from captcha_config import config


def preprocess(tfrecord):
    features = tf.parse_single_example(tfrecord, features={
        "filename": tf.FixedLenFeature([], dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.string)
    })
    label = tf.decode_raw(features['label'], tf.int32)
    image_file = tf.read_file(features['filename'])
    image = tf.image.decode_png(image_file, channels=3)
    image.set_shape([config['image_height'], config['image_width'], 3])
    return {"image": image, "label": label}


def generate_handle(tfrecord_name):
    dataset = tf.data.TFRecordDataset(tfrecord_name)
    repeated_dataset = dataset.repeat()
    preprocessed_dataset = repeated_dataset.map(preprocess)
    batched_dataset = preprocessed_dataset.batch(config['batch_size'])
    iterator = batched_dataset.make_one_shot_iterator()

    handle = iterator.string_handle()
    output_types = batched_dataset.output_types
    output_shapes = batched_dataset.output_shapes
    return output_types, output_shapes, handle
