#!/usr/bin/python
import tensorflow as tf

def read_and_decode(path):
    os.chdir(path)
    filenames = os.listdir(os.getcwd())
    labels = map(lambda x:x.split(".")[0],filenames)
    filename_tensor = tf.convert_to_tensor(filenames)
    label_tensor = tf.convert_to_tensor(labels)
    input_queue = tf.train.slice_input_producer([filename_tensor,label_tensor])
    image_name,label = input_queue
    image_file = tf.read_file(image_name)
    image = tf.image.decode_png(image_file)
    return image,label

def inputs(path,batch_size):
    image,label = read_and_decode(path)
    image_batch,label_batch  = tf.train.shuffle_batch([image,label],batch_size=batch_size)
    return image_batch,label_batch
