# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:10:47 2016

@author: jie.zhou@sjtu.edu.cn
"""
import tensorflow as tf
import numpy as np
from captcha.image import ImageCaptcha
import re
import random
import string
import argparse
import os

from captcha_config import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--path",
                        required=False,
                        help="path to store generated images")
    parser.add_argument("-n",
                        "--number",
                        required=True,
                        help="number of images generated")
    producer = ImageCaptcha(width=config['image_width'],
                            height=config['image_height'],
                            font_sizes=[40])
    args = vars(parser.parse_args())
    if "path" not in args:
        path = config['images_path']
    else:
        path = args['path']
    if not os.path.exists(path):
        os.makedirs(path)
    print("Starting generating %d images in %s" % (int(args['number']), path))
    for i in range(int(args['number'])):
        number_to_write = "".join([random.choice(string.digits) for _ in range(4)])
        producer.write(number_to_write, os.path.join(path, str(i)+"_"+number_to_write+".png"))
    print("images generated!")
    print("-" * 30)

    print("Staring convert tfrecords of these generated images!")
    generate_tfrecords(path)
    print("tfrecords generated!")


def generate_tfrecords(path):
    image_names = [os.path.join(path, image_name_) for image_name_ in os.listdir(path)]

    def convert_filename_to_label_str(image_name):
        image_label_strs = re.findall("_(.*?)\.", image_name)[0]
        image_label_np = np.array(list(image_label_strs), dtype=np.int32)
        image_label_np_str = image_label_np.tostring()
        return image_label_np_str

    name = path.split("/")[1]
    tfrecord_writer = tf.python_io.TFRecordWriter("tfrecords/" + name + ".tfrecords")

    for index, image_name in enumerate(image_names):
        image_label_np_str = convert_filename_to_label_str(image_name)
        # print(index)
        example = tf.train.Example(features=tf.train.Features(feature={
            "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label_np_str]))
        }))
        tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()


if __name__ == "__main__":
    main()