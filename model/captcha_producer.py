# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:10:47 2016

@author: jie.zhou@sjtu.edu.cn
"""

from captcha.image import ImageCaptcha
import random
import string
import argparse
import os


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
    producer = ImageCaptcha(width=128,
                            height=64,
                            font_sizes=[40])
    args = vars(parser.parse_args())
    if "path" not in args:
        path = ""
    else:
        path = args['path']
    print(args)
    for i in range(int(args['number'])):
        number_to_write = "".join([random.choice(string.digits) for _ in range(4)])
        producer.write(number_to_write, os.path.join(path, str(i)+"_"+number_to_write+".png"))


if __name__ == "__main__":
    main()