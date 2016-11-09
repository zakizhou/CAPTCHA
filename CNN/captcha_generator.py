# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:10:47 2016

@author: Windo
"""

from captcha.image import ImageCaptcha
import random
import string
import argparse


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
    #parser.add_argment("-")
    maker = ImageCaptcha()
    args = vars(parser.parse_args())
    if "path" not in args:
        path = ""
    else:
        print("2")
        path = args['path']
    print(args)
    for i in range(int(args['number'])):
        number_to_write = "".join([random.choice(string.digits) for j in range(4)])
        maker.write(number_to_write, path+str(i)+"_"+number_to_write+".PNG")


if __name__ == "__main__":
    main()