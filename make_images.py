#!/usr/bin/python

from captcha.image import ImageCaptcha
import string
import random

maker = ImageCaptcha()
captcha = "".join([random.choice(string.digits) for _ in range(4)])
maker.write(captcha,"/home/zhoujie/TensorFlow/application/CAPTCHA/images/"+captcha+".jpg")
