#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import captcha
import captcha_inputs

def train():
    images,labels = captcha_inputs.inputs("/home/zhoujie/TensorFlow/application/CAPTCHA/images/PNG",5,100,300)
    logits = captcha.inference(images)
    loss = captcha.loss(logits,labels)
    train_op = captcha.train(loss)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            sess.run([train_op,loss])
        except Exception:
            pass

if __name__ == "__main__":
   train()
