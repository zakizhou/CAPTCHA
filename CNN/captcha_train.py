#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import captcha
import captcha_inputs


def main():
    images, labels = captcha_inputs.inputs("/home/windows98/TensorFlow/application/CAPTCHA/images/PNG", 5, 100, 300)
    eval_images, eval_labels = captcha_inputs("/home/windows98/TensorFlow/application/CAPTCHA/eval_images/PNG", 5, 100, 300)
    with tf.variable_scope("inference") as scope:
        logits = captcha.inference(images)
        scope.reuse_variables()
        eval_logits = captcha.inference(eval_images)
    accuracy = captcha.evaluation(eval_logits, eval_labels)
    loss = captcha.loss(logits, labels)
    train_op = captcha.train(loss)
    with tf.Session() as sess:
        init_all = tf.initialize_all_variables()
        init_local = tf.initialize_local_variables()
        init = tf.group(init_all, init_local)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            sess.run([train_op, loss])
        except Exception:
            pass


if __name__ == "__main__":
   main()
