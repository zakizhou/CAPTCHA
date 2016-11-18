#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import captcha_model
import captcha_inputs
from datetime import datetime


def main():
    images, labels = captcha_inputs.inputs("/home/windows98/TensorFlow/application/CAPTCHA/images/PNG", 5, 100, 300)
    # eval_images, eval_labels = captcha_inputs("/home/windows98/TensorFlow/application/CAPTCHA/eval_images/PNG",
    # 5, 100, 300)
    with tf.variable_scope("inference") as scope:
        logits = captcha_model.inference(images)
        # scope.reuse_variables()
        # eval_logits = captcha_model.inference(eval_images)
    # accuracy = captcha_model.evaluation(eval_logits, eval_labels)
    loss = captcha_model.loss(logits, labels)
    train_op = captcha_model.train(loss)

    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = datetime.now()
    try:
        index = 1
        while not coord.should_stop():
            _, loss_value = sess.run([train_op, loss])
            print("step: " + str(index) + " loss:" + str(loss_value))
            # if index % 10 == 0:
            #     accuracy = sess.run(validation_accuracy)
            #     print("validation accuracy: "+str(accuracy))
            index += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        end_time = datetime.now()
        print("Time Consumption: " + str(end_time - start_time))
    except KeyboardInterrupt:
        print("keyboard interrupt detected, stop running")
        del sess

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
   main()
