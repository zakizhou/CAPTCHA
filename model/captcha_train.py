#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import captcha_model
import captcha_inputs
import tensorflow as tf
from datetime import datetime


FRACTION = 0.4
NUM_EXAMPLES_PER_EPOCH = captcha_model.NUM_EXAMPLES_PER_EPOCH
MIN_AFTER_DEQUEUE = int(FRACTION * NUM_EXAMPLES_PER_EPOCH)
VALIDATION_MIN_AFTER_DEQUEUE = 4000


def main():
    filename_queue = tf.train.string_input_producer(["../images/tfrecords/train.tfrecords"])
    image, label = captcha_inputs.read_records(filename_queue)
    images, labels = captcha_inputs.records_inputs(image, label, MIN_AFTER_DEQUEUE)

    validation_filename_queue = tf.train.string_input_producer(["../images/tfrecords/validation.tfrecords"])
    validation_image, validation_label = captcha_inputs.read_records(validation_filename_queue)
    validation_images, validation_labels = captcha_inputs.records_inputs(validation_image,
                                                                         validation_label,
                                                                         VALIDATION_MIN_AFTER_DEQUEUE)

    # images, labels = captcha_inputs.inputs("../images/train", MIN_AFTER_DEQUEUE)
    # validation_images, validation_labels = captcha_inputs.inputs("../images/validation", VALIDATION_MIN_AFTER_DEQUEUE)

    # images, labels = captcha_inputs.inputs("/home/windows98/TensorFlow/application/CAPTCHA/images/"
    # , MIN_AFTER_DEQUEUE)
    # eval_images, eval_labels = captcha_inputs("/home/windows98/TensorFlow/application/CAPTCHA/eval_images/PNG",
    # 5, 100, 300)
    # TODO: add validation
    logits = captcha_model.inference(images)
    tf.get_variable_scope().reuse_variables()
    with tf.device("/cpu:0"):
        validation_logits = captcha_model.inference(validation_images)
        accuracy = captcha_model.evaluation(validation_logits, validation_labels)
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
            if index % 5 == 0:
                validation_accuracy = sess.run(accuracy)
                print("validation accuracy: "+str(validation_accuracy))
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
    del sess


if __name__ == "__main__":
    main()