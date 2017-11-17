from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf


import captcha_model
import captcha_data


def main():
    output_types, output_shapes, train_handle = captcha_data.generate_handle("tfrecords/train.tfrecords")
    _, _, validation_handel = captcha_data.generate_handle("tfrecords/validation.tfrecords")

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(string_handle=handle,
                                                   output_types=output_types,
                                                   output_shapes=output_shapes)
    next_element = iterator.get_next()
    images = next_element['image']
    labels = next_element['label']

    logits = captcha_model.inference(images)
    loss = captcha_model.loss(logits, labels)
    accuracy = captcha_model.evaluation(logits, labels)
    train_op = captcha_model.train(loss)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(init)

    train_handle_value = sess.run(train_handle)
    validation_handel_value = sess.run(validation_handel)

    for i in range(10000):
        _, loss_value = sess.run([train_op, loss], feed_dict={handle: train_handle_value})
        if (i + 1) % 10 == 0:
            accuracy_value = sess.run(accuracy, feed_dict={handle: validation_handel_value})
            print("loop: %d, loss: %f accuracy: %f" % (i + 1, loss_value, accuracy_value))
    saver = tf.train.Saver()
    saver.save(sess, "save/model")
    sess.close()


if __name__ == "__main__":
    main()