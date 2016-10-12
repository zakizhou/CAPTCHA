#!/usr/bin/python
# author : Windows98@ruc.edu.cn

import tensorflow as tf

WIDTH = 80
HEIGHT = 160
CHANNELS = 3
CLASSES = 10
NUMBERS = 4

graph = tf.Graph()

import reader

def convolution(input_images,filter_shape):
    filter = tf.Variable(tf.random_normal(filter_shape),dtype=tf.float32,name="filter")
    bias = tf.Variable(tf.zeros([filter_shape[3]]),name="bias")
    conv = tf.nn.conv2d(input_images,filter,strides=[1,1,1,1],padding="SAME",name="conv")
    conv_bias = tf.add(conv,bias,name="add_bias")
    relu = tf.nn.relu(conv,name="relu")
    pool = tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool")
    return pool

def fully_connect(inputs,hidden_units):
    former_units = inputs.get_shape().as_list()[-1]
    Weights = tf.Variable(tf.random_normal([former_units,hidden_units]),dtype=tf.float32,name="weights")
    bias = tf.Variable(tf.zeros([hidden_units]),name="bias")
    logits = tf.nn.xw_plus_b(inputs,Weights,bias,name="logits")
    return logits

with graph.as_default():
    batch_images,batch_labels  = reader.inputs("~/TensorFlow/application/CAPTCHA/images/PNG/",128)

    reshaped_labels = tf.reshape(labels,[-1,NUMBERS*CLASSES])

    with tf.name_scope("conv_1"):
        conv_1 = convolution(inputs,[5,5,3,64])
    with tf.name_scope("conv_2"):
        conv_2 = convolution(conv_1,[5,5,64,128])
    with tf.name_scope("conv_3"):
        conv_3 = convolution(conv_2,[5,5,128,256])
    with tf.name_scope("conv_4"):
        conv_4 = convolution(conv_3,[5,5,256,128])


    reshaped_conv_4 = tf.reshape(conv_4,[-1,10*20*128],name="reshaped_conv_4")

    with  tf.name_scope("fully_connect"):
        fc = fully_connect(reshaped_conv_4,1024)
        fc_result = tf.nn.relu(fc,name="fc_result")
    with tf.name_scope("output"):
        logits = fully_connect(fc_result,NUMBERS*CLASSES)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,reshaped_labels),name="cross_entropy")
    reshaped_logits = tf.reshape(logits,[-1,NUMBERS,CLASSES])

    inference = tf.argmax(reshaped_logits,2)
    true_labels = tf.argmax(labels,2)

    equal = tf.reduce_all(tf.equal(inference,true_labels),1)

    accuracy = tf.reduce_mean(tf.cast(equal,tf.float32),name="accuracy")

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy,name="optimizer")

with tf.Session(graph=graph) as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess,coord=coord)
    saver = tf.train.Saver()

    saver.save(sess,"/home/zhoujie/TensorFlow/application/CAPTCHA/proto_buffer/CNN")




