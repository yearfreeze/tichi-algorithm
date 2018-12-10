# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:11:42 2018

@author: freeze
"""
import tensorflow as tf
import numpy as np


def func(in_put,in_channel,out_channel):
    with tf.variable_scope(name_or_scope="yes",reuse=tf.AUTO_REUSE):
        weights=tf.get_variable(name="weight",shape=[2,2,in_channel,out_channel],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(input=in_put, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    return convolution

def main():
    with tf.Graph().as_default():
        x=tf.placeholder(tf.float32,shape=[1,4,4,1])
        
        for _ in range(5):
            output = func(x, 1, 1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                _output = sess.run([output], feed_dict={x:np.random.uniform(low=0, high=255, size=[1, 4, 4, 1])})
                print (_output)
                
                
if __name__ == "__main__":
    main()
        