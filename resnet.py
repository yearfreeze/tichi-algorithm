# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:54:08 2018

@author: freeze
"""

import tensorflow as tf
import numpy as np
#import data_utils
#import cv2
#import matplotlib.pyplot as plt
#import math


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name = 'x')

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name = 'keep_prob')



def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape = [conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                              initializer = tf.random_normal_initializer(stddev = np.sqrt(2.0 / n)),
                              regularizer = regularizer)
        L = tf.nn.conv2d(x_tensor, weights, strides = [1, conv_strides[0], conv_strides[1], 1], padding = 'SAME')
        L = batch_norm(L,'bn')
        L = relu(L)
    return L

def batch_norm(x_tensor,name=None):
    mean,variance=tf.nn.moments(x_tensor,axes = [0])
    L=tf.nn.batch_normalization(x_tensor,mean,variance,0.01,1,0.001,name = name)
    return L


def maxpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.max_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def avgpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.avg_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def relu(L):
    return tf.nn.relu(L)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001)

    weights = tf.get_variable('weight', shape = [x_shape[1], num_outputs],
                              initializer = tf.uniform_unit_scaling_initializer(factor = 1.0),
                              regularizer = regularizer)
    bias = tf.Variable(tf.zeros([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, weights), bias)

    return out


def residual_block(x,filters,name=None):
    with tf.variable_scope(name):
        L=batch_norm(x,'bn')
        L=relu(L)
        L=conv2d(L,filters,(3,3),(1,1),'conv1')
        L=conv2d(L,filters,(3,3),(1,1),'conv2')
        L=L+x
        return L

def residual_blocks(x,n,filters=16,name=None):
    for i in range(n):
        L=residual_block(x,filters = filters,name = name+'{0}'.format(i+1))
        x=L
    return L





def resnet_110(x,keep_prob=None,phase_train=None,output_num=17):
    with tf.variable_scope("resnet_110"):
        conv1 = conv2d(x, 16, (3,3), (1,1), 'conv1')
        # resnet-1-18(32x32)
        with tf.variable_scope('32x32'):
            conv2=residual_blocks(conv1,18,name = 'residual_')

        # resnet-19-36(16x16)
        with tf.variable_scope('16x16'):
            conv3=batch_norm(conv2,name = 'bn_19')
            conv3=relu(conv3)
            conv3=conv2d(conv3,32,(1,1),(2,2),'subsample_1')
            conv3=residual_blocks(conv3,18,filters = 32,name = 'residual_')

        # resnet-36-53(8x8)
        with tf.variable_scope('8x8'):
            conv4=batch_norm(conv3,name = 'bn_36')
            conv4=relu(conv4)
            conv4=conv2d(conv4,64,(1,1),(2,2),'subsample_2')
            conv4=residual_blocks(conv4,18,filters = 64,name = 'residual_')
            conv=batch_norm(conv4,name = 'bn_53')
            conv=relu(conv)

        # global avgpool
        conv=tf.reduce_mean(conv,[1,2])

        out=output(conv,output_num)
        return out

"""
##############################
## Build the Neural Network ##
##############################
tf.reset_default_graph()
# Remove previous weights, bias, inputs, etc..
x = neural_net_image_input((32, 32, 3))
y = tf.placeholder(tf.int64,[None],name = 'y')
keep_prob = neural_net_keep_prob_input()
training = tf.placeholder(tf.bool, name = 'training')
learning_rate = tf.placeholder(tf.float32)
# Model

logits = resnet_110(x, keep_prob, training)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name = 'logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(y,10)))
starter_learning_rate = learning_rate
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, use_nesterov = True, momentum = 0.9).minimize(
    cost)

save_model_path= 'D:/tmp/ResNet'


sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess, logits, cost, X_train, y_train, 20, 64, 100, optimizer)

# Save Model
saver = tf.train.Saver()
save_path = saver.save(sess, save_model_path)

print('Validation')
run_model(sess, logits, cost, X_val, y_val, 1, 64)
"""