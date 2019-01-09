# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:56:50 2018

@author: freeze
"""

import tensorflow as tf

def block(x,out_dim):
    """
    if input tensor dim equals output dim,use identity shortcut
    else use 1*1 conv increse dim
    """
    if(x.shape.as_list()[-1]==out_dim):
        short_cut=tf.identity(x)
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=out_dim,strides=1,padding='SAME'))
    else:
        short_cut=tf.layers.conv2d(x,kernel_size=1,filters=out_dim,strides=2,padding='SAME')
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=out_dim,strides=2,padding='SAME'))
    c2=tf.layers.conv2d(c1,kernel_size=3,filters=out_dim,strides=1,padding='SAME')
    return tf.nn.relu(c2+short_cut)

def bottleneck(x,mid_filter):
    """
    bottleneck residual v1 arch
    increase and decrease tensor x dim
    """
    fin_dim=x.shape.as_list()[-1]
    c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=1,filters=mid_filter,strides=1,padding='SAME'))
    c2=tf.nn.relu(tf.layers.conv2d(c1,kernel_size=3,filters=mid_filter,strides=1,padding='SAME'))
    c3=tf.layers.conv2d(c2,kernel_size=3,filters=fin_dim,strides=1,padding='SAME')
    return tf.nn.relu(x+c3)
    

def residual_v1_18(x):
    # x means input tensor with shape [None,224,224,channle]
    L2=tf.contrib.layers.l2_regularizer(5e-4)
    with tf.variable_scope(name_or_scope="res_net_v1_18",reuse=tf.AUTO_REUSE):
        conv1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=7,filters=64,strides=2,padding='SAME'))
        mp1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
        conv2_1=block(mp1,64)
        conv2_2=block(conv2_1,64)
        conv3_1=block(conv2_2,128)
        conv3_2=block(conv3_1,128)
        conv4_1=block(conv3_2,256)
        conv4_2=block(conv4_1,256)
        conv5_1=block(conv4_2,512)
        conv5_2=block(conv5_1,512)
        
        #fc3
        flat=tf.contrib.layers.flatten(conv5_2)
        fc1=tf.layers.dense(flat,units=1024,activation=tf.nn.relu,kernel_regularizer=L2)
        fc2=tf.layers.dense(fc1,units=256,activation=tf.nn.relu,kernel_regularizer=L2)
        fc3=tf.layers.dense(fc2,units=17,activation=tf.nn.relu,kernel_regularizer=L2)
        return fc3

def residual_v2_18(x):
    # x means input tensor with shape[None,32,32,channle]
    L2=tf.contrib.layers.l2_regularizer(5e-4)
    with tf.variable_scope(name_or_scope="res_net_v2_18",reuse=tf.AUTO_REUSE):
        bn1=bottleneck(x,32)
        bn2=bottleneck(bn1,32)
        block_64=block(bn2,64)
        bn3=bottleneck(block_64,64)
        bn4=bottleneck(bn3,64)
        block_128=block(bn4,128)
        bn5=bottleneck(block_128,128)
        bn6=bottleneck(bn5,128)
        block_256=block(bn6,256)
        bn6=bottleneck(block_256,256)
        bn7=bottleneck(bn6,256)
        
        #fc3
        flat=tf.contrib.layers.flatten(bn7)
        fc1=tf.layers.dense(flat,units=1024,activation=tf.nn.relu,kernel_regularizer=L2)
        fc2=tf.layers.dense(fc1,units=256,activation=tf.nn.relu,kernel_regularizer=L2)
        fc3=tf.layers.dense(fc2,units=17,activation=tf.nn.relu,kernel_regularizer=L2)
        return fc3
    
    
    