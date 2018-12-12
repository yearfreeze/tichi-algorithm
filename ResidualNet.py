# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:55:08 2018

@author: freeze
"""

import tensorflow as tf

def block(x,ksize=3,filters=64,kstride=1):
    """
    x is input tensor
    this function to implention residual block
    """
    layer_name='block_'+str(filters)
    in_channle=x.get_shape().as_list()[-1]
    
    with tf.variable_scope(name_or_scope=layer_name,reuse=tf.AUTO_REUSE):
        if in_channle==filters:
            shortcut=tf.identity(x)
        else:
            if kstride==1:
                shortcut=tf.layers.conv2d(x,kernel_size=1,filters=filters,strides=kstride,padding='VALID')
            else:
                shortcut=tf.layers.conv2d(x,kernel_size=1,filters=filters,strides=2*kstride,padding='VALID')
        
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=ksize,filters=filters,strides=kstride,padding='SAME'))
        c2=tf.layers.conv2d(c1,kernel_size=ksize,filters=filters,strides=kstride,padding='SAME')
        
        output=tf.nn.relu(c2+shortcut)
    return output


def res18(x,kp):
    """
    x means input tensor
    kp means dropout prob
    """
    L2=tf.contrib.layers.l2_regularizer(0.5)
    shape=[1,2,2,1]
    with tf.variable_scope(name_or_scope='Res18',reuse=tf.AUTO_REUSE):
        conv1_x=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=64,strides=1,padding='SAME'))
        conv2_x=block(conv1_x,filters=64)
        conv2_x=tf.nn.max_pool(conv2_x,ksize=shape,strides=shape,padding='SAME')
        conv3_x=block(conv2_x,filters=128)
        conv3_x=tf.nn.max_pool(conv3_x,ksize=shape,strides=shape,padding='SAME')
        conv4_x=block(conv3_x,filters=256)
        conv4_x=tf.nn.max_pool(conv4_x,ksize=shape,strides=shape,padding='SAME')
        conv5_x=tf.nn.relu(tf.layers.conv2d(conv4_x,kernel_size=3,filters=512,strides=1,padding='SAME'))
        flat=tf.contrib.layers.flatten(conv5_x)
        #fully connected
        fc1=tf.layers.dense(flat,units=1024,activation=tf.nn.relu,kernel_regularizer=L2)
        fc1=tf.layers.dropout(fc1,rate=kp)
        fc2=tf.layers.dense(fc1,units=256,activation=tf.nn.relu,kernel_regularizer=L2)
        fc2=tf.layers.dropout(fc2,rate=kp)
        fc3=tf.layers.dense(fc2,units=17)
    return fc3
        
        
        
        
        
        
        
        
        
        
        
        