# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:39:32 2018

@author: freeze
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv_5_net(image,kp):
    #momentum=0.9
    L2=tf.contrib.layers.l2_regularizer(0.5)
    with tf.variable_scope(name_or_scope='conv',reuse=tf.AUTO_REUSE):
        h0=tf.nn.relu(tf.layers.conv2d(image,kernel_size=3,filters=64,strides=2,padding='same'))
        h1=tf.nn.relu(tf.layers.conv2d(h0,kernel_size=3,filters=128,strides=2,padding='same'))
        h2=tf.nn.relu(tf.layers.conv2d(h1,kernel_size=3,filters=256,strides=2,padding='same'))
        h3=tf.contrib.layers.flatten(h2)
        
        #desen layers dropout-50%
        h4=tf.layers.dense(h3,units=1024,activation=tf.nn.relu,kernel_regularizer=L2)
        h4=tf.layers.dropout(h4,rate=kp)
        h5=tf.layers.dense(h4,units=256,activation=tf.nn.relu,kernel_regularizer=L2)
        h5=tf.layers.dropout(h5,rate=kp)
        h6=tf.layers.dense(h5,units=17)
        return tf.nn.softmax(h6),h6

#hypamater  batch_size=200  learning_rate=0.0005
batch_size=400
WIDTH=28
HEIGHT=28
DEEP=18
learning_rate=0.0005
iteration=5000
#define tensorflow graph

keep_prob=tf.placeholder(tf.float32)
x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])

#crop_x=tf.random_crop(x,[None,28,28,DEEP])
sotf_y_,y_=conv_5_net(x,keep_prob)

#var_name=[var for var in tf.trainable_variables() if var.name.startswith('conv')]
with tf.variable_scope(name_or_scope='conv',reuse=tf.AUTO_REUSE):
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
	train=optimizer.minimize(loss)

#estimator acc
pred_number=tf.argmax(y_,1)
correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))   
    
#init 
init=tf.global_variables_initializer()


sess=tf.Session()

sess.run(init)




#data IO

strs='D:\\dataset\\training.h5'
h=h5py.File(strs,'r')
for key in h.keys():
    print (h[key].name)
    print (h[key].shape)


holy_samples_num=h['label'].shape[0]

samples_num=10000  #step length
holy_start=0

while (holy_start<36):
    print ("-----------------------------------")
    print ("batch wave %d" % holy_start)
    print ("-----------------------------------")
    holy_end=min(holy_samples_num,(holy_start+1)*samples_num)
    sen1=h['sen1'][holy_start*samples_num:holy_end]
    sen2=h['sen2'][holy_start*samples_num:holy_end]
    label=h['label'][holy_start*samples_num:holy_end]
    
    
    x_train=np.concatenate([sen1,sen2],axis=3)
    
    train_acc=[]
    loss_set=[]
    
    crop_shape=(batch_size,WIDTH,HEIGHT,DEEP)
    for i in range(iteration):
        choose_index=np.random.choice(len(x_train),size=batch_size,replace=False)
        #batch
        batch_x=x_train[choose_index]
        batch_y=label[choose_index]
        
        seed=np.random.randint(low=0,high=32-WIDTH)
        
        batch_xs=batch_x[:,seed:seed+WIDTH,seed:seed+WIDTH]
        #run traning step
        sess.run(train,feed_dict={x:batch_xs,y:batch_y,keep_prob:0.5})
        #sess.run(train,feed_dict={x:train_x,y:train_y})
        
        x_train_s=x_train[:,seed:seed+WIDTH,seed:seed+WIDTH]
        
        
        a=sess.run(accuracy,{x:x_train_s,y:label,keep_prob:1.0})
        lose=sess.run(loss,{x:batch_xs,y:batch_y,keep_prob:1.0})
        print ("wave %d , step %d train acc = %f , loss = %f" % (holy_start,i,a,lose))
        train_acc.append(a)
        loss_set.append(lose)
        
    plt.plot(train_acc,'k-',label='Trainacc')
    plt.title('acc per generation')
    plt.xlabel('generation')
    plt.ylabel('acc')
    plt.legend(loc='lower right')
    plt.show()
    
    
    plt.plot(loss_set,'k-')
    plt.title('loss per generation')
    plt.xlabel('generation')
    plt.ylabel('loss')
    plt.show()
    holy_start=holy_start+1