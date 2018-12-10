# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:50:43 2018

@author: freeze
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#IO
h=h5py.File("C:\\Users\\freeze\\Desktop\\round1_test_a_20181109.h5",'r')
df=pd.read_csv('C:\\Users\\freeze\\Desktop\\sample.csv',encoding='gbk')


def conv_5_net(image):
    #momentum=0.9
    with tf.variable_scope('conv'):
        h0=tf.nn.relu(tf.layers.conv2d(image,kernel_size=3,filters=64,strides=2,padding='same'))
        h1=tf.nn.relu(tf.layers.conv2d(h0,kernel_size=3,filters=128,strides=2,padding='same'))
        h2=tf.nn.relu(tf.layers.conv2d(h1,kernel_size=3,filters=256,strides=2,padding='same'))
        h3=tf.contrib.layers.flatten(h2)
        h4=tf.layers.dense(h3,units=17)
        return tf.nn.softmax(h4),h4


for key in h.keys():
    print (h[key].name)
    print (h[key].shape)
    #print (h[key].value)
"""
im=h['sen2'][10]
red_channle=(im[:,:,2].reshape(32,32,1))*255
gree_channle=(im[:,:,1].reshape(32,32,1))*255
blue_channle=(im[:,:,0].reshape(32,32,1))*255
RGB_im=np.concatenate([red_channle,gree_channle,blue_channle],axis=2)

plt.imshow(RGB_im)
plt.show()
"""
data_x=h['sen1']
data_y=df.values


#split train and test data
train_index=np.random.choice(len(data_y),round(len(data_y)*0.8),replace=False)
test_index=np.array(list(set(range(len(data_y)))-set(train_index)))

train_index.sort()
train_index=list(train_index)
test_index.sort()
test_index=list(test_index)

train_x=data_x[train_index]
train_y=data_y[train_index]
test_x=data_x[test_index]
test_y=data_y[test_index]


#hypamater 
batch_size=100
WIDTH=32
HEIGHT=32
DEEP=8
learning_rate=0.001
iteration=1000
#define tensorflow graph

x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])
sotf_y_,y_=conv_5_net(x)

#var_name=[var for var in tf.trainable_variables() if var.name.startswith('conv')]
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
train=optimizer.minimize(loss)

#estimator acc
pred_number=tf.argmax(y_,1)
correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))   
    
#init 
init=tf.global_variables_initializer()


config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4 
sess=tf.Session(config=config)



sess.run(init)

train_acc=[]
test_acc=[]
loss_set=[]
for i in range(iteration):
    choose_index=np.random.choice(len(train_x),size=batch_size,replace=False)
    #batch
    batch_x=train_x[choose_index]
    batch_y=train_y[choose_index]
    #run traning step
    sess.run(train,feed_dict={x:batch_x,y:batch_y})
    #sess.run(train,feed_dict={x:train_x,y:train_y})
    
    a=sess.run(accuracy,{x:train_x,y:train_y})
    b=sess.run(accuracy,{x:test_x,y:test_y})
    lose=sess.run(loss,{x:batch_x,y:batch_y})
    print ("step %d train acc = %f ,test acc = %f, loss = %f" % (i,a,b,lose))
    train_acc.append(a)
    test_acc.append(b)
    loss_set.append(lose)
        

plt.plot(train_acc,'k-',label='Trainacc')
plt.plot(test_acc,'r--',label='testacc')
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








