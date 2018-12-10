# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:19:30 2018

@author: freeze
"""


import resnet
from DataLoader import Dataloader
#import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
def conv_5_net(image,kp):
    #momentum=0.9
    L2=tf.contrib.layers.l2_regularizer(0.5)
    with tf.variable_scope(name_or_scope='conv',reuse=tf.AUTO_REUSE):
        h0=tf.nn.relu(tf.layers.conv2d(image,kernel_size=3,filters=64,strides=2,padding='same'))
        h0=tf.nn.relu(tf.layers.conv2d(h0,kernel_size=3,filters=64,strides=1,padding='same'))
        h0=tf.nn.relu(tf.layers.conv2d(h0,kernel_size=3,filters=64,strides=1,padding='same'))
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
"""

#hypamater  batch_size=200  learning_rate=0.0005
batch_size=50
WIDTH=32
HEIGHT=32
DEEP=18
learning_rate=0.0001
iteration=0
#define tensorflow graph

keep_prob=tf.placeholder(tf.float32)
x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])
is_training = tf.placeholder(tf.bool, name = 'training')
#learning_rate = tf.placeholder(tf.float32)

#crop_x=tf.random_crop(x,[None,28,28,DEEP])
y_=resnet.resnet_110(x,keep_prob,is_training)

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


sess=tf.Session()

sess.run(init)




#data IO
dl=Dataloader(epoch=4,total_num=50000)
valid_x,valid_y=dl.get_valid_sample(num=5000)
train_x,train_y=dl.get_train_sample()

train_acc=[]
test_acc=[]
loss_set=[]
    


while(dl.epoch>0):
    
    #batch from dataloader
    batch_x,batch_y=dl.batch_train_sample(batch_size=batch_size)
    #run traning step
    if (dl.epoch>0):
        sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5,is_training:True})
        lose=sess.run(loss,{x:batch_x,y:batch_y,keep_prob:1.0,is_training:False})
    #sess.run(train,feed_dict={x:train_x,y:train_y})    
        a=sess.run(accuracy,{x:batch_x,y:batch_y,keep_prob:1.0,is_training:False})
        b=sess.run(accuracy,{x:valid_x,y:valid_y,keep_prob:1.0,is_training:False}) 
        print ("epoch %d step %d train batch acc = %f ,valid acc = %f, loss = %f" % (dl.epoch,iteration,a,b,lose))
        train_acc.append(a)
        test_acc.append(b)
        loss_set.append(lose)
    iteration=iteration+1
    
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
