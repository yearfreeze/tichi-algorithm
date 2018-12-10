# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:34:44 2018

@author: freeze
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#data IO
samples_num=10000
strs='D:\\dataset\\training.h5'
h=h5py.File(strs,'r')


def conv_5_net(image,is_train):
	#momentum=0.9
	with tf.variable_scope('conv'):
		#define conv2d and batch_normal layers
		h0=tf.nn.relu(tf.layers.conv2d(image,kernel_size=3,filters=64,strides=2,padding='same'))
		h1=tf.nn.relu(tf.layers.conv2d(h0,kernel_size=3,filters=128,strides=2,padding='same'))
		h2=tf.nn.relu(tf.layers.conv2d(h1,kernel_size=3,filters=256,strides=2,padding='same'))
		h3=tf.contrib.layers.flatten(h2)
		h4=tf.layers.dense(h3,units=17)
		return tf.nn.softmax(h4),h4


for key in h.keys():
    print (h[key].name)
    print (h[key].shape)



#split train and test data
train_num=int(0.9*samples_num)
test_num=int(0.1*samples_num)

sen1=h['sen1'][0:samples_num]
sen2=h['sen2'][0:samples_num]
label=h['label'][0:samples_num]


train_index=np.random.choice(samples_num,train_num,replace=False)
test_index=np.array(list(set(range(samples_num))-set(train_index)))

train_index.sort()
train_index=list(train_index)
test_index.sort()
test_index=list(test_index)


sen1_split_train=sen1[train_index]
sen2_split_train=sen2[train_index]

sen1_split_test=sen1[test_index]
sen2_split_test=sen2[test_index]


label_train=label[train_index]
label_test=label[test_index]

x_train=np.concatenate([sen1_split_train,sen2_split_train],axis=3)
x_test=np.concatenate([sen1_split_test,sen2_split_test],axis=3)



#hypamater  batch_size=200  learning_rate=0.0005
batch_size=200
WIDTH=28
HEIGHT=28
DEEP=18
learning_rate=0.0005
iteration=5000
#define tensorflow graph

x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])
is_training=tf.placeholder(tf.bool)

#crop_x=tf.random_crop(x,[None,28,28,DEEP])
sotf_y_,y_=conv_5_net(x,is_training)

#var_name=[var for var in tf.trainable_variables() if var.name.startswith('conv')]
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train=optimizer.minimize(loss)


#estimator acc
pred_number=tf.argmax(y_,1)
correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))   
    
#init 
init=tf.global_variables_initializer()




sess=tf.Session()


sess.run(init)

train_acc=[]
test_acc=[]
loss_set=[]
crop_shape=(batch_size,WIDTH,HEIGHT,DEEP)

for i in range(iteration):
    choose_index=np.random.choice(len(x_train),size=batch_size,replace=False)
    #batch
    batch_x=x_train[choose_index]
    batch_y=label_train[choose_index]
    
    seed=np.random.randint(low=0,high=4)
    
    batch_xs=batch_x[:,seed:seed+28,seed:seed+28]
    #run traning step
    sess.run(train,feed_dict={x:batch_xs,y:batch_y,is_training:True})
    #sess.run(train,feed_dict={x:train_x,y:train_y})
    
    x_train_s=x_train[:,seed:seed+28,seed:seed+28]
    x_test_s=x_test[:,seed:seed+28,seed:seed+28]
  
    
    a=sess.run(accuracy,{x:x_train_s,y:label_train,is_training:False})
    b=sess.run(accuracy,{x:x_test_s,y:label_test,is_training:False})
    lose=sess.run(loss,{x:batch_xs,y:batch_y,is_training:False})
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


