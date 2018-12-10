# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:45:54 2018

@author: freeze
"""


import op
import resnet
from DataLoader import Dataloader
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
#from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def down_sampling(path):
    h=h5py.File(path,'r')
    gaint_label=h['label'].value
    #每类的样本数目
    class_num=np.sum(gaint_label,axis=0)
    
    down_samp_num=int(np.min(class_num))
    
    roman_label=np.argmax(gaint_label,axis=1)
    #init 
    L=[]
    for i in range(17):
        L.append(list())
    #fill in L
    for it in range(len(roman_label)):
        L[roman_label[it]].append(it)
    
    BAL=[]
    for part in L:
        if(len(part)>down_samp_num):
            A=np.array(part)
            ch=np.random.choice(len(part),down_samp_num,replace=False)
            B=A[ch]
            B.sort()
            B=list(B)
            BAL.append(B)
        else:
            part.sort()
            BAL.append(part)
    return BAL

def valid_acc(sess,valid_x,y):
    div=y.shape[0]
    count=0
    for i in range(div):
        temp_x=valid_x[i]
        #print (temp_x.shape)
        rotate90=np.array(op.rotate18(temp_x,90))
        rotate180=np.array(op.rotate18(temp_x,180))
        rotate270=np.array(op.rotate18(temp_x,270))
        
        fin_x=np.concatenate([temp_x,rotate90,rotate180,rotate270],axis=0)
        fin_x=fin_x.reshape(-1,32,32,18)
        #print (fin_x.shape)
        y_=sess.run(pred_number,feed_dict={x:fin_x,keep_prob:1.0,is_training:False})
        y_=np.argmax(np.bincount(y_))
        label=np.argmax(y[i])
        if(y_==label):
            count=count+1
    return count/div


#hypamater  batch_size=200  learning_rate=0.0005
batch_size=50
WIDTH=32
HEIGHT=32
DEEP=18
learning_rate=0.001
iteration=0
#define tensorflow graph


x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])
LR=tf.placeholder(tf.float32)
keep_prob=tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, name = 'training')


#network
y_=resnet.resnet_110(x,keep_prob,is_training)

#var_name=[var for var in tf.trainable_variables() if var.name.startswith('conv')]

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
optimizer=tf.train.AdamOptimizer(learning_rate=LR,beta1=0.5)
train=optimizer.minimize(loss)


#estimator acc
pred_number=tf.argmax(y_,1)
correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))   
    
#init 
init=tf.global_variables_initializer()


sess=tf.Session()

sess.run(init)




#
paths='..\\dataset\\training.h5'
indces=np.array(down_sampling(paths)) #transform to one dim
one_indces=indces.reshape(-1)
one_indces=list(one_indces)
one_indces.sort()

#data IO
dl=Dataloader(confidence=list(one_indces),epoch=3,total_num=len(one_indces))
valid_x,valid_y=dl.get_valid_sample(num=1000)



train_acc=[]
test_acc=[]
loss_set=[]
    


while(dl.epoch>0):
    
    #batch from dataloader
    batch_x,batch_y=dl.batch_train_sample(batch_size=batch_size)
    #run traning step
    if (dl.epoch>0):
        sess.run(train,feed_dict={x:batch_x,y:batch_y,LR:learning_rate,keep_prob:0.5,is_training:True})
        lose=sess.run(loss,{x:batch_x,y:batch_y,keep_prob:1.0,is_training:False})
        #
        a=sess.run(accuracy,{x:batch_x,y:batch_y,keep_prob:1.0,is_training:False})
        b=valid_acc(sess,valid_x,valid_y) 
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
