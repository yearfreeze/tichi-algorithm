# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:24:14 2018

@author: freeze
"""
"""

1.训练集合样本均衡 
2.旋转data augumention 
3.valid集合data augumention
4.翻转data augumention    
5.增加了翻转操作  
6.增加指标 
7.21channle
8.resnet
9.cv2 resize=32

acc=86.4%
"""
from Resnet_v1 import residual_v2_18
#from resize import batch_resize
import op
from newDataLoader import Dataloader
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
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
        fin_x=op.single_data(temp_x)
      
        #print (fin_x.shape)
        y_=sess.run(pred_number,feed_dict={x:fin_x})
        y_=np.argmax(np.bincount(y_))
        label=np.argmax(y[i])
        if(y_==label):
            count=count+1
    return count/div

#hypamater  batch_size=200  learning_rate=0.0005
batch_size=64
WIDTH=32
HEIGHT=32
DEEP=16
learning_rate=0.0005
iteration=0
#define tensorflow graph


x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
y=tf.placeholder(tf.float32,shape=[None,17])
LR=tf.placeholder(tf.float32)


#crop_x=tf.random_crop(x,[None,28,28,DEEP])
y_=residual_v2_18(x)

#var_name=[var for var in tf.trainable_variables() if var.name.startswith('conv')]
with tf.variable_scope(name_or_scope='conv',reuse=tf.AUTO_REUSE):
	entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss=tf.add_n([entropy_loss]+regularization_losses)
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
paths='..\\..\\dataset\\training.h5'
indces=np.array(down_sampling(paths)) #transform to one dim
one_indces=indces.reshape(-1)
one_indces=list(one_indces)
one_indces.sort()

#data IO
dl=Dataloader(confidence=list(one_indces),epoch=5,total_num=len(one_indces))
valid_x,valid_y=dl.get_valid_sample(num=1000)




train_acc=[]
test_acc=[]
loss_set=[]
    
max_acc=0
saver=tf.train.Saver(max_to_keep=1)
while(dl.epoch>0):
    
    #batch from dataloader
    batch_x,batch_y=dl.batch_train_sample(batch_size=batch_size)
    batch_vx,batch_vy=dl.batch_valid_sample(batch_size=batch_size)
    batch_x=np.concatenate([batch_x,batch_vx],axis=0)
    batch_y=np.concatenate([batch_y,batch_vy],axis=0)
    
    #run traning step
    if (dl.epoch>0):
        sess.run(train,feed_dict={x:batch_x,y:batch_y,LR:learning_rate})
        lose=sess.run(loss,{x:batch_x,y:batch_y})
        #sess.run(train,feed_dict={x:train_x,y:train_y})
        a=sess.run(accuracy,{x:batch_x,y:batch_y})
        b=valid_acc(sess,valid_x,valid_y)
        print ("epoch %d step %d train batch acc = %f ,test acc = %f, loss = %f" % (dl.epoch,iteration,a,b,lose))
        if (b>max_acc):
            max_acc=b
            saver.save(sess,'new-start/model.ckpt',global_step=iteration)
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
