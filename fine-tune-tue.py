# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:29:39 2018

@author: freeze
"""

from Sen2DataLoader import Dataloader
import tensorflow as tf
import h5py
import numpy as np
import op
import matplotlib.pyplot as plt

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
        rotate90=np.array(op.rotate18(temp_x,90,channle=10))
        rotate180=np.array(op.rotate18(temp_x,180,channle=10))
        rotate270=np.array(op.rotate18(temp_x,270,channle=10))
        
        fin_x=np.concatenate([temp_x,rotate90,rotate180,rotate270],axis=0)
        fin_x=fin_x.reshape(-1,32,32,10)
        #print (fin_x.shape)
        y_=sess.run(pred_number,feed_dict={x:fin_x,keep_prob:1.0})
        y_=np.argmax(np.bincount(y_))
        label=np.argmax(y[i])
        if(y_==label):
            count=count+1
    return count/div


sess=tf.Session()
saver=tf.train.import_meta_graph('fine-tune-sen2/model.ckpt-1947.meta')
saver.restore(sess,tf.train.latest_checkpoint('fine-tune-sen2/'))

#default graph
graph=tf.get_default_graph()

#get placeholder and variable
x=graph.get_tensor_by_name("Placeholder:0")
y=graph.get_tensor_by_name("Placeholder_1:0")
keep_prob=graph.get_tensor_by_name("Placeholder_2:0")
LR=graph.get_tensor_by_name("Placeholder_3:0")

y_=graph.get_tensor_by_name("conv/dense_2/BiasAdd:0")
loss=graph.get_tensor_by_name("conv_1/Mean:0")
#train=graph.get_operation_by_name("conv_1/Adam:0")
pred_number=graph.get_tensor_by_name("ArgMax:0")
correct_pred=graph.get_tensor_by_name("Equal:0")
accuracy=graph.get_tensor_by_name("Mean:0")

train=graph.get_operation_by_name("conv_1/Adam")


#hypamater
batch_size=200
learning_rate=0.000005
iteration=0


#
paths='..\\dataset\\training.h5'
indces=np.array(down_sampling(paths)) #transform to one dim
one_indces=indces.reshape(-1)
one_indces=list(one_indces)
one_indces.sort()

#data IO
dl=Dataloader(confidence=list(one_indces),epoch=10,total_num=len(one_indces))
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
        sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5,LR:learning_rate})
        lose=sess.run(loss,{x:batch_x,y:batch_y,keep_prob:1.0})
        #sess.run(train,feed_dict={x:train_x,y:train_y})
        a=sess.run(accuracy,{x:batch_x,y:batch_y,keep_prob:1.0})
        b=valid_acc(sess,valid_x,valid_y)
        print ("epoch %d step %d train batch acc = %f ,test acc = %f, loss = %f" % (dl.epoch,iteration,a,b,lose))
        if (b>max_acc):
            max_acc=b
            saver.save(sess,'fine-tune-tune-sen2/model.ckpt',global_step=iteration)
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