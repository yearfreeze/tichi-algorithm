# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:44:42 2019

@author: freeze
"""


from newDataLoader import Dataloader
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
        fin_x=op.single_data(temp_x)
      
        #print (fin_x.shape)
        y_=sess.run(pred_number,feed_dict={x:fin_x})
        y_=np.argmax(np.bincount(y_))
        label=np.argmax(y[i])
        if(y_==label):
            count=count+1
    return count/div


sess=tf.Session()
saver=tf.train.import_meta_graph('2019-1-7/model.ckpt-2146.meta')
saver.restore(sess,tf.train.latest_checkpoint('2019-1-7/'))

#default graph
graph=tf.get_default_graph()

#get placeholder and variable
x=graph.get_tensor_by_name("Placeholder:0")
y=graph.get_tensor_by_name("Placeholder_1:0")
LR=graph.get_tensor_by_name("Placeholder_2:0")

y_=graph.get_tensor_by_name("res_net_v2_18/dense_2/Relu:0")
loss=graph.get_tensor_by_name("conv/AddN:0")

pred_number=graph.get_tensor_by_name("ArgMax:0")
correct_pred=graph.get_tensor_by_name("Equal:0")
accuracy=graph.get_tensor_by_name("Mean:0")

train=graph.get_operation_by_name("conv/Adam")


#hypamater
batch_size=64
#LR=0.00005
learning_rate=0.0001 
iteration=0


#
paths='..\\..\\dataset\\training.h5'
indces=np.array(down_sampling(paths)) #transform to one dim
one_indces=indces.reshape(-1)
one_indces=list(one_indces)
one_indces.sort()

res_indces=list(set(range(352366))-set(one_indces))
#data IO
dl=Dataloader(confidence=list(res_indces),epoch=2,total_num=len(one_indces))
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
            saver.save(sess,'2019-1-8/model.ckpt',global_step=iteration)
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