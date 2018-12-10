# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:40:29 2018


tichi 17分类结果
@author: freeze
"""

import tensorflow as tf
import h5py
import numpy as np
import op

sess=tf.Session()
#saver=tf.train.import_meta_graph('fine-tune-sen2/model.ckpt-1947.meta')
saver=tf.train.import_meta_graph('fine-tune-tune-sen2/model.ckpt-1646.meta')
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


#data IO
path='..\\dataset\\round1_test_a_20181109.h5'
h=h5py.File(path,'r')
sen2=h['sen2'].value

num=sen2.shape[0]
ans=np.zeros((num,17))

for i in range(num):
    
    img=sen2[i]
    img_90=np.array(op.rotate18(img,90,channle=10))
    img_180=np.array(op.rotate18(img,180,channle=10))
    img_270=np.array(op.rotate18(img,270,channle=10))
    
    fin_img=np.concatenate([img,img_90,img_180,img_270],axis=0)
    fin_img=fin_img.reshape(-1,32,32,10)
    
    
    y_=sess.run(pred_number,feed_dict={x:fin_img,keep_prob:1.0})  
    y_=np.argmax(np.bincount(y_))
    
    ans[i][y_]=1
    
np.savetxt("round1_test_a_20181109.csv",ans,delimiter=',')
    