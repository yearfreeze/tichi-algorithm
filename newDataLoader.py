# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:15:07 2018

1.sen1 4-5独立
2.sen2 10通道
3.4个指标
4.合并通道
5.数据增强

@author: freeze
"""


import h5py
import numpy as np
from op import batch_data

def get_index(array,index):
    P=[]
    for i in index:
        P.append(array[i])
    return P

#归一化差异水指数
def NDWI(sen2):
    ndwi=(sen2[:,:,1]-sen2[:,:,6])/(sen2[:,:,1]+sen2[:,:,6])*0.5+0.5
    return ndwi.reshape(32,32,1)
#归一化植被指数
def NDVI(sen2):
    ndvi=(sen2[:,:,6]-sen2[:,:,2])/(sen2[:,:,6]+sen2[:,:,2])*0.5+0.5
    return ndvi.reshape(32,32,1)
#归一化差异建筑指数
def NDBI(sen2):
    ndbi=(sen2[:,:,8]-sen2[:,:,6])/(sen2[:,:,8]+sen2[:,:,6])*0.5+0.5
    return ndbi.reshape(32,32,1)
#归一化燃烧指数
def NBR(sen2):
    nbr=(sen2[:,:,6]-sen2[:,:,9])/(sen2[:,:,6]+sen2[:,:,9])*0.5+0.5
    return nbr.reshape(32,32,1)


def shrink(data,channle):
    #"data" shape like(weight,height,channle)
    return (1-np.exp(np.negative(data[:,:,channle]))).reshape(32,32,1)


def normalize(data):
    #which has shape like(weight,height)
    ma=np.max(data)
    mi=np.min(data)
    return (data-mi)/(ma-mi)

def nor(tensor):
    #tensor means shape (samples,weight,height,channle)
    s=tensor.shape
    for i in range(s[0]):
        for j in range(s[-1]):
            tensor[i,:,:,j]=normalize(tensor[i,:,:,j])
    return tensor
    


def indicator(sen1,sen2):
    #input shape(batch_size,width,height,channle)
    #return 6 indicator and fix channel*2
    wi=np.array([NDWI(var) for var in sen2])
    vi=np.array([NDVI(var) for var in sen2])
    bi=np.array([NDBI(var) for var in sen2])
    br=np.array([NBR(var) for var in sen2])
    
    #print (isi.shape)
    s4=np.array([shrink(var,4) for var in sen1])
    s5=np.array([shrink(var,5) for var in sen1])
    #print (s3.shape)
    return np.concatenate([s4,s5],axis=3),np.concatenate([wi,vi,bi,br],axis=3)
    
class Dataloader(object):
    def __init__(self,confidence,epoch=2,total_num=352366):
        self.train_dir='..\\..\\dataset\\training.h5'
        self.valid_dir='..\\..\\dataset\\validation.h5'
        self.sample_space=h5py.File(self.train_dir,'r')
        self.valid_sapce=h5py.File(self.valid_dir,'r')
        self.total_num=total_num
        self.epoch=epoch
        self.confidence_list=confidence
        self.random_list=list(self.confidence_list)
        
    def get_valid_sample(self,num=5000):
        h=h5py.File(self.valid_dir,'r')
        s1=h['sen1'][0:num]
        #取可以可视化的两个通道
        #sen1=s1[:,:,:,4:6]
        s2=h['sen2'][0:num]
        #选择最后两个通道
      
        #s1=s1[:,:,:,4:6]
        #s3=indicator(s2)
        s3,s4=indicator(s1,s2)
        #s2=nor(s2)
        sample=np.concatenate([s3,s2,s4],axis=3)
        label=h['label'][0:num]
        return sample,label
    

        
    def batch_train_sample(self,batch_size=100):
        if (self.epoch==0):
            return None,None
        else:
            if (len(self.random_list)<batch_size):
                self.epoch=self.epoch-1
                self.random_list=list(self.confidence_list)
                return self.batch_train_sample(batch_size=batch_size)
            else:
                choose_index=np.random.choice(len(self.random_list),size=batch_size,replace=False)
                choose_index.sort()
                choose_index=list(choose_index)
                choose=get_index(self.random_list,choose_index)
                #h=self.sample_space
                s1=self.sample_space['sen1'][choose]
                s2=self.sample_space['sen2'][choose]
                #sen_x=s1[:,:,:,6]
                #sen_s=s1[:,:,:,7]
                #s1[:,:,:,6]=(sen_x**2+sen_s**2)**0.5
                #s1=s1[:,:,:,4:7]
                #s1=s1[:,:,:,4:6]
                #s3=indicator(s2)
                s3,s4=indicator(s1,s2)
                #s2=nor(s2)
                lab=self.sample_space['label'][choose]
                sam=np.concatenate([s3,s2,s4],axis=3)
                x,y=batch_data(sam,lab)
                
                #delete recodes has choose
                for c in sorted(choose_index,reverse=True):
                    self.random_list.pop(c)
                return x,y
            
    def batch_valid_sample(self,batch_size=100):
        num=self.valid_sapce['label'].shape[0]
        choose=np.random.choice(num,batch_size,replace=False)
        choose.sort()
        choose=list(choose)
        
        s1=self.valid_sapce['sen1'][choose]
        s2=self.valid_sapce['sen2'][choose]
        #sen_x=s1[:,:,:,6]
        #sen_s=s1[:,:,:,7]
        #s1[:,:,:,6]=(sen_x**2+sen_s**2)**0.5
        #s1=s1[:,:,:,4:6]
        #s3=indicator(s2)
        s3,s4=indicator(s1,s2)
        #s2=nor(s2)
        lab=self.valid_sapce['label'][choose]
        sam=np.concatenate([s3,s2,s4],axis=3)
        x,y=batch_data(sam,lab)
        return x,y