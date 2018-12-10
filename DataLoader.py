# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:22:20 2018

@author: freeze
"""
import h5py
import numpy as np
from op import batch_rotate

def get_index(array,index):
    P=[]
    for i in index:
        P.append(array[i])
    return P

class Dataloader(object):
    def __init__(self,confidence,epoch=2,total_num=352366):
        self.train_dir='..\\dataset\\training.h5'
        self.valid_dir='..\\dataset\\validation.h5'
        self.sample_space=h5py.File(self.train_dir,'r')
        self.total_num=total_num
        self.epoch=epoch
        self.confidence_list=confidence
        self.random_list=list(self.confidence_list)
        
    def get_valid_sample(self,num=5000):
        h=h5py.File(self.valid_dir,'r')
        s1=h['sen1'][0:num]
        s2=h['sen2'][0:num]
        sample=np.concatenate([s1,s2],axis=3)
        label=h['label'][0:num]
        return sample,label
    
    def get_train_sample(self):
        h=h5py.File(self.train_dir,'r')
        s1=h['sen1'][0:self.total_num]
        s2=h['sen2'][0:self.total_num]
        sample=np.concatenate([s1,s2],axis=3)
        label=h['label'][0:self.total_num]
        #self.train_sample=sample
        #self.train_label=label
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
                lab=self.sample_space['label'][choose]
                sam=np.concatenate([s1,s2],axis=3)
                x,y=batch_rotate(sam,lab)
                
                #delete recodes has choose
                for c in sorted(choose_index,reverse=True):
                    self.random_list.pop(c)
                return x,y
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                