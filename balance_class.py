# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:57:57 2018

@author: freeze
"""

import h5py
import numpy as np
"""
strs='D:\\dataset\\training.h5'
h=h5py.File(strs,'r')
for key in h.keys():
    print (h[key].name)
    print (h[key].shape)
    
hl=h['label'].value


dis_hl=np.sum(hl,axis=0) 


class_hl=np.argmax(hl,axis=1)

#init
L=[]
for i in range(17):
    L.append(list())
#fill in
for i in range(len(class_hl)):
    L[class_hl[i]].append(i)

for l in L:
    print (len(l))
    
minm=int(np.min(dis_hl))

NL=[]
for arr in L:
    if(len(arr)>minm):
        A=np.array(arr)
        choice=np.random.choice(len(arr),minm,replace=False)
        B=A[choice]
        B.sort()
        B=list(B)
        NL.append(B)
    else:
        NL.append(arr)
"""     
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

strs='D:\\dataset\\training.h5'

k=down_sampling(strs)
            
            




