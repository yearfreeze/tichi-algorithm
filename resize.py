# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:18:57 2018

@author: freeze
"""

import numpy as np
import cv2

def batch_resize(x,size=96):
    #input data has shape(None,32,32,channle)
    #return need be shape(None,224,224,channle)
    shape=x.shape
    num=shape[0]
    channle=shape[-1]
    L=list()
    for i in range(num):
        value=np.zeros((size,size,channle))
        for j in range(channle):
            value[:,:,j]=cv2.resize(x[i,:,:,j],(size,size),interpolation=cv2.INTER_LINEAR)
        L.append(value)
    return np.array(L)