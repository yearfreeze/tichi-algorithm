# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:19:53 2018

@author: freeze
"""

import numpy as np
import random

"""
data whitening to scale mean 0 std 1
"""
def whiten(images):
	for i in range(images.shape[0]):
		old_image = images[i,:,:,:]
		new_image = (old_image - np.mean(old_image)) / np.std(old_image)
		images[i,:,:,:] = new_image
		
	return images

	
	
	
def crop(images, shape):
	old_shape=images.shape
	res=old_shape[1]-shape[1]
	index=np.random.randint(low=0,high=res)
	#print (index)
	return images[:,index:index+shape[1],index:index+shape[2],:]


	
def image_noise(images, mean=0, std=0.01):
	for i in range(images.shape[0]):
		for j in range(images.shape[3]):
			for m in range(images.shape[1]):
				for n in range(images.shape[2]):
					images[i,m,n,j]=images[i,m,n,j]+random.gauss(mean,std)
	return images

"""
x=np.random.random((50,32,32,18))
#y=crop(x,(50,28,28,18))
#xi=x[0]
#yi=y[0]
y=image_noise(x)
xi=x[0]
yi=y[0]
"""