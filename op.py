import cv2
from skimage import transform
import numpy as np
"""
对这个数据集一张图片先进行水平翻转
得到两种表示，再配合0度，90度，180
度，270度的旋转，可以获得一张图的
八种表示
"""
def rotate18(image18,angle=90,channle=18):
	image_rotated=np.zeros((32,32,channle))
	for i in range(channle):
		image=image18[:,:,i]
		image_rotated[:,:,i]=transform.rotate(image,angle)
	return image_rotated

def transpose18(image18,flag=1,channle=18):
	image_transposed = np.zeros((32, 32, channle))
	for i in range(channle):
		image=image18[:,:,i]
		image_transposed[:,:,i]=cv2.flip(image,flag,dst=None)
	return image_transposed
	
def batch_rotate(image,label):
	"""
	image :[batch_size,weight,height,channle]
	label : one-hot 
	"""
	shape=image.shape
	if(len(shape)!=4):
		return None
	else:
		rotate_90_image=np.array([rotate18(var,90,shape[3]) for var in image])
		rotate_180_image=np.array([rotate18(var,180,shape[3]) for var in image])
		rotate_270_image=np.array([rotate18(var,270,shape[3]) for var in image])
		
		repeat_label=np.array([label]*3)
		repeat_label=repeat_label.reshape(-1,label.shape[1])
		
		fin_image=np.concatenate([image,rotate_90_image,rotate_180_image,rotate_270_image],axis=0)
		fin_label=np.concatenate([label,repeat_label],axis=0)
		
		index=np.random.choice(len(fin_label),size=len(fin_label),replace=False)
		return fin_image[index],fin_label[index]

		
		