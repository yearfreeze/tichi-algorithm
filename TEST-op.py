# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:04:36 2018

@author: freeze
"""

from DataLoader import Dataloader


dl=Dataloader(total_num=320)

train_x,train_y=dl.get_train_sample()

while(dl.epoch>0):
    batch_x,batch_y=dl.batch_train_sample()
    print ("epoch %d , random_list %d " % (dl.epoch,len(dl.random_list)))
    