# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:42:29 2015

@author: frickjm
"""

import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize 
import numpy as np
from os.path import isfile

from os import listdir
from random import shuffle
import cPickle

from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad


vmap     = False
vecfp    = True

if vmap == True:
    d     = "../map/layer"
    
elif vecfp == True:
    d     = "../ecfp/layer"
else:
    d     = "../molecularFormula/layer"

""" VISUALIZATION """

for i in range(0,100):
    if isfile(d+str(i)+".pickle"):
        with open(d+str(i)+".pickle",'rb') as f:
            weights     = cPickle.load(f)
        greymap     = plt.get_cmap('gray')
        size        = len(weights)
        if size < 100:
            print "visualizing layer ", i
            rows        = size/4
            #f, axarr    = plt.subplots(rows,8)
            count       = 0
            for x in weights:
                row     = count/4
                col     = count%4
                p   = x[0,:,:]
                plt.subplot(rows,rows,row*4+col+1)
                plt.axis('off')
                plt.imshow(p,cmap = greymap)
                #axarr[row,col].axis('off')
                #axarr[row,col].imshow(p,cmap = greymap)
                count   +=1
                
            plt.suptitle("Layer "+str(i))
            plt.savefig(d+str(i)+".jpg")
    


        
