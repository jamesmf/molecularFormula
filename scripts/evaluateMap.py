# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:23:07 2015

@author: frickjm
"""



import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize 
import numpy as np

from os import listdir
from random import shuffle
import cPickle

from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad


  
size    = 200
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"
direct2 = "../data/locations"+str(size)+"/"
ld      = listdir(direct)
numEx   = len(ld)

shuffle(ld)


testFs  = ld[int(numEx*0.8):]
testL   = len(testFs)

print "number of examples: ", numEx
print "test examples : ", testL


slideSize       = 64
batch_size      = 128
scaling         = 4
chunkSize       = 2048
testChunkSize   = 256
skipFactor      = 16
numTrainEx      = min(testL,chunkSize)
    
outsize         = (size/scaling)**2

#testImages      = np.zeros((testChunkSize,1,slideSize,slideSize),dtype=np.float)
#testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)
holder           = np.zeros((1,1,size,size),dtype=np.float)

PRED_THRESHOLD  = 0.6


with open("../map/wholeModel.pickle", 'rb') as f:
    model     = cPickle.load(f)

for x in testFs[0:chunkSize]:
    if x.find(".png") > -1:
        CID     = x[:x.find(".png")]
        image   = io.imread(direct+x,as_grey=True)         
        image   = np.where(image > 0.1,1.0,0.0)
        target  = io.imread(direct2+x,as_grey=True)
        holder[0,0,:,:] = image
              
        tmp        = model.predict(holder)[0]  
        
        imageOut   = np.reshape(tmp,(size/scaling,size/scaling))

        plt.figure(0)    
        plt.gcf().suptitle("Original Image")                    
        plt.imshow(image)
        plt.figure(1)
        plt.gcf().suptitle("Map")        
        plt.imshow(imageOut)

        plt.show()
            # Pick a window
