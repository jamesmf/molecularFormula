# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:23:32 2015

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




#def visualize(model):
#    """ VISUALIZATION """
#    greymap     = plt.get_cmap('gray')
#        
#    layercount  = 0
#    for layer in model.layers:
#        try:
#            weights     = model.layers[layercount].get_weights()[0]
#            size        = len(weights)
#            if size < 100:
#                print "visualizing layer ", layercount
#                rows        = size/8
#                f, axarr    = plt.subplots(rows,8)
#                count       = 0
#                for x in weights:
#                    row     = count/8
#                    col     = count%8
#                    p   = x[0,:,:]
#                    axarr[row,col].axis('off')
#                    axarr[row,col].imshow(p,cmap = greymap)
#                    count   +=1
#                    
#                plt.suptitle("Layer "+str(layercount))
#                plt.savefig("../layer"+str(layercount)+".jpg")
#
#            else:
#                pass
#                
#        except IndexError:
#            pass
#        layercount  +=1




def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                print "visualizing layer ", layercount
                
                with open("../layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1



size    = 200
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"
ld      = listdir(direct)
numEx   = len(ld)




shuffle(ld)

trainFs = ld[:int(numEx*0.9)]
testFs  = ld[int(numEx*0.9):]
trainL  = len(trainFs)
testL   = len(testFs)

print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

batch_size      = 32
chunkSize       = 2000
numTrainEx      = min(trainL,chunkSize)

with open("../cidsMF.pickle",'rb') as f:
    mfs    = cPickle.load(f)
    
outsize         = len(mfs[mfs.keys()[0]])

windowSize  = 12
squares     = imdim - 2*windowSize
    

count   = 0
for x in trainFs:
    if x.find(".png") > -1:
        CID     = x[:x.find(".png")]
        image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
        image   = np.where(image > 0.1,1.0,0.0)

        for rowInd in range(0,squares):
            for colInd in range(0,squares):
                threshold   = windowSize**2/4.
                im  = image[rowInd:rowInd+windowSize,colInd:colInd+windowSize]
                if np.sum(im) > threshold:
                    plt.imshow(im)  
                    plt.show()
