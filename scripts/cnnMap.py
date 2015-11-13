# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:51:38 2015

@author: frickjm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:22:30 2015

@author: frickjm
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import sys
import cPickle

from skimage.transform import resize 
from random import shuffle
from os import listdir
from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad


       


def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:                
                with open("../map/layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


sys.setrecursionlimit(10000)

"""Define some parameters"""
DUMP_WEIGHTS    = True
size            = 200
scaling         = 4
batch_size      = 8
chunkSize       = 2048
testChunkSize   = 256   
direct          = "../data/images"+str(size)+"/"
direct2         = "../data/locations"+str(size)+"/"

"""Calculate some parameters"""
outsize         = (size/scaling)**2
ld              = listdir(direct)
numEx           = len(ld)

"""Randomly shuffle the examples"""
shuffle(ld)

"""Split into test and train images"""
trainFs = ld[:int(numEx*0.8)]
testFs  = ld[int(numEx*0.8):]
trainL  = len(trainFs)
testL   = len(testFs)

print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL



numTrainEx      = min(trainL,chunkSize)
trainImages     = np.zeros((numTrainEx,1,size,size),dtype=np.float)
trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
#testImages      = np.zeros((testChunkSize,1,size,size),dtype=np.float)
#testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)

if len(sys.argv) <= 1:
    print "Must run with either 'python cnnMap.py new' or 'python cnnMap.py update'"
    sys.exit(1)

if sys.argv[1].lower().strip() == "update":
    with open("../map/wholeModel.pickle",'rb') as f:
        model     = cPickle.load(f)
elif sys.argv[1].lower().strip() == "new":
    model = Sequential()
    
    model.add(Convolution2D(32, 1, 4, 4, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(32, 32, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(64, 32, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(64, 64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(64, 64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    
    
    model.add(Flatten())
    model.add(Dense(64*5**2, outsize))
    model.add(Activation('relu'))
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

else:
    print "Must run with either 'python cnnMap.py new' or 'python cnnMap.py update'"
    sys.exit(1)
    

""" TRAINING """

#Define the number of iterations per epoch and the number of epochs    
numIterations   = trainL/chunkSize + 1
superEpochs     = 10

for sup in range(0,superEpochs):
    
    #for each epoch after the first, dump the model out
    if sup > 0:
        with open("../map/wholeModel.pickle", 'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(model)
            
            
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    
    
    
    for i in range(0,numIterations):

        if sup%5 == 0:
            with open("../map/wholeModel.pickle", 'wb') as f:
                cp     = cPickle.Pickler(f)
                cp.dump(model)        
        
        #Reset the training tensors
        trainL          = len(trainFs[i*chunkSize:(i+1)*chunkSize])
        numTrainEx      = min(trainL,chunkSize)
        trainImages     = np.zeros((numTrainEx,1,size,size),dtype=np.float)
        trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
        
        
        print "iteration ",i,": ", i*chunkSize," through ", (i+1)*chunkSize
        
        
        #Run through each training example in this chunk of data
        count   = 0
        for x in trainFs[i*chunkSize:(i+1)*chunkSize]:
            if x.find(".png") > -1:
                
                #Load the image
                CID     = x[:x.find(".png")]
                image   = io.imread(direct+x,as_grey=True)         
                image   = np.where(image > 0.01,1.0,0.0)
                
                #Load the target image, which contains the atom locations, resize it
                target  = io.imread(direct2+x,as_grey=True)
                target  = resize(target,(size/scaling,size/scaling))
                target  = np.where(target>0.01,50.,0.)

                
                #Add the image and the smaller target to their X , ytrain
                trainImages[count,0,:,:]    = image
                trainTargets[count]         = np.reshape(target,(outsize,))

                count+=1
                        
        print np.mean(target)                        
                
    
        #Fit the model using this training data
        model.fit(trainImages,trainTargets,batch_size=batch_size,nb_epoch=1)
        

        


#        count = 0
#        for x in testFs[i*testChunkSize:(i+1)*testChunkSize]:
#            if x.find(".png") > -1:
#                
#                #Load the image
#                CID     = x[:x.find(".png")]
#                image   = io.imread(direct+x,as_grey=True)         
#                image   = np.where(image > 0.01,1.0,0.0)
#                
#                #Load the target image, which contains the atom locations, resize it
#                target  = io.imread(direct2+x,as_grey=True)
#                target  = resize(target,(size/scaling,size/scaling))
#                target  = np.where(target>0.01,10000.,0.)
#
#                
#                #Add the image and the smaller target to their Xtrain, ytrain
#                testImages[count,0,:,:]    = image
#                testTargets[count]         = np.reshape(target,(outsize,))
#                
#                count+=1
#        
#        preds =  model.predict(testImages)       
#        
#        count2 = 0        
#        for x in preds:
#            print np.mean(x)
#
#            count2+=1

        if DUMP_WEIGHTS:
            dumpWeights(model)
        


del trainTargets, trainImages
""" END TRAINING """



