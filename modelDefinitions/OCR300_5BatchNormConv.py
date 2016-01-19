# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""

import sys
sys.path.append("../scripts/")
import helperFuncs
import numpy as np
from os import listdir
from random import shuffle
import cPickle

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import SGD, Adadelta, Adagrad
sys.setrecursionlimit(10000)
np.random.seed(0)
"""Require an argument specifying whether this is an update or a new model, parse input"""
#update, size, lay1size, run     = handleArgs(sys.argv,size=300)


def getLastSize(model):
    i   = -1
    while True:
        try:
            print model.layers[i].get_output(False)
            W   = model.layers[i].W_shape
            break
        except AttributeError as e:
            z = e
            print z
            i -= 1
    return W


"""Define parameters of the run"""

size            = 300           #EDIT ME!   #how large the images are
outType         = "ocr" #EDIT ME!   #what the CNN is predicting

imdim           = size - 20                 #strip 10 pixels buffer from each size
direct          = "../data/SDF/"            #directory containing the SD files
ld              = listdir(direct)                   #contents of that directory
shuffle(ld)                                 #shuffle the image list for randomness
numEx           = len(ld)                   #number of images in the directory
DUMP_WEIGHTS    = True                      #will we dump the weights of conv layers for visualization
trainTestSplit  = 0.90                      #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 6000                      #how many examples to evaluate per iteration
run             = "1"





"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(False,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
testDirect      = folder+"tempTest/"

#if update:     
#    stop = raw_input("Loading from folder "+folder+" : Hit enter to proceed or ctrl+C to cancel")
#else:
#    print "Initializing in folder "+folder





"""Load the train/test split information if update, else split and write out which images are in which dataset"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(False,folder,numEx,trainTestSplit,ld)
trainL  = len(trainFs)
testL   = len(testFs)


print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

features,labels  = helperFuncs.getTargets("ocr")            #get the target vector for each CID
outsize             = len(features[features.keys()[0]])  #this it the size of the target (# of OCRfeatures)

"""DEFINE THE MODEL HERE"""  

model = Sequential()

model.add(Convolution2D(8, 1, 5, 5, border_mode='full')) 
model.add(Activation('relu'))

model.add(MaxPooling2D(poolsize=(2,2)))
lastLayerSize   = getLastSize(model)
print lastLayerSize
print model.layers[-1]
model.add(BatchNormalization())

model.add(Convolution2D(16, 8, 5, 5, border_mode='full')) 
model.add(Activation('relu'))

model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(BatchNormalization())

model.add(Convolution2D(32, 16, 5, 5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 32, 5, 5)) 
model.add(Activation('relu'))    

model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(BatchNormalization())


model.add(Convolution2D(64, 64, 5, 5)) 
model.add(Activation('relu'))

model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(BatchNormalization())


model.add(Convolution2D(128, 64, 4, 4)) 
model.add(Activation('relu'))

model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4608, 512, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, outsize, init='normal'))


model.compile(loss='mean_squared_error', optimizer='adadelta')

#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))



#with open(folder+"bestModel.pickle", 'wb') as f:
#    cp     = cPickle.Pickler(f)
#    cp.dump(model)        

with open(folder+"wholeModel.pickle", 'wb') as f:
    cp     = cPickle.Pickler(f)
    cp.dump(model) 



