# -*- coding: utf-8 -*-
"""
Created on Nov 12 15:22:30 2015

@author: frickjm
"""

import helperFuncs

import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize 
import numpy as np

from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import sys

from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad


sys.setrecursionlimit(10000)
np.random.seed(0)


"""*************************************************************************"""
"""*************************************************************************"""

"""get the solubility for training"""
def getTargets():
    out     = {}
    with open("../data/sols.pickle",'rb') as f:
        d =  cPickle.load(f)
    for k,v in d.iteritems():
        out[k] = [float(v)]
    return out

"""Dump the weights of the model for visualization"""
def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                with open(folder+"layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


"""Find out the RMSE of just guessing the mean solubility for comparison purposes"""
def testAverages(direct,targets):
    means = np.mean(targets.values(),axis=0)  
    s     = len(means)
    ld    = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,s),dtype=np.float)
    y       = np.zeros((num,s),dtype=np.float)
    count   = 0
    for x in ld[:num]:
        CID     = x[:x.find(".png")]
        y[count]  = targets[CID]
        preds[count] = means
        count+=1
   
    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


"""*************************************************************************"""
"""*************************************************************************"""

"""Require an argument specifying whether this is an update or a new model, parse input"""
update, size, lay1size, run     = helperFuncs.handleArgs(sys.argv)



"""Define parameters of the run"""
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"    #directory containing the images
ld      = listdir(direct)                   #contents of that directory
numEx   = len(ld)                           #number of images in the directory
shuffle(ld)                                 #shuffle the image list for randomness
outType = "solubility"                      #what the CNN is predicting
DUMP_WEIGHTS = True                         #will we dump the weights of conv layers for visualization
trainTestSplit   = 0.90                     #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 5000                      #how many examples to evaluate per iteration

"""Define the folder where the model will be stored based on the input arguments"""
folder     = helperFuncs.defineFolder(outType,size,lay1size,run)


"""Load the train/test split information if update, else split and write out which images are in which dataset"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(update,folder,numEx,trainTestSplit)
trainL  = len(trainFs)
testL   = len(testFs)
   

print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL


#batch_size      = 32            #how many training examples per batch
#chunkSize       = 5000          #how much data to ever load at once      
#testChunkSize   = 600           #how many examples to evaluate per iteration
numTrainEx      = min(trainL,chunkSize)

targets           = helperFuncs.getSolubilityTargets()     #get the solubility value for each CID   
outsize         = len(targets[targets.keys()[0]])          #this it the size of the target (# of targets)

"""Initialize empty matrices to hold our images and our target vectors"""
trainImages     = np.zeros((numTrainEx,1,imdim,imdim),dtype=np.float)
trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
testImages      = np.zeros((testChunkSize,1,imdim,imdim),dtype=np.float)
testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)


"""If we are training a new model, define it"""   
if sys.argv[1].lower().strip() == "new":
    model = Sequential()
    
    model.add(Convolution2D(32, 1, lay1size, lay1size, border_mode='full')) 
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 32, lay1size, lay1size, border_mode='full')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(32, 32, 5, 5))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 32, 5, 5)) 
    model.add(Activation('relu'))    
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(64, 64, 5, 5)) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(64, 64, 4, 4)) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(4096, 512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, outsize, init='normal'))
    
    
    model.compile(loss='mean_squared_error', optimizer='adadelta')

    model.set_weights(helperFuncs.getWeights("../OCRfeatures/300_5_1/wholeModel.pickle"))


"""If we are continuing to train an old model, load it"""
if update:
    with open(folder+"wholeModel.pickle",'rb') as f:
        model     = cPickle.load(f)



""" TRAINING """

    
numIterations   = trainL/chunkSize + 1
superEpochs     = 100
RMSE            = 1000000
oldRMSE         = 1000000
for sup in range(0,superEpochs):

    shuffle(trainFs)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    
    for i in range(0,numIterations):
        print "iteration ",i,": ", i*chunkSize," through ", (i+1)*chunkSize
        count   = 0
        for x in trainFs[i*chunkSize:(i+1)*chunkSize]:
            if x.find(".png") > -1:
                CID     = x[:x.find(".png")]
                image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
                #image   = np.where(image > 0.1,1.0,0.0)
                trainImages[count,0,:,:]    = image
                trainTargets[count]         = targets[CID]
                count +=1
    
        model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)
        


        if oldRMSE == RMSE:  
            if DUMP_WEIGHTS:
                dumpWeights(model)
    
            with open(folder+"bestModel.pickle", 'wb') as f:
                cp     = cPickle.Pickler(f)
                cp.dump(model)        

        else:
            with open(folder+"wholeModel.pickle", 'wb') as f:
                cp     = cPickle.Pickler(f)
                cp.dump(model) 
            


    shuffle(testFs)
    count   = 0
    for x in testFs[:testChunkSize]:
        if x.find(".png") > -1:
            CID     = x[:x.find(".png")]
            image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
            #image   = np.where(image > 0.1,1.0,0.0)
            testImages[count,0,:,:]    = image
            testTargets[count]         = targets[CID]
            count +=1
    
    preds   = model.predict(testImages)
    RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
    print "RMSE of epoch: ", RMSE
    
    oldRMSE     = min(oldRMSE,RMSE)

