# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


import helperFuncs 

#import matplotlib.pyplot as plt
import skimage.io as io
#from skimage.transform import resize 
import numpy as np

from os import listdir
#from os.path import isdir
#from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import sys
import subprocess
import time

from sklearn.metrics import mean_squared_error

#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, Adadelta, Adagrad


sys.setrecursionlimit(10000)
np.random.seed(0)

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


#def testAverages(direct,OCRfeatures):
#    means = np.mean(OCRfeatures.values(),axis=0)    
#    s   = len(means)
#    ld  = listdir(direct)
#    shuffle(ld)
#    num     = 20000
#    preds   = np.zeros((num,s),dtype=np.float)
#    y       = np.zeros((num,s),dtype=np.float)
#    count   = 0
#    for x in ld[:num]:
#        CID     = x[:x.find(".png")]
#        y[count,:]  = OCRfeatures[CID]
#        preds[count,:] = means
#        count+=1
#   
#    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


"""Require an argument specifying whether this is an update or a new model, parse input"""
size, run, outType     = helperFuncs.handleArgs(sys.argv)


"""Define parameters of the run"""
#imdim           = size - 20                         #strip 10 pixels buffer from each size
#direct          = "../data/SDF/"            #directory containing the SD files
#ld              = listdir(direct)                   #contents of that directory
#shuffle(ld)                                 #shuffle the image list for randomness
#numEx           = len(ld)                   #number of images in the directory
#outType         = "OCRfeatures"             #what the CNN is predicting
DUMP_WEIGHTS    = False                     #will we dump the weights of conv layers for visualization
#trainTestSplit  = 0.90                      #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
#chunkSize       = 50000                     #how much data to ever load at once      
#testChunkSize   = 6000                      #how many examples to evaluate per iteration

"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(True,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
trainNP         = folder+"tempTrainNP/"
testDirect      = folder+"tempTest/"
testNP          = folder+"tempTestNP/"

#if update:     
#    stop = raw_input("Loading from folder "+folder+" : Hit enter to proceed or ctrl+C to cancel")
#else:
#    print "Initializing in folder "+folder





"""Load the train/test split information"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(True,folder)

trainL  = len(trainFs)
testL   = len(testFs)


features,labels     = helperFuncs.getTargets("ocr") #get the OCR vector for each CID
outsize             = len(features[features.keys()[0]]) #this it the size of the target (# of OCRfeatures)

#"""If we are training a new model, define it"""   
#print "loading model"
#if sys.argv[1].lower().strip() == "new":
#    model = Sequential()
#    
#    model.add(Convolution2D(32, 1, lay1size, lay1size, border_mode='full')) 
#    model.add(Activation('relu'))
#
#    model.add(Convolution2D(32, 32, lay1size, lay1size, border_mode='full')) 
#    model.add(Activation('relu'))
#    
#    model.add(MaxPooling2D(poolsize=(2, 2)))
#    
#    model.add(Convolution2D(32, 32, 5, 5))
#    model.add(Activation('relu'))
#    
#    model.add(Convolution2D(64, 32, 5, 5)) 
#    model.add(Activation('relu'))    
#    
#    model.add(MaxPooling2D(poolsize=(2, 2)))
#    
#    model.add(Convolution2D(64, 64, 5, 5)) 
#    model.add(Activation('relu'))
#    
#    model.add(MaxPooling2D(poolsize=(2, 2)))
#    
#    model.add(Convolution2D(64, 64, 4, 4)) 
#    model.add(Activation('relu'))
#    
#    model.add(MaxPooling2D(poolsize=(2, 2)))
#    model.add(Dropout(0.25))
#    
#    model.add(Flatten())
#    model.add(Dense(4096, 512, init='normal'))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    
#    model.add(Dense(512, outsize, init='normal'))
#    
#    
#    model.compile(loss='mean_squared_error', optimizer='adadelta')
#
#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

"""load model"""
with open(folder+"wholeModel.pickle",'rb') as f:
    model     = cPickle.load(f)



""" TRAINING """

    
#numIterations   = trainL/chunkSize + 1
superEpochs     = 10000
RMSE            = 1000000
oldRMSE         = 1000000
for sup in range(0,superEpochs):
     
    oldRMSE     = min(oldRMSE,RMSE)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    

    count   = 0
    added   = 0

    #Wait for the other processes to dump a pickle file
    while not isfile(trainNP+"Xtrain.pickle"):   
        print "sleeping because Train folder empty             \r",
        time.sleep(1.)
    print ""

    
    #Load the training data   
    print "Loading np  training arrays"
    while True:
        try:
            with open(trainNP+"Xtrain.pickle",'rb') as f:
                trainImages     = cPickle.load(f)
        
            with open(trainNP+"ytrain.pickle",'rb') as f:
                trainTargets    = cPickle.load(f)
        except Exception as e:
            print e, "                                   \r",
    print ""

    subprocess.call("rm "+trainNP+"Xtrain.pickle",shell=True)
    subprocess.call("rm "+trainNP+"ytrain.pickle",shell=True)

    #train the model on it
    model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)

   
    del trainImages, trainTargets


    while not isfile(testNP+"Xtest.pickle"):
        print "sleeping because Test folder empty             \r",
        time.sleep(1.)
    print ""

    print "Loading np test arrays" 

    while True:       
        try:
            with open(testNP+"Xtest.pickle",'rb') as f:
                testImages     = cPickle.load(f)
        
            with open(testNP+"ytest.pickle",'rb') as f:
                testTargets    = cPickle.load(f)

        except Exception as e:
            print e, "                           \r", 
            
    print ""

    subprocess.call("rm "+testNP+"Xtest.pickle",shell=True)
    subprocess.call("rm "+testNP+"ytest.pickle",shell=True)

    preds   = model.predict(testImages)
    RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
    print "RMSE of epoch: ", RMSE


    del testImages, testTargets    

    
    if oldRMSE > RMSE:
        if DUMP_WEIGHTS:
            dumpWeights(model)

        with open(folder+"bestModel.pickle", 'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(model)        

    else:
        with open(folder+"wholeModel.pickle", 'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(model) 



