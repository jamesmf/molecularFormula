# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


from helperFuncs import defineFolder, handleArgs, getTrainTestSplit, getOCRTargets, getOCRScaledTargets, getWeights

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
import subprocess

from sklearn.metrics import mean_squared_error

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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


def testAverages(direct,OCRfeatures):
    means = np.mean(OCRfeatures.values(),axis=0)    
    s   = len(means)
    ld  = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,s),dtype=np.float)
    y       = np.zeros((num,s),dtype=np.float)
    count   = 0
    for x in ld[:num]:
        CID     = x[:x.find(".png")]
        y[count,:]  = OCRfeatures[CID]
        preds[count,:] = means
        count+=1
   
    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


"""Require an argument specifying whether this is an update or a new model, parse input"""
update, size, lay1size, run     = handleArgs(sys.argv)


"""Define parameters of the run"""
imdim           = size - 20                         #strip 10 pixels buffer from each size
direct          = "../data/SDF/"            #directory containing the SD files
ld              = listdir(direct)                   #contents of that directory
shuffle(ld)                                 #shuffle the image list for randomness
numEx           = len(ld)                   #number of images in the directory
outType         = "OCRfeatures"             #what the CNN is predicting
DUMP_WEIGHTS    = True                      #will we dump the weights of conv layers for visualization
trainTestSplit  = 0.90                      #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 6000                      #how many examples to evaluate per iteration

"""Define the folder where the model will be stored based on the input arguments"""
folder          = defineFolder(outType,size,lay1size,run)
print folder
trainDirect     = folder+"tempTrain/"
testDirect      = folder+"tempTest/"

#if update:     
#    stop = raw_input("Loading from folder "+folder+" : Hit enter to proceed or ctrl+C to cancel")
#else:
#    print "Initializing in folder "+folder





"""Load the train/test split information if update, else split and write out which images are in which dataset"""
if update:
    trainFs, testFs     = getTrainTestSplit(update,folder)
else:
    trainFs, testFs     = getTrainTestSplit(update,folder,numEx,trainTestSplit,ld)
trainL  = len(trainFs)
testL   = len(testFs)


print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

OCRfeatures,labels  = getOCRTargets() #get the ECFP vector for each CID
#testAverages(direct,OCRfeatures)   # determind the RMSE of guessing the mean
outsize             = len(OCRfeatures[OCRfeatures.keys()[0]]) #this it the size of the target (# of OCRfeatures)

"""If we are training a new model, define it"""   
print "loading model"
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

    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

"""If we are continuing to train an old model, load it"""
if update:
    with open(folder+"wholeModel.pickle",'rb') as f:
        model     = cPickle.load(f)



""" TRAINING """

    
#numIterations   = trainL/chunkSize + 1
superEpochs     = 10000
RMSE            = 1000000
oldRMSE         = 1000000
for sup in range(0,superEpochs):
    
    """Initialize empty matrices to hold our images and our target vectors"""
    numTrainEx          = min(len(listdir(folder+"tempTrain/")),chunkSize)
    trainImages     = np.zeros((numTrainEx,1,imdim,imdim),dtype=np.float)
    trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)

    
    
    
    oldRMSE     = min(oldRMSE,RMSE)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    

    count   = 0
    added   = 0
    traind  = listdir(trainDirect)
    while len(traind) < 2000:
	traind = listdir(trainDirect)
    testd   = listdir(testDirect)
    while len(testd) < 1:
	testd = listdir(testDirect)
    shuffle(testd)
    shuffle(traind)
    while added < numTrainEx:
        x   = traind[count]
        if x.find(".sdf") > -1:
            try:
                try:
                    CID     = x[:x.find(".sdf")]
                    image   = io.imread(trainDirect+x,as_grey=True)[10:-10,10:-10]         
                    image   = np.where(image > 0.1,1.0,0.0)
                    #plt.imshow(image)
                    #plt.savefig("../evaluation/"+CID)
                    trainImages[added,0,:,:]    = image
                    trainTargets[added]         = OCRfeatures[CID]
                    subprocess.call("rm "+trainDirect+x,shell=True)
                    added+=1
	            #print x
                except (IOError,ValueError) as e:
                    pass
            except (KeyError, ValueError) as e:
                subprocess.call("rm "+trainDirect+x,shell=True)  #This means this molecule was too big
        count+=1
        if count > len(traind)-1:
            count = 0
            traind = listdir(trainDirect)
	    while len(traind) == 0:
		traind = listdir(trainDirect)

    model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)


 
    numTestEx           = min(len(listdir(folder+"tempTest/")),testChunkSize)    
    testImages      = np.zeros((numTestEx,1,imdim,imdim),dtype=np.float)
    testTargets     = np.zeros((numTestEx,outsize),dtype=np.float)
    

    count   = 0
    added   = 0
    while added < numTestEx:
        x   = testd[count]
        if x.find(".sdf") > -1:
            try:
		try:
                    CID     = x[:x.find(".sdf")]
                    image   = io.imread(testDirect+x,as_grey=True)[10:-10,10:-10]         
                    image   = np.where(image > 0.1,1.0,0.0)
                    testImages[added,0,:,:]    = image
                    testTargets[added]         = OCRfeatures[CID]
                    subprocess.call("rm "+testDirect+x,shell=True)
                    added +=1
		    #print x
		except (IOError, ValueError) as e:
		    pass
            except (KeyError,ValueError):
                subprocess.call("rm "+testDirect+x,shell=True) #This means this molecule was too big
        count+=1
        if count > len(testd)-1:
            count = 0
	    testd = listdir(testDirect)
   	    while len(testd) == 0:
		testd = listdir(testDirect)
    preds   = model.predict(testImages)
    RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
    print "RMSE of epoch: ", RMSE

    
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



