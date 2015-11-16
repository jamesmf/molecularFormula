# -*- coding: utf-8 -*-
"""
Created on Nov 12 15:22:30 2015

@author: frickjm
"""


import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize 
import numpy as np

from os import listdir
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

"""get the ECFP vectors for training"""
def getECFPvecs():
    ecfps = {}
    if isfile("../cidsECFP.pickle"):
        with open("../cidsECFP.pickle",'rb') as f:
            return cPickle.load(f)
    else:
        
        with open("../cidsECFP.txt",'rb') as f:
            f.readline() #ignore the header line
            for x in f:
                sp     = x.split("\t") 
                #ignore blank lines
                if len(sp) > 1:
                    CID         = sp[0]
                    vec         = np.array([int(x) for x in sp[2][1:-2].split(',')],dtype=np.float)
                    ecfps[CID]  = vec
        
        with open("../cidsECFP.pickle",'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(ecfps)
    return ecfps

def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                with open("../ecfp/layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


def testWAverages(direct,ecfps,means):
    ld  = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,16),dtype=np.float)
    y       = np.zeros((num,16),dtype=np.float)
    count   = 0
    for x in ld[:num]:
        CID     = x[:x.find(".png")]
        y[count,:]  = ecfps[CID]
        preds[count,:] = means
        count+=1
   
    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


"""Define parameters of the run"""
size    = 200                               #size of the images
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"    #directory containing the images
ld      = listdir(direct)                   #contents of that directory
numEx   = len(ld)


DUMP_WEIGHTS = True  # will we dump the weights of conv layers for visualization

shuffle(ld)

trainTestSplit     = 0.80

trainFs = ld[:int(numEx*trainTestSplit)]
testFs  = ld[int(numEx*trainTestSplit):]
trainL  = len(trainFs)
testL   = len(testFs)

print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

batch_size      = 32
chunkSize       = 20048
testChunkSize   = 1024
numTrainEx      = min(trainL,chunkSize)



ecfps           = getECFPvecs()
    
outsize         = len(ecfps[ecfps.keys()[0]])

trainImages     = np.zeros((numTrainEx,1,imdim,imdim),dtype=np.float)
trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
testImages      = np.zeros((testChunkSize,1,imdim,imdim),dtype=np.float)
testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)



if len(sys.argv) <= 1:
    print "needs 'update' or 'new' as first argument"
    sys.exit(1)
    
if sys.argv[1].lower().strip() == "new":
    model = Sequential()
    
    model.add(Convolution2D(32, 1, 10, 10, border_mode='full')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(32, 32, 5, 5))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(64, 32, 5, 5)) 
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

elif sys.argv[1].lower().strip() == "update":
    with open("../ecfp/wholeModel.pickle",'rb') as f:
        model     = cPickle.load(f)
        
else:
    print "needs 'update' or 'new' as first argument"
    sys.exit(1)



""" TRAINING """

    
numIterations   = trainL/chunkSize + 1
superEpochs     = 10
for sup in range(0,superEpochs):

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
                image   = np.where(image > 0.1,1.0,0.0)
                trainImages[count,0,:,:]    = image
                trainTargets[count]         = ecfps[CID]
                count +=1
    
        model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)
        
        
        shuffle(testFs)
        count   = 0
        for x in testFs[:testChunkSize]:
            if x.find(".png") > -1:
                CID     = x[:x.find(".png")]
                image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
                image   = np.where(image > 0.1,1.0,0.0)
                testImages[count,0,:,:]    = image
                testTargets[count]         = ecfps[CID]
                count +=1
        
        preds   = model.predict(testImages)
        RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
        print RMSE
        if RMSE < 300:
            for ind1 in range(0,len(preds)):
                if ind1 < 2:
                    p   = [preds[ind1][ind2] for ind2 in range(0,len(preds[0]))]
                    t   = [int(testTargets[ind1][ind2]) for ind2 in range(0,len(testTargets[0]))]
                    print p, t
        
        if DUMP_WEIGHTS:
            dumpWeights(model)

        with open("../ecfp/wholeModel.pickle", 'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(model)

del trainTargets, trainImages
""" END TRAINING """



