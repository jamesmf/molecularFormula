# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:24:53 2015

@author: frickjm
"""

"""Evaluate Solubility"""

from helperFuncs import *

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

"""Define parameters of the run"""
size            = 200                               #size of the images
imdim           = size - 20                         #strip 10 pixels buffer from each size
direct          = "../data/images"+str(size)+"/"    #directory containing the images
ld              = listdir(direct)                   #contents of that directory
numEx           = len(ld)
batch_size      = 32
testChunkSize   = 1000


sols            = getSolubilityTargets()
outsize         = len(sols[sols.keys()[0]])

testImages      = np.zeros((testChunkSize,1,imdim,imdim),dtype=np.float)
testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)


folder          = sys.argv[1]

with open(folder+"testdata.csv",'rb') as f:        
    testFs  = f.read().split("\n")
    
with open(sys.argv[1]+"wholeModel.pickle",'rb') as f:
    model     = cPickle.load(f)


shuffle(testFs)
count    = 0
cids     = []
print testFs
while count < testChunkSize:
    x     = testFs[count]       
    if x.find(".png") > -1:
        CID     = x[:x.find(".png")]
        cids.append(CID)
        image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
        image   = np.where(image > 0.1,1.0,0.0)
        testImages[count,0,:,:]    = image
        testTargets[count]         = sols[CID]
        count +=1
        print count

preds   = model.predict(testImages)
RMSE    = np.sqrt(mean_squared_error(testTargets, preds))   

print cids
print preds
print testTargets      
print RMSE
TP,FP,TN,FN     = [0,0,0,0]
print preds.shape
print testTargets.shape
if RMSE < 300:
    for ind1 in range(0,len(preds)):
        if ind1 < 100:

            sol          = testTargets[ind1][0] > 10
            notSol      = not sol
            predSol     = preds[ind1][0] > 10
            predNotSol   = not predSol
            
            if (sol and predSol):
                TP+=1
                category = "TP"
            elif (notSol and predSol):
                FP+=1
                category = "FP"
            elif (notSol and predNotSol):
                TN+=1
                category = "TN"
            elif (sol and predNotSol):
                FN+=1
                category = "FN"

            print testTargets[ind1], preds[ind1], category        
            
print "TP", TP
print "FP", FP
print "TN", TN
print "FN", FN

sensitivity     = TP*1. / (TP + FN)
specificity     = TN*1. / (TN + FP)
print sensitivity, specificity
GMean           = np.sqrt(sensitivity*specificity)
print GMean
