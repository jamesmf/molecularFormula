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



def printFormula(p,t,atomlist,cid):
    print '\t',cid
    print 'ATOM\t\tACTUAL\tPREDICTED\tFLOAT PREDICTION\tHEADERIND'
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            print atomlist[ind],'\t\t',int(t[ind]),'\t\t',int(np.round(p[ind])),'\t\t', p[ind],'\t',ind
        elif np.round(p[ind]) > 0:
            print atomlist[ind],'\t\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind],'\t',ind


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


def getAtomList():
    b     = {}
    with open("../cidsECFP.txt",'rb') as f:
        a     =  f.readline().split(",")
        for x in a:
            sp     = x.split("=")
            name   = sp[0].replace("}",'').replace("{",'').strip()
            ind    = int(sp[1].replace("}",'').replace("{",'').strip())
            b[ind] = name
    return b

def testWAverages(direct,ecfps):
    means   = np.mean(ecfps.values(),axis=0) 
    s       = len(means)
    ld      = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,s),dtype=np.float)
    y       = np.zeros((num,s),dtype=np.float)
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

testFs  = ld[int(numEx*trainTestSplit):]
testL   = len(testFs)

print "number of examples: ", numEx
print "test examples : ", testL

batch_size      = 32
chunkSize       = 2048
testChunkSize   = 1000
numTestEx      = min(testL,testChunkSize)


ecfps           = getECFPvecs()
testWAverages(direct,ecfps)
atomlist        = getAtomList()
    
outsize         = len(ecfps[ecfps.keys()[0]])

testImages      = np.zeros((testChunkSize,1,imdim,imdim),dtype=np.float)
testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)


with open(sys.argv[1]+"wholeModel.pickle",'rb') as f:
    model     = cPickle.load(f)


numIterations = 5
for i in range(0,numIterations):
    shuffle(testFs)
    count    = 0
    cids     = []
    for x in testFs:
        while count < testChunkSize:
            if x.find(".png") > -1:
                CID     = x[:x.find(".png")]
                cids.append(CID)
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
                printFormula(p,t,atomlist,cids[ind1])


