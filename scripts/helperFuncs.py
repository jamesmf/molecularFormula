# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:46:08 2015

@author: frickjm
"""

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



"""get the ECFP vectors for training"""
def getECFPTargets():
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
    
    
"""get the solubility for training"""
def getSolubilityTargets():
    out     = {}
    with open("../data/sols.pickle",'rb') as f:
        d =  cPickle.load(f)
    for k,v in d.iteritems():
        out[k] = [float(v)]
    return out
    
    
    
    
    
    

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
        
        
        
def getTrainTestSplit(update,numEx,trainTestSplit):
    if not update:
        trainFs = ld[:int(numEx*trainTestSplit)]
        testFs  = ld[int(numEx*trainTestSplit):]
        with open(folder+"traindata.csv",'wb') as f:
            f.write('\n'.join(trainFs))
        with open(folder+"testdata.csv",'wb') as f:        
            f.write('\n'.join(testFs))
    else:
        with open(folder+"traindata.csv",'rb') as f:
            trainFs = f.read().split("\n")
        with open(folder+"testdata.csv",'rb') as f:        
            testFs  = f.read().split("\n")
            
    return trainFs, testFs
    
    
def handleArgs(arglist):
    if len(arglist) <= 1:
        print "needs 'update' or 'new' as first argument"
        sys.exit(1)

    if arglist[1].lower().strip() == "update":
        update     = True    
        if len(arglist) < 5:
            print "needs image size, layer size, run # as other inputs"
            sys.exit(1)
        else:
            size = int(arglist[2])     #size of the images
            lay1size = int(arglist[3]) #size of the first receptive field
            run     = "_"+str(arglist[4].strip())
            print size, lay1size
    else:
        update     = False
        size    = 200                               #size of the images
        lay1size= 5                                #size of the first receptive field
        run     = ""
        
    return update, size, lay1size, run



def defineFolder(outType,size,lay1size,run):
    folder  = outType+'/'+str(size)+"_"+str(lay1size)+run+"/"
    if not isdir(folder):
        mkdir(folder)
        
    if (run == "") and (isdir(folder)):
        i=1
        oldfolder = folder
        while isdir(folder):
            i+=1
            folder  = oldfolder[:-1]+"_"+str(i)+'/'
            print folder
        mkdir(folder)
    return folder
            