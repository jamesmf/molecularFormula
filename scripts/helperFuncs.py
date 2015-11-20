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
    
    