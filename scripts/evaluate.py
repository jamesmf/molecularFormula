# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:41:01 2016

@author: test
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


import helperFuncs 

#import matplotlib.pyplot as plt
import skimage.io as io
import scipy.misc as mi
#from skimage.transform import resize 
import numpy as np
from tabulate import tabulate
from os import listdir
#from os.path import isdir
#from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import sys
import subprocess
import time
from scipy.spatial import distance
#from scipy.stats import entropy
from sklearn.metrics import mean_squared_error


np.random.seed(0)

def printFormula(p,t,cid,atomlist,means):
    print '\t',cid
    headers     = ["FEATURE","ACTUAL","PREDICTED","FLOAT","MEAN"]
    tab         = []
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
            #print atomlist[ind],'\t\t',int(t[ind]),'\t\t',int(np.round(p[ind])),'\t\t', p[ind],'\t',ind
        elif np.round(p[ind]) > 0:
            #print atomlist[ind],'\t\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind],'\t',ind
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
    print tabulate(tab,headers=headers)

def printFormula(p,t,cid,atomlist,means):
    print '\t',cid
    headers     = ["FEATURE","ACTUAL","PREDICTED","FLOAT","MEAN"]
    tab         = []
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
            
        elif np.round(p[ind]) > 0:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
        else:
            printIt = False
            
        if printIt:
            name    = atomlist[ind]
            real    = int(t[ind])
            
            tab.append()
    print tabulate(tab,headers=headers)


def printFormula2(p,atomlist):
    headers     = ["FEATURE","PREDICTED","FLOAT"]
    tab         = []
    for ind in range(0,len(atomlist)):
        tab.append([atomlist[ind],int(np.round(p[ind])),p[ind]])
    print tabulate(tab,headers=headers)




def getRank(cid,vec,allVec):

    tosort  = []    
    for k,vec2 in allVec.iteritems():
        cos     = distance.cosine(vec, vec2)
        row     = [k, cos]        
        tosort.append(row)
           
    data    = np.array(tosort)    
    sCos    = data[np.argsort(data[:,1])]   
    c   = list(sCos[:,0]).index(cid)
    return c, sCos[:10]


"""Require an argument specifying whether this is an update or a new model, parse input"""
size, run, outType     = helperFuncs.handleArgs(sys.argv)


"""Define parameters of the run"""
batch_size      = 32                        #how many training examples per batch


"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(True,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
trainNP         = folder+"tempTrainNP/"
testDirect      = folder+"tempTest/"
testNP          = folder+"tempTestNP/"

"""Load the train/test split information"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(True,folder)

trainL  = len(trainFs)
testL   = len(testFs)


features,labels     = helperFuncs.getTargets(outType) #get the OCR vector for each CID
outsize             = len(features[features.keys()[0]]) #this it the size of the target (# of OCRfeatures)
means,stds          = helperFuncs.getMeansStds()


"""load model"""
#with open(folder+"bestModel.pickle",'rb') as f:
#    model     = cPickle.load(f)

model   = helperFuncs.loadModel(folder+"wholeModel")


while not isfile(testNP+"Xtest.h5"):
    print "sleeping because Test folder empty             \r",
    time.sleep(1.)
print ""

print "Loading np test arrays" 

loadedUp    = False
while not loadedUp:       
    try:
#        with open(testNP+"Xtest.pickle",'rb') as f:
#            testImages     = cPickle.load(f)
#    
#        with open(testNP+"ytest.pickle",'rb') as f:
#            testTargets    = cPickle.load(f)
        testImages  = helperFuncs.loadData(testNP+"Xtest",'h5')
        testTargets = helperFuncs.loadData(testNP+"ytest",'h5')            
            
        with open(testNP+"testCIDs.pickle",'rb') as f:
            testCIDs       = cPickle.load(f)

        loadedUp = True
    except Exception as e:
        err     = e
        print err, "                              \r",
        time.sleep(2)
        
print ""

preds   = model.predict(testImages)
RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
print "RMSE of epoch: ", RMSE

for i in range(0,len(preds)):
    printFormula(preds[i],testTargets[i],testCIDs[i],labels,means)
    mi.imsave("../evaluation/"+testCIDs[i]+".jpg",testImages[i][0])
    
    rank,mostSim     = getRank(testCIDs[i],preds[i],features)
    print "correct rank: ", rank
    print "most similar: ", mostSim
    
    
    stop=raw_input("")


del testImages, testTargets    
