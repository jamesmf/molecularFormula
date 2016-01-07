# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:35:52 2016

@author: test
"""
from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import sys
import time
import numpy as np
import subprocess
import cPickle
import helperFuncs

"""Processes images and pickles numpy arrays of them for training/testing"""

size, indir, binarize, blur, padding, targetType = helperFuncs.dataProcessorArgs(sys.argv[1:])

targets, labels     = helperFuncs.getTargets(targetType)
outsize             = len(targets[targets.keys()[0]])

while not isdir(indir+"tempTrain/"):
    time.sleep(10.)
    print "I'm sleeping", isdir(indir), indir, "                     \r",

if not isdir(indir+"tempTrainNP/"):
    mkdir(indir+"tempTrainNP/")
    mkdir(indir+"tempTestNP/")
    
#print "reading Train/Test files"   
#train   = [x for x in file(indir+"traindata.csv").read().split("\n") if x != '']    
#test    = [x for x in file(indir+"testdata.csv").read().split("\n") if x != '']    

trainFolder     = indir+"tempTrain/"
trainNPfolder   = indir+"tempTrainNP/"
testFolder      = indir+"tempTest/"
testNPfolder    = indir+"tempTestNP/"


while True:

    if isfile(trainNPfolder+ "Xtrain.pickle"):
        time.sleep(1)
        print "sleeping because Train folder full      \r",
    else:
        ld  = listdir(trainFolder)
        shuffle(ld)
        numTrainEx      = len(listdir(indir+"tempTrain/"))
        trainImages     = np.zeros((numTrainEx,1,size,size),dtype=np.float)
        trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)

        added   = 0
        count   = 0
        while added < numTrainEx:
            x   = ld[count]
            print x, added
            if x.find(".sdf") > -1:
                try:
                    try:
                        CID     = x[:x.find(".sdf")]
                        
                        image   = helperFuncs.processImage(CID,trainFolder,binarize,blur,padding,size,noise=True)                        
                        subprocess.call("rm "+trainFolder+x,shell=True)                        
                        trainImages[added,0,:,:]    = image
                        trainTargets[added]         = targets[CID]                        
                        added+=1

                    except (IOError,ValueError) as e:
                        print e
                except (KeyError, ValueError) as e:
                    subprocess.call("rm "+trainFolder+x,shell=True) #This means this molecule was too big
            count+=1
            if count > len(ld)-1:
                count = 0
                ld = listdir(trainFolder)
            while len(ld) == 0:
                ld = listdir(trainFolder)



        
        with open(trainNPfolder+ "Xtrain.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(trainImages)
        with open(trainNPfolder+ "ytrain.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(trainTargets)  


    if isfile(testNPfolder+ "Xtest.pickle"):
        time.sleep(1)
        print "sleeping because test folder full      \r",
    else:
        ld  = listdir(testFolder)
        shuffle(ld)
        numTestEx      = len(listdir(indir+"tempTest/"))
        testImages     = np.zeros((numTestEx,1,size,size),dtype=np.float)
        testTargets    = np.zeros((numTestEx,outsize),dtype=np.float)

        added   = 0
        count   = 0
        while added < numTestEx:
            x   = ld[count]
            if x.find(".sdf") > -1:
                try:
                    try:
                        CID     = x[:x.find(".sdf")]
                        
                        image   = helperFuncs.processImage(CID,testFolder,binarize,blur,padding,size,noise=True)                        
                        subprocess.call("rm "+testFolder+x,shell=True)                        
                        testImages[added,0,:,:]    = image
                        testTargets[added]         = targets[CID]                        
                        added+=1

                    except (IOError,ValueError) as e:
                        pass
                except (KeyError, ValueError) as e:
                    subprocess.call("rm "+testFolder+x,shell=True) #This means this molecule was too big
            count+=1
            if count > len(ld)-1:
                count = 0
                ld = listdir(testFolder)
            while len(ld) == 0:
                ld = listdir(testFolder)
        


        with open(testNPfolder+ "Xtest.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(testImages)
        with open(testNPfolder+ "ytest.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(testTargets)            
        

