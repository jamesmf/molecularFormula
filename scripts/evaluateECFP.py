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
from tabulate import tabulate
import cPickle
import sys

from sklearn.metrics import mean_squared_error

#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, Adadelta, Adagrad

from scipy.spatial import distance
from scipy.stats import entropy

def getRank(cid,vec,ecfps):

    tosort  = []    
    for k,vec2 in ecfps.iteritems():
        #euc     = distance.euclidean(vec,vec2)
        cos     = distance.cosine(vec, vec2)
        #KL1     = entropy(vec,vec2)
        #KL2     = entropy(vec2,vec)
        
        #row     = [k, euc, cos, KL1, KL2]        
        row     = [k, cos]        
        tosort.append(row)
      
        
    data    = np.array(tosort)    
    sCos    = data[np.argsort(data[:,1])]
    #sEuc    = data[np.argsort(data[:,1])]
    #sCos    = data[np.argsort(data[:,2])]
    #sKL1    = data[np.argsort(data[:,3])]
    #sKL2    = data[np.argsort(data[:,4])]  

    #e1  = list(sEuc[:,0]).index(cid)      
    c   = list(sCos[:,0]).index(cid)
    #k1  = list(sKL1[:,0]).index(cid)
    #k2  = list(sKL2[:,0]).index(cid)
    
    return c
    #return [e1, c, k1, k2]
    
    
def printFormula(p,t,atomlist,cid):
    print '\t',cid
    headers     = ["ATOM","ACTUAL","PREDICTED","FLOAT"]
    tab         = []
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind]])
            #print atomlist[ind],'\t\t',int(t[ind]),'\t\t',int(np.round(p[ind])),'\t\t', p[ind],'\t',ind
        elif np.round(p[ind]) > 0:
            #print atomlist[ind],'\t\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind],'\t',ind
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind]])
    print tabulate(tab,headers=headers)

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


def main():
    """Define parameters of the run"""
    size    = 200                               #size of the images
    imdim   = size - 20                         #strip 10 pixels buffer from each size
    direct  = "../data/images"+str(size)+"/"    #directory containing the images
    ld      = listdir(direct)                   #contents of that directory
    numEx   = len(ld)
    
    
    DUMP_WEIGHTS = True  # will we dump the weights of conv layers for visualization
    
    shuffle(ld)
    
    trainTestSplit     = 0.80
    
    with open(sys.argv[1]+"testdata.csv",'rb') as f:
        testFs  = [pn for pn in f.read().split("\n") if pn != '']
    testL   = len(testFs)
    
    print "number of examples: ", numEx
    print "test examples : ", testL
    
    batch_size      = 32
    chunkSize       = 2048
    testChunkSize   = testL
    numTestEx      = min(testL,testChunkSize)
    
    
    ecfps           = getECFPvecs()
    atomlist        = getAtomList()
        
    outsize         = len(ecfps[ecfps.keys()[0]])
    
    testImages      = np.zeros((testChunkSize,1,imdim,imdim),dtype=np.float)
    testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)
    
    
    with open(sys.argv[1]+"wholeModel.pickle",'rb') as f:
        model     = cPickle.load(f)
    
    
    numIterations = 1
    for i in range(0,numIterations):
        shuffle(testFs)
        count    = 0
        cids     = []
        while count < testChunkSize:    
            for x in testFs:        
                if x.find(".png") > -1:
                    CID     = x[:x.find(".png")]
                    cids.append(CID)
                    image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
                    #image   = np.where(image > 0.1,1.0,0.0)
                    testImages[count,0,:,:]    = image
                    testTargets[count]         = ecfps[CID]
                    count +=1
        
        preds   = model.predict(testImages)
        RMSE    = np.sqrt(mean_squared_error(testTargets, preds)) 
        ranks   = []        
        print RMSE
        if RMSE < 300:
            for ind1 in range(0,len(preds)):
                p   = [preds[ind1][ind2] for ind2 in range(0,len(preds[0]))]
                t   = [int(testTargets[ind1][ind2]) for ind2 in range(0,len(testTargets[0]))]
                #printFormula(p,t,atomlist,cids[ind1])
                ranks.append(getRank(cids[ind1],p,ecfps))
                print ranks[ind1]
    print np.mean(ranks)
    with open("~/CHECKME.txt",'wb') as f:
        f.write('\n'.join(ranks))
    

    
if __name__ == "__main__":
    main()