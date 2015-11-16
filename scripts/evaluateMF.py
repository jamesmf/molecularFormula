# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:47:43 2015

@author: frickjm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:22:30 2015

@author: frickjm
"""


import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize 
import numpy as np

from os import listdir
from random import shuffle
import cPickle

from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad


def printFormula(p,t,atomlist,cid):
    print '\t',cid
    print 'ATOM\tACTUAL\tPREDICTED\tFLOAT PREDICTION'
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            print atomlist[ind],'\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind]
        elif np.round(p[ind]) > 0:
            print atomlist[ind],'\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind]

def getTargetMeans(mfs):
    x   = np.mean(mfs.values(),axis=0)
    y   = np.std(mfs.values(),axis=0)
    #print "means", x
    #print "stds", y
#    np.savetxt("../targetMeans.txt",x,delimiter=',')
#    stop=raw_input("")
    return x,y
        


def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                print "visualizing layer ", layercount
                
                with open("../layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


def testWAverages(direct,mfs,means):
    ld  = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,16),dtype=np.float)
    y       = np.zeros((num,16),dtype=np.float)
    count   = 0
    for x in ld[:num]:
        CID     = x[:x.find(".png")]
        y[count,:]  = mfs[CID]
        preds[count,:] = means
        count+=1
   
    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


atomlist= ['C', 'B', 'F', 'I', 'H', 'K', 'O', 'N', 'P', 'Si', 'Se', 'Cl', 'S', 'As', 'Br', 'Na']
size    = 200
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"
ld      = listdir(direct)
numEx   = len(ld)


DUMP_WEIGHTS = True

shuffle(ld)

testFs  = ld[int(numEx*0.8):]

testL   = len(testFs)

print "number of examples: ", numEx
print "test examples : ", testL

batch_size      = 32
chunkSize       = 2048
testChunkSize   = 256
numTrainEx      = min(testL,chunkSize)

with open("../cidsMF.pickle",'rb') as f:
    mfs    = cPickle.load(f)
    
outsize         = len(mfs[mfs.keys()[0]])

testImages      = np.zeros((numTrainEx/10,1,imdim,imdim),dtype=np.float)
testTargets     = np.zeros((numTrainEx/10,outsize),dtype=np.float)

targetMeans,stds= getTargetMeans(mfs)

with open("../molecularFormula/wholeModel.pickle", 'rb') as f:
    model     = cPickle.load(f)


        
shuffle(testFs)
count    = 0
cids     = []
for x in testFs[:200]:
    if x.find(".png") > -1:
        CID     = x[:x.find(".png")]
        cids.append(CID)
        image   = io.imread(direct+x,as_grey=True)[10:-10,10:-10]         
        image   = np.where(image > 0.1,1.0,0.0)
        testImages[count,0,:,:]    = image
        testTargets[count]         = np.divide(np.subtract(mfs[CID],targetMeans),stds)
        count +=1

preds   = model.predict(testImages)
RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
print RMSE
if RMSE < 3:
    for ind1 in range(0,len(preds)):
        if ind1 < 10 :
            
            p   = [x for x in preds[ind1]]
            p   = [(p[ind2]*stds[ind2])+targetMeans[ind2] for ind2 in range(0,len(targetMeans))]
            t   = testTargets[ind1]
            t   = [int((t[ind2]*stds[ind2])+targetMeans[ind2]) for ind2 in range(0,len(targetMeans))]
            printFormula(p,t,atomlist,cids[ind1])
            plt.imshow(testImages[ind1,0,:,:])
            plt.show()


