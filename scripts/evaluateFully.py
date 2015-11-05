# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:19:41 2015

@author: frickjm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:23:07 2015

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


  
size    = 200
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"
direct2 = "../data/locations"+str(size)+"/"
ld      = listdir(direct)
numEx   = len(ld)

shuffle(ld)


testFs  = ld[int(numEx*0.8):]
testL   = len(testFs)

print "number of examples: ", numEx
print "test examples : ", testL


slideSize       = 32
numWins         = 10
batch_size      = 128
chunkSize       = 1024
testChunkSize   = 256
skipFactor      = 164
numTrainEx      = min(testL,chunkSize)*numWins
    
outsize         = (slideSize/2)**2

testImages      = np.zeros((testChunkSize,slideSize**2),dtype=np.float)
testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)


PRED_THRESHOLD  = 0.001
OP_THRESHOLD    = .99


with open("../fully/wholeModel2.pickle", 'rb') as f:
    model     = cPickle.load(f)

for x in testFs[0:chunkSize]:
    atomLocations     = []
    if x.find(".png") > -1:
        CID     = x[:x.find(".png")]
        image   = io.imread(direct+x,as_grey=True)         
        image   = np.where(image > 0.1,1.0,0.0)
        target  = io.imread(direct2+x,as_grey=True)
#        
#        plt.imshow(target)
#        plt.show()        
        
        
        while len(atomLocations) < 10:
            print np.sum(image)
            outpred = np.zeros((size/2,size/2),dtype=np.float)
            counts  = np.ones((size/2,size/2),dtype=np.float)*1
            As      = [int(np.floor(a)) for a in np.linspace(0,size-slideSize,skipFactor)]
            numTest = len(As)**2
            inp     = np.zeros((numTest,slideSize**2),dtype=np.float)
            
            count1 = 0
            for a in As:
                for b in As:
                    inp[count1]     = np.reshape(image[a:a+slideSize,b:b+slideSize],(slideSize**2,))
                    count1+=1
                    
            tmps        = model.predict(inp)  
            
    
            
            
            count2 = 0
            for a in As:
                a = a/2
                for b in As:
                    b = b/2
                    tmp         = tmps[count2]
                    #print a, b, tmp.shape  
                    tmp         = np.reshape(tmp,(slideSize/2,slideSize/2))
                    tmpmax      = np.max(tmp)*OP_THRESHOLD
                    #print tmp
                    tmp         = np.where(tmp>tmpmax,2*tmp,0.)
                    #print tmp              
                  
                    outpred[a:a+slideSize/2,b:b+slideSize/2]    = np.add(outpred[a:a+slideSize/2,b:b+slideSize/2], tmp)
                    counts[a:a+slideSize/2,b:b+slideSize/2]     = np.add(counts[a:a+slideSize/2,b:b+slideSize/2],np.ones((slideSize/2,slideSize/2)))
                    
                    count2+=1
            
            counts     = resize(counts,(size,size))
            outpred    = resize(outpred,(size,size))        
            image2     = np.divide(outpred,counts)
            
            
            maxX     = 0
            maxY     = 0
            mVal     = 0
            Bs      = [int(np.floor(a)) for a in np.linspace(7,size-3,190)]
            stop    = ''

            for yval in Bs:
                for xval in Bs:
                    littleWindow = image2[yval:yval+6,xval:xval+6]
                    if np.sum(littleWindow) > mVal:
                        maxX     = xval
                        maxY     = yval
                        mVal     = np.sum(littleWindow) 
                        #print mVal
            print maxY, maxX
            
            mWind     = image[maxY-slideSize/2:maxY+slideSize/2,maxX-slideSize/2:maxX+slideSize/2]
            topred    = np.zeros((1,slideSize**2),dtype=np.float)
            mWind2    = np.reshape(mWind,(slideSize**2,))
            topred[0] = mWind2
            predawg   = model.predict(topred)

            maxPredY  = np.argmax(predawg[0])/(slideSize/2)
            maxPredX  = np.argmax(predawg[0])%(slideSize/2)
                    
            print maxPredY, maxPredX

            imy, imx     = (maxY+2*maxPredY-slideSize/2 , maxX+2*maxPredX-slideSize/2)
            print imy, imx
            atomLocations.append( (imy, imx) )

            plt.figure(4)
            plt.imshow(mWind)
#            image4     = image[:,:]
            image[imy-5:imy+5,imx-5:imx+5] = np.zeros(image[imy-5:imy+5,imx-5:imx+5].shape)

            plt.figure(5)
            plt.imshow(np.reshape(predawg,(slideSize/2,slideSize/2)))

            print atomLocations

            
            m          = np.max(image2)
            threshold  = m*PRED_THRESHOLD
            image3     = np.where(image2>threshold,image2,0.)
        plt.figure(0)    
        plt.gcf().suptitle("Original Image")                    
        plt.imshow(image)
        plt.figure(1)
        plt.gcf().suptitle("Probability of Being an Atom")        
        plt.imshow(image2)
#                plt.figure(2)
#                plt.gcf().suptitle("Thresholded Probability of Being an Atom")        
#                plt.imshow(image3)
        plt.show()

            # Pick a window
