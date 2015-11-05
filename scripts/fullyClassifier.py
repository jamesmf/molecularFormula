# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:57:05 2015

@author: frickjm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:00:09 2015


Train a fully connected neural network on small image regions to predict
the locations of atoms within an image of a molecule using a sliding classifier

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


   


def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                print "visualizing layer ", layercount
                
                with open("../fullyClassifier/layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


DUMP_WEIGHTS = True
size    = 200
imdim   = size - 20                         #strip 10 pixels buffer from each size
direct  = "../data/images"+str(size)+"/"
direct2 = "../data/locations"+str(size)+"/"
ld      = listdir(direct)
numEx   = len(ld)

shuffle(ld)

trainFs = ld[:int(numEx*0.8)]
testFs  = ld[int(numEx*0.8):]
trainL  = len(trainFs)
testL   = len(testFs)

print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL


slideSize       = 32
numWins         = 10
batch_size      = 16
chunkSize       = 4096
testChunkSize   = 512
numTrainEx      = min(trainL,chunkSize)*numWins*2

classWeights    = {1:10,0:1}
    
outsize         = (slideSize/2)**2

trainImages     = np.zeros((numTrainEx,1,slideSize,slideSize),dtype=np.float)
trainTargets    = np.zeros((numTrainEx,1),dtype=np.float)
#testImages      = np.zeros((testChunkSize,1,slideSize,slideSize),dtype=np.float)
#testTargets     = np.zeros((testChunkSize,outsize),dtype=np.float)


model = Sequential()
model.add(Dense(slideSize**2,1024))
model.add(Activation('relu'))

model.add(Dense(1024,512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512, 1))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='squared_hinge', optimizer='rmsprop')
#model.compile(loss='categorical_crossentropy', optimizer=sgd



""" TRAINING """

upperbound = size-slideSize
lowerbound = (slideSize/2)+1
r     = range(lowerbound,upperbound)
    
numIterations   = trainL/chunkSize + 1
superEpochs     = 100
for sup in range(0,superEpochs):
    if sup > 0:
        with open("../fullyClassifier/wholeModel.pickle", 'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(model)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    
    for i in range(0,numIterations):
        trainL          = len(trainFs[i*chunkSize:(i+1)*chunkSize])
        numTrainEx      = min(trainL,chunkSize)*numWins*2
        trainImages     = np.zeros((numTrainEx,slideSize**2),dtype=np.float)
        trainTargets    = np.zeros((numTrainEx,1),dtype=np.float)
       
    
        print "iteration ",i,": ", i*chunkSize," through ", (i+1)*chunkSize
        count   = 0
        for name in trainFs[i*chunkSize:(i+1)*chunkSize]:
            if name.find(".png") > -1:
                CID     = name[:name.find(".png")]
                image   = io.imread(direct+name,as_grey=True)         
                image   = np.where(image > 0.1,1.0,0.0)
                target  = io.imread(direct2+name,as_grey=True)
                target  = np.where(target>0.1,1.,0.)

                # Pick a window
                inds     = range(0,size-slideSize)
                points     = [(y,x) for y in inds for x in inds if target[y,x] == 1]
                shuffle(points)
                

                added         = 0
                pointInd      = 0
                while added < numWins:
                    lily, lilx     = points[pointInd]
                    lilWindow = image[lily-slideSize/2:lily+slideSize/2,lilx-slideSize/2:lilx+slideSize/2]
                    trainImages[count] = np.reshape(lilWindow,(slideSize**2,))
                    trainTargets[count] = 1
                    count+=1
                    added+=1
                while added < 2*numWins:
                    shuffle(r)
                    lily     = r[0]
                    shuffle(r)
                    lilx     = r[0]
                    if (lily, lilx) not in points:
                        lilWindow = image[lily-slideSize/2:lily+slideSize/2,lilx-slideSize/2:lilx+slideSize/2]
#                        plt.imshow(lilWindow)
#                        plt.show()
                        if (np.sum(lilWindow) > 0) or (np.random.rand() > 0.9):
                            trainImages[count] = np.reshape(lilWindow,(slideSize**2,))
                            trainTargets[count] = 0
                            count+=1
                            added+=1                        

        print np.mean(trainTargets)
        model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)

        
        if DUMP_WEIGHTS:
            dumpWeights(model)
        


del trainTargets, trainImages
""" END TRAINING """

