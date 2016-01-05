# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:00:21 2015

@author: test
"""

#Evaluating the uniqueness of the feature set
from helperFuncs import getOCRTargets
import numpy as np
import cPickle
import matplotlib.pyplot as plt

targets,labels         = getOCRTargets()

feature2CID     = {}

for k,v in targets.iteritems():
    v   = tuple(v)
    if v in feature2CID:
        feature2CID[v] +=1
    else:
        feature2CID[v]  =1
        
x   = feature2CID.values()
print np.mean(x)

hist, bins  = np.histogram(x,bins= 50000)
width       = 0.7 * (bins[1]-bins[0])
center      = (bins[:-1]+bins[1:])/2
plt.bar(center, hist, align='center',width=width)
plt.savefig("../uniquenessHist",format="jpg")