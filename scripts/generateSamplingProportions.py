# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:27:09 2015

@author: test
"""

import cPickle
import numpy as np
from scipy.spatial.distance import cosine


with open("../data/cidsFeatureVectors.pickle",'rb') as f:
    vectors  = cPickle.load(f)

binary  = {}
for k,v in vectors.iteritems():
    binary[k]   = [1*(value > 0.1) for value in v]
#    print v
#    print binary[k]
#    stop=raw_input("")

labels  = file("../data/cidsFeatureKeys.txt").read().split(',')
meanfreq= np.mean(binary.values(),axis=0)
freqs   = np.sum(binary.values(),axis=0)

for i in range(0,len(freqs)):
    if freqs[i] == 1:
        print labels[i], freqs[i]


means   = np.mean(vectors.values(),axis=0)
stds    = np.std(vectors.values(),axis=0)
sums    = np.sum(vectors.values(),axis=0)

#for i in range(0,len(means)):
#    print labels[i], means[i], sums[i]
    
rarities = {}   
for k,v in binary.iteritems():
    #rarity  = 3 + cosine(v,means)
    rarity  = np.sum([1/means[i] for i in range(0,len(v)) if ((v[i]-means[i] ) >.9)])
    rarity  = rarity + 100  
    rarities[k] = rarity
#    print rarity
#    for i in range(0,len(means)):
#        if v[i] > 0.1:
#            print labels[i], means[i], v[i]
#        
#    stop=raw_input("")

sampfreq= {}
tot     = np.sum(rarities.values())  
baseline = 100/tot  
for k,v in vectors.iteritems():
    rarity  = rarities[k]/tot
    fold    = rarity/baseline
    sampfreq[k] = rarity
#    print rarity, fold
#    for i in range(0,len(means)):
#        if v[i] > 0.1:
#            print labels[i], means[i], v[i], (binary[k][i]-means[i])


with open("../data/cidsSampleFreq.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(sampfreq)
        
print len(labels)
print len([1 for i in range(0,len(freqs)) if (freqs[i] ==1) ])