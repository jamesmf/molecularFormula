# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:17:45 2015

@author: test
"""
import cPickle
import numpy as np
from operator import itemgetter

MAX_SSSR_FEATURE_SIZE = 7   #Only include features for SSSRs of size < X
MAX_ATOM_COUNT_STDDEV = 2   #Only include molecules w/ (atom count) < (mean) + MAX_ATOM*(std_dev)
FEATURE_FREQ_CUTOFF   = 100 #Only include features that occur more than N times in the data

#load the dictionary pickled by 'fromSDFs.py'
with open("../data/cidsFEATURES.pickle",'rb') as f:  
    features    = cPickle.load(f)
  
atomCounts  = []    #size of the molecule in atoms
SSSRnums    = []  
pairskeys   = set()
bondskeys   = set()
atomskeys   = set()
SSSRkeys    = set()
#get the set of values each feature can take on    
for k,v in features.iteritems():
    pairs   = v[0]
    bonds   = v[1]
    atoms   = v[2]
    SSSR    = v[3]
    SSSRnum = v[4]

    
    pairskeys   = set(pairskeys)|set(pairs.keys())
    bondskeys   = set(bondskeys)|set(bonds.keys())
    atomskeys   = set(atomskeys)|set(atoms.keys())
    SSSRkeys    = set(SSSRkeys)|set(SSSR.keys())
    atomCounts.append(np.sum(atoms.values()))
    
print "Mean of atomcount", np.mean(atomCounts)
print "STD of atomcount", np.std(atomCounts)
print len(atomCounts)

    
pairskeys   = list(pairskeys)
pairsize    = len(pairskeys)

bondskeys   = list(bondskeys)
bondsize    = len(bondskeys)

atomskeys   = list(atomskeys)
atomsize    = len(atomskeys)

SSSRkeys    = list(SSSRkeys)   
SSSRkeys    = [k for k in SSSRkeys if int(k.replace('SSSR_','')) <= MAX_SSSR_FEATURE_SIZE] 
SSSRsize    = len(SSSRkeys)

pairvec     = np.zeros((1,pairsize))
atomvec     = np.zeros((1,atomsize))
bondvec     = np.zeros((1,bondsize))
SSSRvec     = np.zeros((1,SSSRsize))

print pairskeys
print bondskeys
print atomskeys
print SSSRkeys

atomCountThreshold  = np.mean(atomCounts)+MAX_ATOM_COUNT_STDDEV*np.std(atomCounts)
vectors     = {}
for k,v in features.iteritems():
    pairvec     = np.zeros((1,pairsize))
    atomvec     = np.zeros((1,atomsize))
    bondvec     = np.zeros((1,bondsize))
    SSSRvec     = np.zeros((1,SSSRsize))
    pairs   = v[0]
    bonds   = v[1]
    atoms   = v[2]
    SSSR    = v[3]
    SSSRnum = v[4]
    
    molAtomCount   = np.sum(atoms.values())
    if molAtomCount < atomCountThreshold:
      
        
        for k2, v2 in pairs.iteritems():
            ind     = pairskeys.index(k2)
            pairvec[0,ind]  = v2
            
        for k2, v2 in bonds.iteritems():
            ind     = bondskeys.index(k2)
            bondvec[0,ind]  = v2
            
        for k2, v2 in atoms.iteritems():
            ind     = atomskeys.index(k2)
            atomvec[0,ind]  = v2
            
        for k2, v2 in SSSR.iteritems():
            k3  = int(k2.replace('SSSR_',''))
            if k3 <= MAX_SSSR_FEATURE_SIZE:
                ind     = SSSRkeys.index(k2)
                SSSRvec[0,ind]  = v2
            
        wholevec    = np.append(pairvec,bondvec)
        #wholevec    = np.append(wholevec,bondvec)
        wholevec    = np.append(wholevec,atomvec)
        wholevec    = np.append(wholevec,SSSRvec)
        wholevec    = np.append(wholevec,SSSRnum)
        
        vectors[k]  = wholevec

#featureVec will contain the names of the columns
featureVec  = [p for p in pairskeys]
[featureVec.append(bond) for bond in bondskeys]
[featureVec.append(atom) for atom in atomskeys]
[featureVec.append(SSSR) for SSSR in SSSRkeys]
featureVec.append("SSSRnum")

#see how many times each feature occurs in the dataset - sort by this number
sums    = np.sum(vectors.values(),axis=0)
sumDict = [ [featureVec[i],sums[i]] for i in range(0,len(featureVec))]
feature2= sorted(sumDict,key=itemgetter(1),reverse=True)
print feature2
#
#for k,v in vectors.iteritems():
#    pass    
print len(vectors.keys())

truncFeatures   = [feat[0] for feat in feature2 if feat[1] > FEATURE_FREQ_CUTOFF]
featuresOut     = {}
for k,v in vectors.iteritems():
    newvector = [v[i] for i in range(0,len(v)) if featureVec[i] in truncFeatures]
    featuresOut[k] = newvector

finalFeatures   =  [featureVec[i] for i in range(0,len(featureVec)) if featureVec[i] in truncFeatures]
print "*"*80 
print finalFeatures
print featuresOut[featuresOut.keys()[0]]
print featuresOut[featuresOut.keys()[1]]





means 	= np.mean(featuresOut.values(),axis=0)
stds 	= np.mean(featuresOut.values(),axis=0)

with open("../data/cidsMeansStds.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump([means, stds])

featuresScaled = {}
for k,v in featuresOut.iteritems():
    v		   = np.subtract(v,means)
    featuresScaled[k] = np.divide(v,stds) 


with open("../data/cidsFeaturesScaled.pickle",'wb') as f:
    cp = cPickle.Pickler(f)
    cp.dump(featuresScaled)

with open("../data/cidsFeatureVectors.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(featuresOut)
    
with open("../data/cidsFeatureKeys.txt",'wb') as f:
    f.write(','.join(finalFeatures))
