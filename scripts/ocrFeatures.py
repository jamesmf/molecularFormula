# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:17:45 2015

@author: test
"""
import cPickle
import numpy as np


MAX_SSSR_FEATURE_SIZE = 7   #Only include features for SSSRs of size < X
MAX_ATOM_COUNT_STDDEV = 2   #Only include molecules w/ (atom count) < (mean) + MAX_ATOM*(std_dev)


with open("../data/cidsFEATURES.pickle",'rb') as f:  
    features    = cPickle.load(f)
  
atomCounts  = []
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
SSSRkeys    = [k for k in SSSRkeys if k <= MAX_SSSR_FEATURE_SIZE] 
SSSRsize    = len(SSSRkeys)

pairvec     = np.zeros((1,pairsize))
atomvec     = np.zeros((1,atomsize))
bondvec     = np.zeros((1,bondsize))
SSSRvec     = np.zeros((1,SSSRsize))

print pairskeys
print bondskeys
print atomskeys
print SSSRkeys

atomCountThreshold  = MAX_ATOM_COUNT_STDDEV*np.std(atomCounts)
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
            if k2 <= MAX_SSSR_FEATURE_SIZE:
                ind     = SSSRkeys.index(k2)
                SSSRvec[0,ind]  = v2
            
        wholevec    = np.append(pairvec,bondvec)
        wholevec    = np.append(wholevec,bondvec)
        wholevec    = np.append(wholevec,atomvec)
        wholevec    = np.append(wholevec,SSSRvec)
        wholevec    = np.append(wholevec,SSSRnum)
        
        vectors[k]  = wholevec

print len(vectors.keys())
with open("../data/cidsFeatureVectors.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(vectors)
    