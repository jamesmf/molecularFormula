# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:26:10 2015

@author: test
"""


from rdkit import Chem
from os import listdir
from os.path import isdir
import cPickle


"""
Generates the following features:

- atomPairs: counts of bonded atoms in the molecule, e.g. 'C-N':1, 'C-C': 5...
- bondTypes: counts of bond number e.g. '1':6, '2':2, '1.5':6 ...
- atomCount: counts of atoms e.g. 'C':10, 'H':18 ...
- SSSRCount: counts of SSSRs of various sizes e.g. '3':1, '5':1, '6':2
- SSSRnum  : count of SSSRs of all sizes in the molecule

"""

ld  = listdir("../data/SDF/")
features    = {} #This will be our feature dictionary


for fi in ld:
    print fi
    s = Chem.SDMolSupplier("../data/SDF/"+fi)
    molBonds    = {}
    molAtoms    = {}
    molBTypes   = {}
    countSSSR   = {}
    if s is not None:
        mol     = s[0]
        bonds   = mol.GetBonds()
        print Chem.MolToSmiles(mol)
        for bond in bonds:
            a1  = bond.GetBeginAtom().GetSymbol()
            a2  = bond.GetEndAtom().GetSymbol()
            bt  = '-'.join(sorted([a1,a2]))
            typ = bond.GetBondTypeAsDouble()
            if bt in molBonds:
                molBonds[bt] +=1
            else:
                molBonds[bt] = 1
                
            if typ in molBTypes:
                molBTypes[typ] +=1
            else:
                molBTypes[typ] = 1
        print molBonds
        print molBTypes
        
        atoms   = mol.GetAtoms()
        for atom in atoms:
            a   = atom.GetSymbol()
            if a in molAtoms:
                molAtoms[a] +=1
            else:
                molAtoms[a] = 1
                
        print molAtoms
        
        SSSRs   = Chem.GetSymmSSSR(mol)
        for SSSR in SSSRs:
            size    = len(SSSR)
            if size in countSSSR: 
                countSSSR[size] +=1
            else:
                countSSSR[size] = 1
        numSSSRs    = len(SSSRs)
        
        features[fi.replace(".sdf",'')]     = [molBonds,molBTypes,molAtoms,countSSSR,numSSSRs]

    
with open("../data/cidsFEATURES.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(features)