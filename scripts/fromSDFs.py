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

"""Loop over every file in data/SDF/"""
for fi in ld:
    print fi
    """Read in the file using RDkit"""
    s = Chem.SDMolSupplier("../data/SDF/"+fi)
    molBonds    = {}
    molAtoms    = {}
    molBTypes   = {}
    countSSSR   = {}
    
    if s[0] is not None:
        try:
            mol     = s[0]
            bonds   = mol.GetBonds()
            #print Chem.MolToSmiles(mol)
            """Loop over the bonds to determine bond type, and atoms paired"""
            for bond in bonds:
                a1  = bond.GetBeginAtom().GetSymbol()
                a2  = bond.GetEndAtom().GetSymbol()
                bt  = '-'.join(sorted([a1,a2]))
                typ = bond.GetBondTypeAsDouble()
                if bt in molBonds:
                    molBonds[bt] +=1
                else:
                    molBonds[bt] = 1
                    
                if str(typ)+"_bond" in molBTypes:
                    molBTypes[str(typ)+"_bond"] +=1
                else:
                    molBTypes[str(typ)+"_bond"] = 1
#           print molBonds
#           print molBTypes
            
            """Loop over atoms to determine atom counts"""
            atoms   = mol.GetAtoms()
            for atom in atoms:
                a   = atom.GetSymbol()
                if a in molAtoms:
                    molAtoms[a] +=1
                else:
                    molAtoms[a] = 1
                    
#           print molAtoms
            """Loop over SSSRs to determine counts of SSSRs of each size"""
            SSSRs   = Chem.GetSymmSSSR(mol)
            for SSSR in SSSRs:
                size    = len(SSSR)
                if "SSSR_"+str(size) in countSSSR: 
                    countSSSR["SSSR_"+str(size)] +=1
                else:
                    countSSSR["SSSR_"+str(size)] = 1
            numSSSRs    = len(SSSRs)
            
            """Store these in a dictionary by cid"""
            features[fi.replace(".sdf",'')]     = [molBonds,molBTypes,molAtoms,countSSSR,numSSSRs]
#            print fi.replace(".sdf",'')            
#            print features[fi.replace(".sdf",'')]
#            stop=raw_input("")
        except  :
            #in case there is a problem, stop and print the mol object
            print s
            stop=raw_input("")
    
with open("../data/cidsFEATURES.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(features)