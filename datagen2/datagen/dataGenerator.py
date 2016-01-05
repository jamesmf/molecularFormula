# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:57:12 2015

@author: test
"""



"""create a service that keeps a folder full of randomly generated images"""

import skimage.io as io
from skimage.transform import resize 
import numpy as np

from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import sys
import time
import subprocess
import json

def makeSDFtrain(train,indir):
    files   = train[:1000]

    with open(indir+"tempSDF.sdf",'wb') as f:    
        for fi in files:
            fi  = fi.replace(".png",".sdf").replace("cid",'')
            t   = file("../data/SDF/"+fi).read()
            f.write(t)

def makeSDFtest(train,indir):
    files   = test[:100]

    with open(indir+"tempSDF.sdf",'wb') as f:    
        for fi in files:
            fi  = fi.replace(".png",".sdf").replace("cid",'')
            t   = file("../data/SDF/"+fi).read()
            f.write(t)

def callToRenderer(parameters,indir,outdir):
    subprocess.call(["java","-jar","../renderer.jar",parameters,outdir])


def getParameters(indir,size=200):
    ps = "structure="+indir+"tempSDF.sdf&standardize=true&shadow=false&preset=DEFAULT&size="+str(size)+"&presetName=&format=png&amap=null&presetMOD="
    """parameters to define:
        PROP_KEY_BOND_STEREO_DASH_NUMBER from 2 to 8
        PROP_KEY_BOND_DOUBLE_GAP_FRACTION from .1 to .4
        PROP_KEY_ATOM_LABEL_FONT_FRACTION: 0.30 to 0.924
        PROP_KEY_BOND_DOUBLE_GAP_FRACTION: 0.12 to 0.36
        PROP_KEY_BOND_STROKE_WIDTH_FRACTION: 0.02 to 0.152
        PROP_KEY_BOND_STEREO_WEDGE_ANGLE: 0.15707964 to 0.5
        ROTATE  = 0 2*3.141592
        """
    keys    = [
        "PROP_KEY_BOND_STEREO_DASH_NUMBER",
        "PROP_KEY_BOND_DOUBLE_GAP_FRACTION",
        "PROP_KEY_ATOM_LABEL_FONT_FRACTION",
        "PROP_KEY_BOND_DOUBLE_GAP_FRACTION",
        "PROP_KEY_BOND_STROKE_WIDTH_FRACTION",
        "PROP_KEY_BOND_STEREO_WEDGE_ANGLE",
        "PROP_KEY_DRAW_CENTER_NONRING_DOUBLE_BONDS",
        "PROP_KEY_DRAW_STEREO_DASH_AS_WEDGE",
        "PROP_KEY_DRAW_IMPLICIT_HYDROGEN",
        "ROTATE"
        ]
    mins    = [2.0, 0.1, 0.3, 0.12, 0.06, 0.16]
    maxs    = [8.0, 0.4, 0.9, 0.36, 0.15, 0.5]    
    means   = [4,0.25,0.61,0.24,0.1,0.33]
    stds    = [1.5,0.075,0.15,0.06,0.027,0.09]
    rands   = np.random.rand(9)
    vals1   = [np.random.normal(means[i],stds[i]) for i in range(0,6)]
    vals1   = [min(vals1[i],maxs[i]) for i in range(0,6)]
    vals1   = [max(vals1[i],mins[i]) for i in range(0,6)]
    #vals1   = [means[i]+stds[i]*rands[i] for i in range(0,6)]
    vals2   = [j == 1. for j in np.round(rands[6:])]
    rot     = np.random.rand()*2*3.141592
    
    values = []
    [values.append(v) for v in vals1]
    [values.append(str(v).lower()) for v in vals2]
    values.append(rot)
   
#    values  = np.append(vals1,vals2)
#    values  = np.append(values,rot)
    print values
    
    count   = 0
    d       = {}
    for key in keys:
        d[key] = values[count]
        count+=1
        
    jstr    = json.dumps(d)
    ps2     = ps+jstr

        
    
    
    
    return ps2


indir       = sys.argv[1]
IN_VALUE    = int(sys.argv[2])

if len(sys.argv) > 3:
    size 	= sys.argv[3] 
else:
    size 	= 200

while not isfile(indir+"traindata.csv"):
    time.sleep(0.1)
    print "I'm sleeping", isdir(indir), indir, "           \r",

if not isdir(indir+"tempTrain/"):
    mkdir(indir+"tempTrain/")
    mkdir(indir+"tempTest/")
    
#    subprocess.call("cp ../renderer.jar "+indir+"tempTrain/",shell=True)
#    subprocess.call("cp ../renderer.jar "+indir+"tempTest/",shell=True)    

    
train   = [x for x in file(indir+"traindata.csv").read().split("\n") if x != '']    
test    = [x for x in file(indir+"testdata.csv").read().split("\n") if x != '']    

while True:
    ld  = listdir(indir+"tempTrain/")
    
    if len(ld) > IN_VALUE:
        time.sleep(1)
	print "sleeping because Train folder full      \r",
    else:
        shuffle(train)
        makeSDFtrain(train,indir)
        parameters  = getParameters(indir,size=size)
        callToRenderer(parameters,indir,indir+"tempTrain")
        
        
    ld2 = listdir(indir+"tempTest/")
    if len(ld2) > IN_VALUE/10:
        time.sleep(1)
	print "sleeping because Test folder full       \r",
    else:
	print len(ld2)
        shuffle(test)
        makeSDFtest(test,indir)
        parameters  = getParameters(indir,size=size)
        callToRenderer(parameters,indir,indir+"tempTest")
        
    
