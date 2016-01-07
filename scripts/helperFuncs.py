# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:46:08 2015

@author: frickjm
"""

import subprocess
import skimage
from skimage.transform import resize
from skimage import io
import numpy as np
import json
import getopt

from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import sys

from sklearn.metrics import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad




"""*******************************************************************"""            
"""                     Targets/Outputs                               """
"""*******************************************************************"""


def getTargets(targetType):
    targetType = targetType.lower()    
    
    if targetType == "ecfp":
        return getECFPTargets()
    elif targetType == "ocr":
        return getOCRTargets()
    elif targetType == "ocrscaled":
        return getOCRScaledTargets()
    elif targetType == "solubility":
        return getSolubilityTargets()

"""get the ECFP vectors for training"""
def getECFPTargets():
    ecfps = {}
    if isfile("../cidsECFP.pickle"):
        with open("../cidsECFP.pickle",'rb') as f:
            return cPickle.load(f)
    else:
        
        with open("../cidsECFP.txt",'rb') as f:
            f.readline() #ignore the header line
            for x in f:
                sp     = x.split("\t") 
                #ignore blank lines
                if len(sp) > 1:
                    CID         = sp[0]
                    vec         = np.array([int(x) for x in sp[2][1:-2].split(',')],dtype=np.float)
                    ecfps[CID]  = vec
        
        with open("../cidsECFP.pickle",'wb') as f:
            cp     = cPickle.Pickler(f)
            cp.dump(ecfps)
    return ecfps
    

"""get the solubility for training"""
def getOCRTargets():
    with open("../data/cidsFeatureVectors.pickle",'rb') as f:
        d =  cPickle.load(f)
    with open("../data/cidsFeatureKeys.txt",'rb') as f:
        keys    = [x for x in f.read().split(",") if x != '']
    return d, keys

def getOCRScaledTargets():
   with open("../data/cidsFeaturesScaled.pickle",'rb') as f:
	d = cPickle.load(f)
   with open("../data/cidsFeatureKeys.txt",'rb') as f:
	keys = [x for x in f.read().split(",") if x != '']
   return d, keys    
    
def getSolubilityTargets():
    out     = {}
    with open("../data/sols.pickle",'rb') as f:
        d =  cPickle.load(f)
        for k,v in d.iteritems():
            out[k] = [float(v)]
    return out   

def getMeansStds():
    with open("../data/cidsMeansStds.pickle",'rb') as f:
        ret     = cPickle.load(f)
    return ret


"""*******************************************************************"""            
"""                             Weights                               """
"""*******************************************************************"""
    
def getWeights(model):
    with open(model,'rb') as f:
        return cPickle.load(f).get_weights()
    

def dumpWeights(model,folder):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                with open(folder+"layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1
        

"""*******************************************************************"""            
"""                     Training Functions                            """
"""*******************************************************************"""
def trainingArgs(argv):
    try:
       opts, args = getopt.getopt(argv,"s:d:m:Ar",["size=","indir=","max=","antiAlias","resize"])
    except getopt.GetoptError:
       print 'dataGenerator.py -s <imageSize> -d <directory> -m <max number of images to create>'
       sys.exit(2)
    
    resize              = True
    modifyAntiAlias     = True
    for opt, arg in opts:

       if opt in ("-s", "--size"):
          size = int(arg)
       elif opt in ("-d", "--indir"):
          indir = arg.strip()
       elif opt in ("-m", "--max"):
          maximum = int(arg)
       elif opt in ("-m", "--max"):
          modifyAntiAlias = (int(arg) == 1)
       elif opt in ("-m", "--max"):
          resize = (int(arg) == 1)
          
    print "Pixel Size" , size
    print 'Model Directory: ', indir
    print 'Maximum Number of Images to Create: ', maximum
    print 'Turn on/off Anti-Aliasing: ', modifyAntiAlias
    print "Resize Images (if False, image size = size): ", resize
    return size, indir, maximum, modifyAntiAlias, resize


def getTrainTestSplit(update,folder,numEx="",trainTestSplit="",ld=""):
    if not update:
        trainFs = ld[:int(numEx*trainTestSplit)]
        testFs  = ld[int(numEx*trainTestSplit):]
        with open(folder+"traindata.csv",'wb') as f:
            f.write('\n'.join(trainFs))
        with open(folder+"testdata.csv",'wb') as f:        
            f.write('\n'.join(testFs))

    else:
        with open(folder+"traindata.csv",'rb') as f:
            trainFs = f.read().split("\n")
        with open(folder+"testdata.csv",'rb') as f:        
            testFs  = f.read().split("\n")
            
    return trainFs, testFs
    
    
def handleArgs(arglist,size=200):  
    if len(arglist) < 2:
        print "needs folder as input"
        sys.exit(1)
    else:
        folder  = arglist[1]
        temp    = folder[folder.find("/")+1:]
        targetType    = temp[:temp.find("/")]
        if folder[-1] == "/":
            folder = folder[:-1]
        f1      = folder[folder.rfind("/")+1:]
        f2      = f1[:f1.find("_")]
        f3      = f1[f1.find("_")+1:]   
        size    = int(f2)                    #size of the images
        #print f3
        run     = f3
        


    print  "size: ", size, "run: ", run, "targetType: ", targetType
    return size, run, targetType



def defineFolder(update,outType,size,run):
    if run != '':
        run     = "_"+run
    
    folder  = "../"+outType+'/'+str(size)+run+"/"
        
    if (run == "_1") and (isdir(folder)) and (not update):
        i=1
        oldfolder = folder[:folder.rfind("_")+1]
        while isdir(folder):
            i+=1
            folder  = oldfolder[:-1]+"_"+str(i)+'/'
            print folder
            
    if not update:
        mkdir(folder)
        mkdir(folder+"tempTest/")        
        mkdir(folder+"tempTrain/")
    return folder

"""*******************************************************************"""            
"""                     Image Generation                              """
"""*******************************************************************"""
def dataGenArgs(argv):
    try:
       opts, args = getopt.getopt(argv,"s:d:m:a:r:",["size=","indir=","max=","antiAlias","resize"])
    except getopt.GetoptError:
       print 'dataGenerator.py -s <imageSize> -d <directory> -m <max number of images to create>'
       sys.exit(2)
    
    resize              = True
    modifyAntiAlias     = True
    for opt, arg in opts:
        print arg, opt
        if opt in ("-s", "--size"):
            size = int(arg)
        elif opt in ("-d", "--indir"):
            indir = arg.strip()
        elif opt in ("-m", "--max"):
            maximum = int(arg)
        elif opt in ("-a", "--antiAlias"):
            arg  = arg.strip()
            modifyAntiAlias = (int(arg) == 1)
        elif opt in ("-r", "--resize"):
            resize = (int(arg) == 1)
          
    print "Pixel Size" , size
    print 'Model Directory: ', indir
    print 'Maximum Number of Images to Create: ', maximum
    print 'Turn on/off Anti-Aliasing: ', modifyAntiAlias
    print "Resize Images (if False, image size = size): ", resize
    return size, indir, maximum, modifyAntiAlias, resize


def callToRenderer(parameters,indir,outdir):
    subprocess.call(["java","-jar","../renderer2.jar",parameters,outdir])


def getParameters(indir,modifySize=True,modifyAntiAlias=True,size=200):
    ps = "structure="+indir+"tempSDF.sdf&standardize=true&shadow=false&preset=DEFAULT&_SIZE_&format=png&amap=null_ANTIALIAS_&presetMOD="
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
        
    #Choose values for renderer - normally distributed at means[i], capped by [mins[i], maxs[i]]
    mins    = [2.0, 0.1, 0.4, 0.12, 0.06, 0.16]
    maxs    = [8.0, 0.4, 0.9, 0.36, 0.15, 0.5]    
    means   = [4,0.25,0.65,0.24,0.1,0.33]
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
    #print values
    
    if modifySize:
        sizeStr     = "size="+str(int(150 +(size-150)*np.random.rand()))
    else:
        sizeStr     = "size="+str(size)
    
    count   = 0
    d       = {}
    for key in keys:
        d[key] = values[count]
        count+=1
        
    jstr    = json.dumps(d)
    if modifyAntiAlias:
        antiAlias   = "&antialias="+str(np.random.rand() > 0.50).lower()
    else:
        antiAlias   = ""        
    
    ps2     = ps+jstr
    ps2     = ps2.replace("_ANTIALIAS_",antiAlias)
    ps2     = ps2.replace("_SIZE_",sizeStr)
    ps2     = ps2.replace(' ','').replace('"',"'")
    print ps2
    return ps2
    
def makeSDFtrain(train,indir):
    files   = train[:500]

    with open(indir+"tempSDF.sdf",'wb') as f:    
        for fi in files:
            fi  = fi.replace(".png",".sdf").replace("cid",'')
            t   = file("../data/SDF/"+fi).read()
            f.write(t)
            
def makeSDFtest(test,indir):
    files   = test[:100]

    with open(indir+"tempSDF.sdf",'wb') as f:    
        for fi in files:
            fi  = fi.replace(".png",".sdf").replace("cid",'')
            t   = file("../data/SDF/"+fi).read()
            f.write(t)
            
            
"""*******************************************************************"""            
"""                     Model Definition                              """
"""*******************************************************************"""



"""*******************************************************************"""            
"""                     Data Processing                               """
"""*******************************************************************"""
def dataProcessorArgs(argv):
    try:
       opts, args = getopt.getopt(argv,"s:d:t:bB:p:",["size=","indir=","targetType=","binarize","blur=","padding"])
    except getopt.GetoptError:
       print 'dataProcessor.py -s <imageSize> -d <directory> -t <targetType "ocr","ecfp","solubility">'
       sys.exit(2)
    
    blur         = True
    binarize     = True
    padding      = "random"
    for opt, arg in opts:

       if opt in ("-s", "--size"):
          size          = int(arg)
          
       elif opt in ("-d", "--indir"):
          indir         = arg.strip()
          
       elif opt in ("-b", "--binarize"):
          binarize      = bool(int(arg))
          
       elif opt in ("-B", "--blur"):
          blur          = float(arg)
          
       elif opt in ("-p", "--padding"):
          padding       = arg
          
       elif opt in ("-t", "--targetType"):
          targetType   = arg

    print "Target Type: " , targetType
    print "Pixel Size" , size
    print 'Model Directory: ', indir
    print 'Blur stdev (0 if off): ', blur
    print 'Binarize: ', binarize
    print "Resize Images (if False, image size = size): ", padding
    return size, indir, binarize, blur, padding, targetType
    

def processImage(CID,folder,binarize,blur,padding,size,noise=False):
    
    image   = io.imread(folder+CID+".sdf",as_grey=True)
    output  = np.zeros((size,size))
    if binarize:
        image   = np.where(image > 0.1,1.0,0.0)
    if blur > 0:
        image   = skimage.filters.gaussian_filter(image,blur)
    if padding == "random":
        pad     = int(np.random.rand()*50)
        image   = resize(image,(size-pad,size-pad))
        
    d   = int(pad/2)
    output [d:d+image.shape[0],d:d+image.shape[1]]  = image
    if noise:
        output  = 0.05*np.random.rand(output.shape[0], output.shape[1]) + output        
        #output   = np.where(output == 0., 0.1*np.random.rand(),output)
    return output
