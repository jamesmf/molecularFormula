# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:57:12 2015

@author: test
"""



"""create a service that keeps a folder full of randomly generated images"""
from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import sys
import time
import helperFuncs


#def makeSDFtest(test,indir):
#    #files   = test[:100]
#    files   = test
#    listd   = listdir("../data/INNimages/")
#    files   = [t for t in test if t.replace('.sdf','') in listd]
#    print files 
#    print len(files)
#    stop=raw_input("alsjdf")
#    
#
#    with open(indir+"tempSDF.sdf",'wb') as f:    
#        for fi in files:
#            fi  = fi.replace(".png",".sdf").replace("cid",'')
#            t   = file("../data/SDF/"+fi).read()
#            f.write(t)

size, indir, maximum, modifyAntiAlias, resize = helperFuncs.dataGenArgs(sys.argv[1:])



while not isfile(indir+"traindata.csv"):
    time.sleep(0.1)
    print "I'm sleeping", isdir(indir), indir, "                     \r",

if not isdir(indir+"tempTrain/"):
    mkdir(indir+"tempTrain/")
    mkdir(indir+"tempTest/")
       

print "reading Train/Test files"   
train   = [x for x in file(indir+"traindata.csv").read().split("\n") if x != '']    
test    = [x for x in file(indir+"testdata.csv").read().split("\n") if x != '']    

while True:
    ld  = listdir(indir+"tempTrain/")
    #ld  = listdir("temp/")

    if len(ld) > maximum:
        time.sleep(1)
        print "sleeping because Train folder full      \r",
    else:
        shuffle(train)
        helperFuncs.makeSDFtrain(train,indir)
        parameters  = helperFuncs.getParameters(indir,size=size,modifyAntiAlias=modifyAntiAlias,modifySize=resize)
        helperFuncs.callToRenderer(parameters,indir,indir+"tempTrain")
        #helperFuncs.callToRenderer(parameters,indir,"temp/")        
        
    ld2 = listdir(indir+"tempTest/")
    #ld2     = listdir("temp/")
    if len(ld2) > maximum/10:
        time.sleep(1)
	print "sleeping because Test folder full       \r",
    else:
	print len(ld2)
        shuffle(test)
        helperFuncs.makeSDFtest(test,indir)
        parameters  = helperFuncs.getParameters(indir,size=size,modifyAntiAlias=modifyAntiAlias,modifySize=resize)
        helperFuncs.callToRenderer(parameters,indir,indir+"tempTest")
        #helperFuncs.callToRenderer(parameters,indir,"temp/")
        
    
