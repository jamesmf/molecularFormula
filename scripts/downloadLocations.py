# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:13:33 2015

@author: frickjm
"""

"""Hit the NCATS renderer to resolve the locations of each atom within an image"""
import skimage.io as io
from os import listdir
import numpy as np
import skimage
import urllib2
import cPickle
from os.path import isfile
import time
import socket
import re
import matplotlib.pyplot as plt

reg     = re.compile("([A-Z])")
  

# timeout in seconds
timeout = 2
socket.setdefaulttimeout(timeout)

size        = 200
size        = str(size)

urlbase2    = "http://tripod.nih.gov/servlet/renderServletv16/?structure=_REPLACE_&standardize=true&preset=DEFAULT&size=200&presetName=&format=png&rotate=0&amap=null&presetMOD={%22PROP_KEY_ATOM_LABEL_FONT_FRACTION%22:%220.132%22,%22PROP_KEY_DRAW_BONDS%22:false,%22PROP_KEY_DRAW_CARBON%22:true,%22PROP_KEY_DRAW_GREYSCALE%22:true,%22PROP_KEY_DRAW_HIGHLIGHT_SHOW_ATOM%22:false,%22PROP_KEY_DRAW_HIGHLIGHT_WITH_HALO%22:false,%22PROP_KEY_DRAW_SYMBOLS%22:false}&shadow=false"


directory   = "../data/images"+size+'/'
directory2  = "../data/locations"+size+"/"


CIDs    = []

with open("../data/AID_1996_datatable_all.csv",'rb') as f:
        for i in range(0,4):
            f.readline()
        for x in f:
            sp      = x.split(",")
            if len(sp) > 1:
                if x.find("Below LOQ") == -1:
                    #print sp
                    CID     = "cid"+sp[2].strip()
                    smiles  = sp[-1]
                    CIDs.append(CID)

                    
                    

print len(CIDs)

    
with open("../CIDs.txt",'wb') as f:
    for x in CIDs:
        f.write(x+"\n")




cantRender  = 0

imCount     = 0

for x in CIDs:
    imCount+=1 
    if imCount%1000 == 0:
        print imCount*1./54000
    #print x
    if not isfile(directory2+x+".png"):
        try:
            headers     = {}
            imgUrl      = urlbase2.replace("_REPLACE_",x)
            #print imgUrl
            imgRequest  = urllib2.Request(imgUrl, headers=headers)
            imgData     = urllib2.urlopen(imgRequest).read()
            
            with open(directory2+x+".png",'wb') as f:
                f.write(imgData)
             
        except urllib2.URLError as e: 
            cantRender +=1 
        
print "failed to render ", cantRender, "images"
