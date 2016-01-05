# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:49:48 2015

@author: frickjm
"""

from os import listdir
from os import mkdir
from os.path import isdir
import numpy as np
import skimage
import urllib2
import cPickle
from os.path import isfile
import time
import socket
import re

reg     = re.compile("([A-Z])")

# timeout in seconds
timeout = 1
socket.setdefaulttimeout(timeout)

size        = 200
size        = str(size)
urlbase     = "http://tripod.nih.gov/servlet/renderServletv16/?structure=_REPLACE_&format=png&shadow=false&size="+size+"&standardize=true"

directory   = "../data/images"+size+'/'
SDdir       = "../data/SDF/"

if not isdir(directory):
    mkdir(directory)
if not isdir(SDdir):
    mkdir(SDdir)
    

"""COMMENTED OUT BECAUSE WE HAVE ALREADY DOWNLOADED THESE"""    
#IDs    = []
#toDL   = []
#with open("../data/INN_DUMP_TAB1439497890663_withPubLink.txt",'rb') as f:
#    for i in range(0,4):
#        f.readline()
#    for x in f:
#        sp      = x.strip().split("\t")
#        if len(sp) > 1:
#            link     = sp[0]
#            name     = sp[1]
#            link2    = sp[-1]
#            
#        if not link2.find(" ") > -1:
#            toDL.append(link2)
#            IDs.append(name)
#
#
#
#for i in range(0,len(toDL)):
#    if i % 1000 == 0:
#        print i
#    headers = {}
#    try:
#        imgUrl      = toDL[i]
#        imgRequest  = urllib2.Request(imgUrl, headers=headers)
#        imgData     = urllib2.urlopen(imgRequest).read() 
#        #print imgData.split("\n")[0]
#        #print imgData
#        cid     = imgData.split("\n")[0]
#
#        with open(SDdir+cid+".sdf",'wb') as f:
#            f.write(imgData)
#
#    except:
#        pass
        

with open("../CIDs.txt",'rb') as f:
    CIDs     = [cid.strip() for cid in f.read().split("\n")]


urlbase     = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/_REPLACE_/record/SDF/?record_type=2d&response_type=display"
headers     = {}
for cid in CIDs:
    cid     = cid.replace("cid",'')
    if not isfile(SDdir+cid+".sdf"):
        print cid
        try:
            url         = urlbase.replace("_REPLACE_",cid)
            request     = urllib2.Request(url, headers=headers)
            data        = urllib2.urlopen(request).read()
            with open(SDdir+cid+".sdf",'wb') as f:
                f.write(data)
        except urllib2.URLError:
            pass
    else:
        pass

ld  = listdir("../data/supp/")
for fi in ld:
    
    with open("../data/supp/"+fi,'rb') as f:
        f.readline()
        rows    = [r for r in f.read().split("\n") if r != '']
        for r in rows:
            sp  = r.strip().split("\t")
            cid = sp[0]
            if not isfile(SDdir+cid+".sdf"):
                print cid
                try:
                    url         = urlbase.replace("_REPLACE_",cid)
                    request     = urllib2.Request(url, headers=headers)
                    data        = urllib2.urlopen(request).read()
                    with open(SDdir+cid+".sdf",'wb') as f:
                        f.write(data)
                except urllib2.URLError:
                    "failed, dawwwwwg!"
