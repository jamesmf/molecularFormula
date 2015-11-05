
from os import listdir
import numpy as np
import skimage
import urllib2
import cPickle
from os.path import isfile
import time
import socket
import re

reg     = re.compile("([A-Z])")


def handleLine(x,reg):
    nums    = re.compile("[0-9]+")
    d       = {}
    sp      = x.split("\t")
    CID     = sp[0]
    molf    = sp[2]
    m2      = re.sub(reg," \g<0>",molf)
    msp     = m2.split(" ")
    for x in msp:
        if x != '':
            atom    = re.sub(nums,'',x)
            count   = re.findall(nums,x)
            if len(count) == 0:
                count = 1
            else:
                count = count[0]
            d[atom] = count
    return CID, d, sp[1]
    
    

# timeout in seconds
timeout = 2
socket.setdefaulttimeout(timeout)

size        = 200
size        = str(size)
urlbase     = "http://tripod.nih.gov/servlet/renderServletv16/?structure=_REPLACE_&format=png&shadow=false&size="+size+"&standardize=true"

directory   = "../data/images"+size+'/'


CIDs    = []
sols    = []
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
                    sol     = sp[8]
                    print CID, sol
                    CIDs.append(CID)
                    sols.append(sol)
                    
                    
print len(sols)
print len(CIDs)
solDict     = {}
for i in range(0,len(sols)):
    solDict[CIDs[i]] = sols[i]
    
with open("../CIDs.txt",'wb') as f:
    
    for x in CIDs:
        f.write(x+"\n")

atomset     = set()        
with open("../cidsMF.txt",'rb') as f:
    for x in f:
        CID, counts, mwt    = handleLine(x,reg)
        print CID, counts
        atomset = atomset | set(counts.keys())

print atomset
atomlist    = list(atomset)
cidMFs      = {}
print atomlist
stop=raw_input("")
with open("../cidsMF.pickle",'wb') as fout:
    with open("../cidsMF.txt",'rb') as f:
        for x in f:
            CID, counts, mwt    = handleLine(x,reg)
            print CID, counts
            vec     = np.zeros(len(atomlist))
            for k,v in counts.iteritems():
                ind     = atomlist.index(k)
                vec[ind]= v 
                
            cidMFs[CID] = vec

    print len(cidMFs.keys())
    print len(cidMFs[cidMFs.keys()[0]])
    cp  = cPickle.Pickler(fout)
    cp.dump(cidMFs)    
stop=raw_input("")

with open("../data/sols.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(solDict)


cantRender  = 0

for x in CIDs:
    if not isfile(directory+x+".png"):
        try:
            headers     = {}
            imgUrl      = urlbase.replace("_REPLACE_",x)
            imgRequest  = urllib2.Request(imgUrl, headers=headers)
            imgData     = urllib2.urlopen(imgRequest).read()
            
            with open(directory+x+".svg",'wb') as f:
                f.write(imgData)
        except urllib2.URLError, e:
            cantRender +=1 
        
print "failed to render ", cantRender, "images"
    