from __future__ import absolute_import, division

from os import listdir
from os.path import isfile, isdir, join, exists
import fileinput

def dirsize(dir):
    files = listdir(dir)
    counter = 0
    for f in files:
        if isfile(join(dir,f)):
            counter += 1
    return counter

def isValidDataDir(dir):
    rep = isdir(dir)
    rep = rep and isfile(join(dir,'labels.txt'))
    rep = rep and isfile(join(dir,'Projections.txt'))
    rep = rep and isdir(join(dir,'RGB'))
    rep = rep and isdir(join(dir,'Labels'))
    rep = rep and isdir(join(dir,'Depth'))
    rep = rep and isdir(join(dir,'Altitude'))
    return rep

def number_labels(dir):
    if not exists(join(dir,'labels.txt')):
        return 0
    counter = 0
    with fileinput.input((join(dir,'labels.txt'))) as f:
        for line in f:
            if not line[0]=='#':
               counter += 1
    return counter