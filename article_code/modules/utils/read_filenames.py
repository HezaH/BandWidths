import os
from os import listdir
from os.path import isfile, join

def readFilesInDict(path, extension):
    onlymtxComp = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(extension)]
    return onlymtxComp
