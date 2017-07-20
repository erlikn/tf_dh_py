# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import os
import time
import math
import random
from shutil import copy
import numpy as np

from joblib import Parallel, delayed
import multiprocessing


def process_dataset(readFolder, filenames, writeFolder, imgsqrsize, id):
    filename=filenames[id]
    img = cv2.imread(readFolder+filename, 0)
    img = cv2.resize(img, (imgsqrsize, imgsqrsize))
    cv2.imwrite(writeFolder+str(id)+".jpg", img)

def prepare_dataset(readFolder, writeFolder, imgsqrsize):
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
    filenames.sort()
    #for filename in filenames:
    num_cores = multiprocessing.cpu_count()-1
    Parallel(n_jobs=num_cores)(delayed(process_dataset)(readFolder, filenames, writeFolder, imgsqrsize, i) for i in range(len(filenames)))
    print('100%  Done')

####################################
dataRead = "../../Data/clutter_single/"
dataOut = "../../Data/clutter/clutter_sin_64/"
prepare_dataset(dataRead, dataOut, 64)
print('Single 64 done -----------------------')
dataOut = "../../Data/clutter/clutter_sin_32/"
prepare_dataset(dataRead, dataOut, 32)
print('Single 32 done -----------------------')
dataOut = "../../Data/clutter/clutter_sin_16/"
prepare_dataset(dataRead, dataOut, 16)
print('Single 16 done -----------------------')

dataRead = "../../Data/clutter_all/"
dataOut = "../../Data/clutter/clutter_all_64/"
prepare_dataset(dataRead, dataOut, 64)
print('All 64 done -----------------------')
dataOut = "../../Data/clutter/clutter_all_32/"
prepare_dataset(dataRead, dataOut, 32)
print('All 32 done -----------------------')
dataOut = "../../Data/clutter/clutter_all_16/"
prepare_dataset(dataRead, dataOut, 16)
print('All 16 done -----------------------')
