# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import os
import random
from shutil import copy
from shutil import move
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing

###################################
def image_resize_write(img, obstclSiz, obstclFld, idx, k):
    img = cv2.resize(img, (obstclSiz[k],obstclSiz[k]))
    cv2.imwrite(obstclFld[k]+str(idx)+".jpg", img)
###################################
def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

####################################
dataRead = "../../Data/MSCOCO_orig/"
dataReadGray = "../../Data/MSCOCO_gray/"
test640 = "../../Data/640_480_test/"
noiseRead = "../../Data/101_ObjectCategories/"

# setup folders
# obstacle 16x16
testtfRecordFLD = "../../Data/128_test_tfrecords_ob_16/"
_set_folders(testtfRecordFLD)
obstaclefolder16 = "../../Data/clutter_16/"
_set_folders(obstaclefolder16)
# obstacle 32x32
testtfRecordFLD = "../../Data/128_test_tfrecords_ob_32/"
_set_folders(testtfRecordFLD)
obstaclefolder32 = "../../Data/clutter_32/"
_set_folders(obstaclefolder32)
# obstacle 64x64
testtfRecordFLD = "../../Data/128_test_tfrecords_ob_64/"
_set_folders(testtfRecordFLD)
obstaclefolder64 = "../../Data/clutter_64/"
_set_folders(obstaclefolder64)

obstclFld = list()
obstclFld.append(obstaclefolder16)
obstclFld.append(obstaclefolder32)
obstclFld.append(obstaclefolder64)
obstclSiz = list()
obstclSiz.append(16)
obstclSiz.append(32)
obstclSiz.append(64)

WRTIE = True
# Move noise data to one location
idx=100000
for fileList in os.walk(noiseRead):
    #first fileList will be the level1 folders
    #Rest will be recursive level2 folders with files
    if fileList[2] == 0:
        continue # skip the first level --- as there are no files here
    else:
        # fileList[0] = folder0+folder1
        # fileList[1] = foldernames
        # fileList[2] = filenames
        print("Resizing folder ", str(fileList[0]))
        for j in range(len(fileList[2])):
            img = cv2.imread(fileList[0]+"/"+fileList[2][j], 0)
#            for k in range(3):
#                image_resize_write(img, obstclSiz, obstclFld, idx, k)
            num_cores = 3
            Parallel(n_jobs=num_cores)(delayed(image_resize_write)(img, obstclSiz, obstclFld, idx, k) for k in range(3))
            idx+=1

print("All noise images are resized and located at corresponding folders")