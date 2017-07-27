# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import time
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
#import tensorflow as tf

from joblib import Parallel, delayed
import multiprocessing

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tfrecord_io

# Global Variables
NUM_OF_TEST_PERTURBED_IMAGES = 5
NUM_OF_TRAIN_PERTURBED_IMAGES = 5


def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def image_process_subMean_divStd(img):
    out = img - np.mean(img)
    out = out / img.std()
    return out

def image_process_subMean_divStd_n1p1(img):
    out = img - np.mean(img)
    out = out / img.std()
    out = (2*((out-out.min())/(out.max()-out.min())))-1
    return out

def perturb_writer( ID, idx,
                    imgOrig, imgPatchOrig, imgPatchPert, targetHAB, pOrig,
                    tfRecFolder):
    ##### original patch
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_orig.jpg")
    #cv2.imwrite(patchFolder+filename, imgPatchOrig)
    ##### perturbed patch
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_pert.jpg")
    #cv2.imwrite(patchFolder+filename, imgPatchPert)
    ##### HAB
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_HAB.csv")
    #with open(HABFolder+filename, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    if (HAB.ndim > 1):
    #        writer.writerows(HAB)
    #    else:
    #        writer.writerow(HAB)
    ##### Original square
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +".csv")
    #with open(squareFolder+filename, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    if (pOrig.ndim > 1):
    #        writer.writerows(pOrig)
    #    else:
    #        writer.writerow(pOrig)
    # Tensorflow record
    filename = str(ID) + "_" + str(idx)
    fileID = [ID, idx]
    tfrecord_io.tfrecord_writer(imgOrig, imgPatchOrig, imgPatchPert, pOrig, targetHAB, targetHAB*0, tfRecFolder, filename, fileID)

    #imgOp = image_process_subMean_divStd(imgPatchOrig)
    #imgPp = image_process_subMean_divStd(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolder+filename)
    # Tensorflow record in range -1 and 1
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx))
    #imgOp = image_process_subMean_divStd_n1p1(imgPatchOrig)
    #imgPp = image_process_subMean_divStd_n1p1(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolderN1P1+filename)

    return

def generate_random_perturbations(datasetType, img, ID, num, tfRecFolder, obsFolder, noiseFilenames):
    if "train" in datasetType:
        # if 320x240 => 128x128 w thrPerturbation=32
        squareSize=128
        thrPerturbation=32
    if "test" in datasetType:
        # if 640x480 => 256x256 w thrPerturbation=64
        squareSize=256
        thrPerturbation=64
    noiseListSize = len(noiseFilenames)
    if noiseListSize > 0:
        rndListObsSrc = random.sample(range(len(noiseFilenames)), num)
        rndListObsDst = random.sample(range(len(noiseFilenames)), num)
    rndListRowOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    rndListColOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    for i in range(0, len(rndListRowOrig)):
        # read first random perturbation
        pRow = rndListRowOrig[i]
        pCol = rndListColOrig[i]
        imgOrigCopy = img.copy()
        imgTempOrig = imgOrigCopy[pRow:pRow+squareSize, pCol:pCol+squareSize]
        # p & 0 is top left    - 1 is top right
        # 2     is bottom left - 3 is bottom right
        pOrig = np.array([[pRow, pRow, pRow+squareSize, pRow+squareSize],
                          [pCol, pCol+squareSize, pCol, pCol+squareSize]], np.float32)
        # generate random perturbations (H^AB)
        rndListRowPert = np.asarray(random.sample(range(-thrPerturbation, thrPerturbation), 4))
        rndListColPert = np.asarray(random.sample(range(-thrPerturbation, thrPerturbation), 4))
        H_AB = np.asarray([rndListRowPert, rndListColPert], np.float32)
        #
        pPert = np.asarray(pOrig+H_AB)
        # get transformation matrix and transform the image to new space
        Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
        dst = cv2.warpPerspective(img, Hmatrix, (img.shape[1], img.shape[0]))
        #print(img.shape)
        # crop the image at original location
        imgTempPert = dst[pRow:pRow+squareSize, pCol:pCol+squareSize].copy()
        if dst.max() > 256:
            print("NORMALIZATION to uint8 NEEDED!!!!!!!!!!")
        # Write down outputs
        # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
        if "test" in datasetType:
            imgTempOrig = cv2.resize(imgTempOrig, (128,128))
            imgTempPert = cv2.resize(imgTempPert, (128,128))
            H_AB = H_AB/2
        # attach obstacles
        global WRTIE
        # Read grayscale obstacle image
        if noiseListSize > 0: 
            imgOb = cv2.imread(obsFolder+noiseFilenames[rndListObsSrc[i]], 0)
            # normalize obstacle image
            if imgOb.max()-imgOb.min() > 0:
                imgOb = (imgOb-imgOb.min())/(imgOb.max()-imgOb.min())
            imgOb = np.asarray(imgOb, np.float32)
            # Attach Obstacle
            ob_loc_row = random.sample(range(0,imgTempOrig.shape[0]-imgOb.shape[0]),1)[0]
            ob_loc_col = random.sample(range(0,imgTempOrig.shape[1]-imgOb.shape[1]),1)[0]
            imgTempOrig[ob_loc_row:ob_loc_row+imgOb.shape[0], ob_loc_col:ob_loc_col+imgOb.shape[1]] = imgOb
            # Read grayscale obstacle image
            imgOb = cv2.imread(obsFolder+noiseFilenames[rndListObsDst[i]], 0)
            # normalize obstacle image
            if imgOb.max()-imgOb.min() > 0:
                imgOb = (imgOb-imgOb.min())/(imgOb.max()-imgOb.min())
            imgOb = np.asarray(imgOb, np.float32)
            ob_loc_row = random.sample(range(0,imgTempPert.shape[0]-imgOb.shape[0]),1)[0]
            ob_loc_col = random.sample(range(0,imgTempPert.shape[1]-imgOb.shape[1]),1)[0]
            imgTempPert[ob_loc_row:ob_loc_row+imgOb.shape[0], ob_loc_col:ob_loc_col+imgOb.shape[1]] = imgOb
        
        if WRTIE:
            cv2.imwrite("img_orig.jpg", imgTempOrig*255)
            cv2.imwrite("img_pert.jpg", imgTempPert*255)
            WRTIE = False

        #cv2.imshow('image',imgOrigCopy)
        #cv2.waitKey(0)
        #cv2.imshow('imageOrig',imgTempOrig)
        #cv2.waitKey(0)
        #cv2.imshow('imagePert',imgTempPert)
        #cv2.waitKey(0)
        perturb_writer(ID, i,
                       imgOrigCopy, imgTempOrig, imgTempPert, H_AB, pOrig,
                       tfRecFolder)
    return

def process_dataset(filenames, datasetType, readFolder, tfRecFolder, obsFolder, noiseFilenames, id):
    filename=filenames[id]
    if "train" in datasetType:
        num = NUM_OF_TRAIN_PERTURBED_IMAGES
    else: # test
        num = NUM_OF_TEST_PERTURBED_IMAGES
    img = cv2.imread(readFolder+filename, 0)
    # normalize image
    if img.max()-img.min() > 0:
        img = (img-img.min())/(img.max()-img.min())
    img = np.asarray(img, np.float32)
    # make sure grayscale
    if img.ndim == 2:
        generate_random_perturbations(datasetType, img, id+1000000, num, tfRecFolder, obsFolder, noiseFilenames)
        #tMu = tMu + mu
        #tVar = tVar + var
        #totalCount = totalCount + (num)
    else:
        print("Not a grayscale")

def prepare_dataset(datasetType, readFolder, tfRecFolder, obsFolder, start):
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
    filenames.sort()
    if obsFolder == "":
        noiseFilenames = list()
    else:
        noiseFilenames = [f for f in listdir(obsFolder) if isfile(join(obsFolder, f))]
        random.shuffle(noiseFilenames)
    if (int(len(filenames)/3) > (start+100)):
        end = int(len(filenames)/3)
    elif (2*int(len(filenames)/3) > start+100):
        end = 2*int(len(filenames)/3)
    else:
        end = len(filenames)
    #for i in range(len(filenames)):
    #    process_dataset(filenames, datasetType, readFolder, tfRecFolder, obsFolder, noiseFilenames, i)
    num_cores = multiprocessing.cpu_count()-2
    Parallel(n_jobs=num_cores)(delayed(process_dataset)(filenames, datasetType, readFolder, tfRecFolder, obsFolder, noiseFilenames, i) for i in range(start, end))
    print('   Done')
    return end

WRTIE = True

train320 = "../../Data/320_240_train/"
test640 = "../../Data/640_480_test/"

testtfRecordFLD = "../../Data/128_test_tfrecords_mix/"
traintfRecordFLD = "../../Data/128_train_tfrecords_mix/"
_set_folders(testtfRecordFLD)
_set_folders(traintfRecordFLD)

obstaclefolder32 = "../../Data/clutter_32/"
obstaclefolder64 = "../../Data/clutter_64/"

# NO OBSTACLES
#print('Wrting train tfrecords cl... from: 0')
#next = prepare_dataset("train", train320, traintfRecordFLD, "", 0)

# obstacle 32x32
#print('Wrting train tfrecords 32... from: ', next)
#next = prepare_dataset("train", train320, traintfRecordFLD, obstaclefolder32, next)


#next = 0

# obstacle 64x64
#print('Wrting train tfrecords 64... from: ', next)
#next = prepare_dataset("train", train320, traintfRecordFLD, obstaclefolder64, next)


print('Wrting test tfrecords cl...from: 0')
next = prepare_dataset("test", test640, testtfRecordFLD, "", 0)
print('Wrting test tfrecords 32... from: ', next)
next = prepare_dataset("test", test640, testtfRecordFLD, obstaclefolder32, next)
print('Wrting test tfrecords 64... from: ', next)
next = prepare_dataset("test", test640, testtfRecordFLD, obstaclefolder64, next)
