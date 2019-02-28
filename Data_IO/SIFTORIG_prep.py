# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, isdir, join
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
NUM_OF_TRAIN_PERTURBED_IMAGES = 7


def perturb_writer( ID, idx,
                    imgOrig, imgPatchOrig, imgPatchPert, HAB, pOrig,
                    tfRecFolder):
    filename = str(ID) + "_" + str(idx)
    fileID = [ID, idx]
    tfrecord_io.tfrecord_writer(imgOrig, imgPatchOrig, imgPatchPert, pOrig, HAB, tfRecFolder, filename, fileID)
    return

def generate_random_perturbations(datasetType, img, ID, num, tfRecFolder, obsFolder, noiseFilenames):
    ### IT IS TEST
    # if 640x480 => 256x256 w thrPerturbation=64
    squareSize=256
    thrPerturbation=64
    ### I DON'T NEED RANDOM POINTS, JUST CROP IMAGE CENTRE
    rndListObsSrc = random.sample(range(len(noiseFilenames)), num)
    rndListObsDst = random.sample(range(len(noiseFilenames)), num)
    rndListRowOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    rndListColOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    
    ### I HAVE THE TRANSFORMATION MATRIX, TRANSFORM IT TO POINT COORDINATES
    
    ### WRITE OUT POINT OUTPUTS FOR TEST
    
    for i in range(0, len(rndListRowOrig)):
        # read first random perturbation
        pRow = rndListRowOrig[i]
        pCol = rndListColOrig[i]
        imgTempOrig = img[pRow:pRow+squareSize, pCol:pCol+squareSize].copy()
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
        imgOb = cv2.imread(obsFolder+noiseFilenames[rndListObsSrc[i]], 0)
        # normalize obstacle image
        if imgOb.max()-imgOb.min() > 0:
            imgOb = (imgOb-imgOb.min())/(imgOb.max()-imgOb.min())
        imgOb = np.asarray(imgOb, np.float32)
        # Attach Obstacle
        ob_loc_row = random.sample(range(0,imgTempOrig.shape[0]-imgOb.shape[0]),1)[0]
        ob_loc_col = random.sample(range(0,imgTempOrig.shape[1]-imgOb.shape[1]),1)[0]
        imgTempOrig[ob_loc_row:ob_loc_row+imgOb.shape[0], ob_loc_col:ob_loc_col+imgOb.shape[1]] = imgOb
        if WRTIE:
            cv2.imwrite("img_"+str(imgOb.shape[0])+"_orig.jpg", imgTempOrig*255)
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
            cv2.imwrite("img_"+str(imgOb.shape[0])+"_pert.jpg", imgTempPert*255)
            WRTIE = False

        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.imshow('imageOrig',imgTempOrig)
        #cv2.waitKey(0)
        #cv2.imshow('imagePert',imgTempPert)
        #cv2.waitKey(0)
        perturb_writer(ID, i,
                       img, imgTempOrig, imgTempPert, H_AB, pOrig,
                       tfRecFolder)
    return

def normalize_image_toNP(img):
    if img.max()-img.min() > 0:
        img = (img-img.min())/(img.max()-img.min())
    img = np.asarray(img, np.float32)
    return img

def hMat_to_points(hmat):
    #deltaP = pTarget-pOrig
    return #pOrig, deltaP

def process_dataset(imgSrc, imgDst, hMat, ID, i, tfRecFolder):
    # normalize image & to NP
    imgSrc = normalize_image_toNP(imgSrc)
    imgDst = normalize_image_toNP(imgDst)
    # get transformation as points
    pOrig, deltaP = hMat_to_points(hMat)
    # Write tf record
    perturb_writer(ID, i,
               imgSrc, patchSrc, patchDst, deltaP, pOrig,
               tfRecFolder)
    return

def prepare_dataset(datasetType, readFolder, tfRecFolder):
    foldersList = [f for f in listdir(readFolder) if isdir(join(readFolder, f))]
    foldersList.sort()
    # ['100', '101', '102', '103']
    for folder in foldersList:
        print('    Processing folder: ', folder)
        filesList = [f for f in listdir(readFolder+folder) if isfile(join(readFolder+folder, f))]
        filesList.sort()
        # ['H1to2p', 'H1to3p', 'H1to4p', 'H1to5p', 'H1to6p', 'img1.ppm', 'img2.ppm', 'img3.ppm', 'img4.ppm', 'img5.ppm', 'img6.ppm']
        print('        Source image address: ', readFolder+folder+"/"+filesList[5])
        imgSrc = cv2.imread(readFolder+folder+"/"+filesList[5], 0)
        imgSrc = cv2.resize(imgSrc, (640,480))
        cv2.imshow('imgSrc',imgSrc)
        cv2.waitKey(0)
        for i in range(5):
            imgDst = cv2.imread(readFolder+folder+filesList[i+6], 0)
            imgDst = cv2.resize(imgDst, (640,480))
            cv2.imshow('imgDst', imgDst)
            cv2.waitKey(0)
            ### hMat = read the H matrix file (readFolder+folder+filesList[i])
            #process_dataset(imgSrc, imgDst, hMat, int(folder), i, tfRecFolder)
    


def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

testFolder = "../../Data/SIFTORIG/"
testtfRecordFLD = "../../Data/128_test_tfrecords/"
_set_folders(testtfRecordFLD)

print('Wrting test tfrecords...')
prepare_dataset("test", testFolder, testtfRecordFLD)
print('Done')